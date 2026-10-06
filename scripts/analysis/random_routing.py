"""
Routing-perturbation ablation for FLAME-MoE / no-share models (Megatron).

At EVAL ONLY, replace each MoE layer's deterministic top-k with:
  uniform : k experts UNIFORMLY at random, uniform 1/k combine (removes ALL routing)
  sample  : k experts ~ multinomial(softmax(Wh)) without replacement, affinity combine
            (stochastic top-k: keeps router preference, removes only determinism)

Reports, for BOTH modes: baseline PPL, random-ALL PPL, per-layer PPL curve.

Run via torchrun with the model's MODEL_ARGS + --load (see launcher .sh), 1 GPU.
"""
import os
import numpy as np
import torch

from megatron.training import get_args, get_tokenizer, print_rank_0
from megatron.training.initialize import initialize_megatron
from megatron.training.checkpointing import load_checkpoint
from megatron.training import get_model as megatron_get_model
from megatron.core.transformer.moe.router import TopKRouter
from pretrain_gpt import model_provider

STATE = {"mode": None, "layers": set(), "beta": None}   # mode: None|"uniform"|"sample"

_orig_routing = TopKRouter.routing


def _perturbed_routing(self, logits):
    if STATE["mode"] is None or self.layer_number not in STATE["layers"]:
        return _orig_routing(self, logits)
    E = self.config.num_moe_experts
    logits = logits.view(-1, E)
    p = torch.softmax(logits, dim=-1, dtype=torch.float32).to(logits.dtype)
    T, k = p.shape[0], self.topk
    if STATE["mode"] == "uniform":
        sel = torch.rand(T, E, device=p.device).topk(k, dim=-1).indices
        routing_map = torch.zeros_like(p, dtype=torch.bool).scatter_(1, sel, True)
        probs = torch.zeros_like(p).scatter_(1, sel, 1.0 / k)
    else:  # sample: stochastic top-k from the router distribution
        beta = STATE["beta"]
        if beta is None:
            # plain sampling-without-replacement (== Gumbel-top-k with beta=1)
            sel = torch.multinomial(p.float(), k, replacement=False)
        else:
            # Gumbel-top-k with noise scale beta: top-k of (logits + beta * Gumbel(0,1)).
            # beta=1 reproduces the multinomial path; smaller beta -> sharper (closer
            # to deterministic top-k), larger beta -> noisier.
            u = torch.rand(T, E, device=logits.device, dtype=torch.float32).clamp_(1e-9, 1.0)
            g = -torch.log(-torch.log(u))
            sel = (logits.float() + beta * g).topk(k, dim=-1).indices
        routing_map = torch.zeros_like(p, dtype=torch.bool).scatter_(1, sel, True)
        probs = p * routing_map                      # affinity combine
    return probs, routing_map


TopKRouter.routing = _perturbed_routing


def _extra_args(parser):
    g = parser.add_argument_group(title="random-routing")
    g.add_argument("--rr-output", type=str, default="random_routing")
    g.add_argument("--rr-num-seqs", type=int, default=32)
    g.add_argument("--rr-seq-len", type=int, default=1024)
    g.add_argument("--rr-data-prefix", type=str, default="")
    g.add_argument("--rr-repeats", type=int, default=3)
    g.add_argument("--rr-gumbel-beta", type=float, default=None,
                   help="sample mode: use Gumbel-top-k with this noise scale beta "
                        "(top-k of logits + beta*Gumbel). Default None == plain "
                        "sampling (beta=1). Set e.g. 0.5 for sharper sampling.")
    return parser


def _read_indexed(prefix, need):
    try:
        from megatron.core.datasets.indexed_dataset import IndexedDataset as _DS
    except Exception:
        from megatron.core.datasets.indexed_dataset import MMapIndexedDataset as _DS
    ds = _DS(prefix)
    toks, i, ndoc = [], 0, len(ds)
    start = max(0, ndoc // 2)
    while len(toks) < need and i < ndoc:
        doc = ds[(start + i) % ndoc]
        arr = doc[0] if isinstance(doc, tuple) else doc
        toks.extend(np.asarray(arr).reshape(-1).tolist())
        i += 1
    return toks


def make_batch(args):
    seq, n = args.rr_seq_len, args.rr_num_seqs
    need = n * (seq + 1)
    ids = None
    if args.rr_data_prefix:
        try:
            ids = _read_indexed(args.rr_data_prefix, need)
            print_rank_0(f"[rr] loaded {len(ids)} tokens from {args.rr_data_prefix}")
        except Exception as ex:
            print_rank_0(f"[rr] indexed read failed ({ex}); using embedded text")
            ids = None
    if ids is None or len(ids) < need:
        text = ("The mixture of experts is a machine learning technique in which "
                "multiple expert networks divide the problem space. " * 80)
        ids = get_tokenizer().tokenize(text)
        while len(ids) < need:
            ids = ids + ids
    ids = torch.tensor(ids[:need], dtype=torch.long).view(n, seq + 1)
    return ids[:, :-1].cuda(), ids[:, 1:].cuda()


@torch.no_grad()
def ppl(model, tokens, labels, micro=4):
    tot_loss, tot_tok = 0.0, 0
    for i in range(0, tokens.shape[0], micro):
        t, l = tokens[i:i + micro], labels[i:i + micro]
        b, s = t.shape
        pos = torch.arange(s, device=t.device).unsqueeze(0).expand(b, s)
        attn = torch.triu(torch.ones(s, s, device=t.device), diagonal=1).bool().view(1, 1, s, s)
        loss = model(t, pos, attn, labels=l)          # [b, s] per-token
        tot_loss += loss.float().sum().item()
        tot_tok += l.numel()
    mean = tot_loss / tot_tok
    return float(np.exp(mean)), mean


def main():
    initialize_megatron(extra_args_provider=_extra_args,
                        args_defaults={"no_load_rng": True, "no_load_optim": True})
    args = get_args()

    model = megatron_get_model(model_provider, wrap_with_ddp=False)
    load_checkpoint(model, None, None)
    model = model[0] if isinstance(model, list) else model
    model.eval()

    moe_layers = sorted({m.layer_number for _, m in model.named_modules()
                         if isinstance(m, TopKRouter) and m.layer_number is not None})
    tokens, labels = make_batch(args)

    def avg_ppl():
        return float(np.mean([ppl(model, tokens, labels)[0] for _ in range(args.rr_repeats)]))

    STATE["mode"], STATE["layers"] = None, set()
    STATE["beta"] = args.rr_gumbel_beta
    base_ppl, base_loss = ppl(model, tokens, labels)
    print_rank_0(f"[rr] baseline PPL = {base_ppl:.4f} (loss {base_loss:.4f})"
                 + (f"  [sample beta={args.rr_gumbel_beta}]"
                    if args.rr_gumbel_beta is not None else ""))

    results = {}
    for mode in ("uniform", "sample"):
        STATE["mode"] = mode
        STATE["layers"] = set(moe_layers)
        all_ppl = avg_ppl()
        print_rank_0(f"[rr] [{mode}] random-ALL PPL = {all_ppl:.4f}  "
                     f"increase = {all_ppl - base_ppl:+.4f} ({100*(all_ppl/base_ppl-1):+.1f}%)")
        per = []
        for L in moe_layers:
            STATE["layers"] = {L}
            p = avg_ppl()
            per.append(p)
            print_rank_0(f"[rr] [{mode}]   layer {L}: PPL {p:.4f}  Δ {p - base_ppl:+.4f}")
        results[mode] = np.array(per)
    STATE["mode"], STATE["layers"] = None, set()

    if torch.distributed.get_rank() == 0:
        out = args.rr_output
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        np.savez(out + ".npz", base_ppl=base_ppl, layers=np.array(moe_layers),
                 uniform_per=results["uniform"], sample_per=results["sample"])
        with open(out + "_summary.txt", "w") as f:
            f.write(f"baseline_ppl {base_ppl}\nlayers {moe_layers}\n")
            for L, u, s in zip(moe_layers, results["uniform"], results["sample"]):
                f.write(f"layer {L}  uniform {u}  sample {s}\n")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            x = np.arange(len(moe_layers))
            fig, ax = plt.subplots(figsize=(max(6, len(moe_layers) * 0.4), 4))
            ax.bar(x - 0.2, results["uniform"] - base_ppl, width=0.4, label="uniform")
            ax.bar(x + 0.2, results["sample"] - base_ppl, width=0.4, label="sample (stochastic top-k)")
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xticks(x); ax.set_xticklabels(moe_layers)
            ax.set_xlabel("MoE layer"); ax.set_ylabel("PPL increase (randomize one layer)")
            ax.set_title(f"Routing perturbation per layer (baseline PPL {base_ppl:.2f})")
            ax.legend()
            fig.tight_layout(); fig.savefig(out + ".png", dpi=150)
            print_rank_0(f"[rr] saved {out}.npz and {out}.png")
        except Exception as ex:
            print_rank_0(f"[rr] plot skipped: {ex}")


if __name__ == "__main__":
    main()
