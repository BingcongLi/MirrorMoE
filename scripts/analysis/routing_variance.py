"""
Routing-induced variance: does the shared expert reduce the variance of the
MoE-layer output under routing stochasticity?

At EVAL, resample the routing M times (stochastic top-k ~ softmax(Wh), no
collapse). For each MoE layer capture the block output out(h) = shared + routed.
Across the M draws, per token, compute:
  V_abs = mean_m || out_m - mean ||^2                 (absolute routing variance)
  V_rel = V_abs / || mean ||^2                         (relative / CV^2 = SNR of noise)
  shared_ratio = || shared(h) || / || out(h) ||        (why: shared adds fixed magnitude)

Prediction: shared model -> lower V_rel at every layer (and lower V_abs if routed
experts also specialize to a lower-variance residual).

Run via torchrun with MODEL_ARGS + --load (see launcher .sh), 1 GPU.
"""
import os
import numpy as np
import torch

from megatron.training import get_args, get_tokenizer, print_rank_0
from megatron.training.initialize import initialize_megatron
from megatron.training.checkpointing import load_checkpoint
from megatron.training import get_model as megatron_get_model
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.moe_layer import MoELayer
from pretrain_gpt import model_provider

RV = {"on": False, "beta": None}
CAPTURE = {}          # layer_idx -> block output (current draw)
SHARED = {}           # layer_idx -> ||shared|| per token (deterministic; captured once)

_orig_routing = TopKRouter.routing


def _sample_routing(self, logits):
    if not RV["on"]:
        return _orig_routing(self, logits)
    E = self.config.num_moe_experts
    logits = logits.view(-1, E)
    p = torch.softmax(logits, dim=-1, dtype=torch.float32).to(logits.dtype)
    beta = RV["beta"]
    if beta is None:
        # plain sampling-without-replacement (== Gumbel-top-k with beta=1)
        sel = torch.multinomial(p.float(), self.topk, replacement=False)
    else:
        # Gumbel-top-k with noise scale beta: top-k of (logits + beta * Gumbel(0,1)).
        # beta=1 reproduces the multinomial path; smaller beta -> sharper (less
        # routing perturbation), larger beta -> noisier.
        u = torch.rand(logits.shape[0], E, device=logits.device,
                       dtype=torch.float32).clamp_(1e-9, 1.0)
        g = -torch.log(-torch.log(u))
        sel = (logits.float() + beta * g).topk(self.topk, dim=-1).indices
    routing_map = torch.zeros_like(p, dtype=torch.bool).scatter_(1, sel, True)
    return p * routing_map, routing_map


TopKRouter.routing = _sample_routing


def register(model):
    blocks = [m for m in model.modules() if isinstance(m, MoELayer)]
    for i, b in enumerate(blocks):
        b._rv_idx = i

        def moe_hook(idx):
            def hook(mod, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                CAPTURE[idx] = o.detach()
            return hook
        b.register_forward_hook(moe_hook(i))

        sh = getattr(b, "shared_experts", None)
        if sh is not None:
            def sh_hook(idx):
                def hook(mod, inp, out):
                    o = out[0] if isinstance(out, tuple) else out
                    SHARED[idx] = o.detach().float().reshape(-1, o.shape[-1]).norm(dim=-1)
                return hook
            sh.register_forward_hook(sh_hook(i))
    return blocks


def _extra_args(parser):
    g = parser.add_argument_group("routing-variance")
    g.add_argument("--rv-output", type=str, default="routing_variance")
    g.add_argument("--rv-num-seqs", type=int, default=16)
    g.add_argument("--rv-seq-len", type=int, default=512)
    g.add_argument("--rv-draws", type=int, default=24)
    g.add_argument("--rv-micro", type=int, default=4,
                   help="sequences per forward pass (micro-batch); keeps large batches in memory")
    g.add_argument("--rv-data-prefix", type=str, default="")
    g.add_argument("--rv-gumbel-beta", type=float, default=None,
                   help="resample routing via Gumbel-top-k with this noise scale beta "
                        "(top-k of logits + beta*Gumbel). Default None == plain sampling "
                        "(beta=1). Set e.g. 0.5 (sharper) or 2 (noisier).")
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
    seq, n = args.rv_seq_len, args.rv_num_seqs
    need = n * (seq + 1)
    ids = None
    if args.rv_data_prefix:
        try:
            ids = _read_indexed(args.rv_data_prefix, need)
            print_rank_0(f"[rv] loaded {len(ids)} tokens from {args.rv_data_prefix}")
        except Exception as ex:
            print_rank_0(f"[rv] indexed read failed ({ex}); embedded text")
            ids = None
    if ids is None or len(ids) < need:
        text = ("The mixture of experts is a machine learning technique. " * 120)
        ids = get_tokenizer().tokenize(text)
        while len(ids) < need:
            ids = ids + ids
    ids = torch.tensor(ids[:need], dtype=torch.long).view(n, seq + 1)
    return ids[:, :-1].cuda(), ids[:, 1:].cuda()


@torch.no_grad()
def forward_once(model, tokens):
    b, s = tokens.shape
    pos = torch.arange(s, device=tokens.device).unsqueeze(0).expand(b, s)
    attn = torch.triu(torch.ones(s, s, device=tokens.device), diagonal=1).bool().view(1, 1, s, s)
    model(tokens, pos, attn)   # no labels; we only want the MoE-layer captures


def main():
    initialize_megatron(extra_args_provider=_extra_args,
                        args_defaults={"no_load_rng": True, "no_load_optim": True})
    args = get_args()

    model = megatron_get_model(model_provider, wrap_with_ddp=False)
    load_checkpoint(model, None, None)
    model = model[0] if isinstance(model, list) else model
    model.eval()

    blocks = register(model)
    L = len(blocks)
    tokens, _ = make_batch(args)
    N, S = tokens.shape
    Ttot = N * S
    mb = args.rv_micro
    RV["beta"] = args.rv_gumbel_beta
    print_rank_0(f"[rv] {L} MoE layers, {Ttot} tokens ({N}x{S}), "
                 f"{args.rv_draws} draws, micro={mb} seqs/fwd, "
                 f"beta={args.rv_gumbel_beta}")

    # streaming accumulators per layer over ALL tokens: sum of out, sum of ||out||^2.
    # forward is micro-batched (mb sequences per pass) so large batches don't OOM;
    # each token maps to a fixed accumulator slot across draws -> per-token variance.
    acc_sum, acc_sq = {}, {}
    sh_full = {}          # idx -> [Ttot] shared-expert norm per token (deterministic)

    RV["on"] = True
    for m in range(args.rv_draws):
        for c in range(0, N, mb):
            CAPTURE.clear()
            forward_once(model, tokens[c:c + mb])
            off = c * S
            for idx, o in CAPTURE.items():
                o = o.float().reshape(-1, o.shape[-1])       # [chunk_tokens, hidden]
                if idx not in acc_sum:
                    acc_sum[idx] = torch.zeros(Ttot, o.shape[-1], device=o.device)
                    acc_sq[idx] = torch.zeros(Ttot, device=o.device)
                acc_sum[idx][off:off + o.shape[0]] += o
                acc_sq[idx][off:off + o.shape[0]] += (o * o).sum(-1)
            # shared norm is deterministic; write this chunk's tokens into the
            # full-token buffer at their offset (hook only holds the last chunk).
            for idx, sn in SHARED.items():
                if idx not in sh_full:
                    sh_full[idx] = torch.zeros(Ttot, device=sn.device)
                sh_full[idx][off:off + sn.shape[0]] = sn
        if (m + 1) % 8 == 0:
            print_rank_0(f"[rv]   draw {m+1}/{args.rv_draws}")
    RV["on"] = False

    Mn = args.rv_draws
    v_abs, v_rel, sh_ratio = np.full(L, np.nan), np.full(L, np.nan), np.full(L, np.nan)
    for idx in sorted(acc_sum.keys()):
        mean = acc_sum[idx] / Mn                            # [tokens, hidden]
        mean_sq_norm = (mean * mean).sum(-1)               # [tokens]
        e_sq = acc_sq[idx] / Mn                             # [tokens]
        va = (e_sq - mean_sq_norm).clamp_min(0)            # per-token abs variance
        vr = va / (mean_sq_norm + 1e-9)
        v_abs[idx] = float(va.mean())
        v_rel[idx] = float(vr.mean())
        if idx in sh_full:
            sh_ratio[idx] = float((sh_full[idx] / (mean.norm(dim=-1) + 1e-9)).mean())

    print_rank_0("[rv] layer |   V_abs   |   V_rel   | shared/out")
    for idx in range(L):
        print_rank_0(f"[rv]  {idx:>3}  {v_abs[idx]:.4e}  {v_rel[idx]:.4e}  "
                     f"{sh_ratio[idx] if not np.isnan(sh_ratio[idx]) else float('nan'):.3f}")

    if torch.distributed.get_rank() == 0:
        out = args.rv_output
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        np.savez(out + ".npz", v_abs=v_abs, v_rel=v_rel, shared_ratio=sh_ratio)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(11, 4))
            x = np.arange(L)
            ax[0].plot(x, v_rel, "o-", label="V_rel (CV^2)")
            ax[0].set_xlabel("MoE layer"); ax[0].set_ylabel("relative routing variance")
            ax[0].set_title("Routing-induced relative variance"); ax[0].legend()
            ax[1].plot(x, v_abs, "s-", color="tab:orange")
            ax[1].set_xlabel("MoE layer"); ax[1].set_ylabel("absolute routing variance")
            ax[1].set_title("Absolute routing variance")
            if not np.all(np.isnan(sh_ratio)):
                ax2 = ax[0].twinx()
                ax2.plot(x, sh_ratio, "^--", color="gray", alpha=0.6, label="shared/out")
                ax2.set_ylabel("shared magnitude / out")
            fig.tight_layout(); fig.savefig(out + ".png", dpi=150)
            print_rank_0(f"[rv] saved {out}.npz and {out}.png")
        except Exception as ex:
            print_rank_0(f"[rv] plot skipped: {ex}")


if __name__ == "__main__":
    main()
