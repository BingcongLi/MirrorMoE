"""
Shared-vs-routed energy split per layer, for a shared-expert (SE) model.

The MoE block output decomposes as h = h_s + h_r, where
  h_s = f_shared(x)                          (shared expert)
  h_r = sum_{e in top-k} p_e f_e(x)          (routed aggregate)

For each MoE layer we report, averaged over tokens (per-token ratio, then mean):
  frac_shared = mean_t ||h_s(x_t)||^2 / ||h(x_t)||^2
  frac_routed = mean_t ||h_r(x_t)||^2 / ||h(x_t)||^2
  frac_cross  = mean_t 2<h_s,h_r> / ||h(x_t)||^2      (so the three sum to 1)

Run via torchrun with the model's MODEL_ARGS + --load (see launcher .sh), 1 GPU
(NPROC=1, EP=1) so all experts are local. Requires a shared expert.
"""
import os
import numpy as np
import torch

from megatron.training import get_args, get_tokenizer, print_rank_0
from megatron.training.initialize import initialize_megatron
from megatron.training.checkpointing import load_checkpoint
from megatron.training import get_model as megatron_get_model
from pretrain_gpt import model_provider


CAP = {}
GRAB = {"on": False}


def _extra_args(parser):
    g = parser.add_argument_group(title="shared-split")
    g.add_argument("--split-output", type=str, default="shared_split")
    g.add_argument("--split-num-seqs", type=int, default=128)
    g.add_argument("--split-seq-len", type=int, default=1024)
    g.add_argument("--split-data-prefix", type=str, default="")
    g.add_argument("--split-tok-chunk", type=int, default=8192)
    return parser


def register(model):
    routers, experts, shared = {}, {}, {}
    for name, module in model.named_modules():
        if name.endswith("mlp.router"):
            idx = int(name.split(".layers.")[1].split(".")[0])
            routers[idx] = module
        elif name.endswith("mlp.experts"):
            idx = int(name.split(".layers.")[1].split(".")[0])
            experts[idx] = module
        elif name.endswith("mlp.shared_experts"):
            idx = int(name.split(".layers.")[1].split(".")[0])
            shared[idx] = module

    def make_hook(idx):
        def hook(mod, inp, out):
            if not GRAB["on"]:
                return
            x = inp[0].detach().reshape(-1, inp[0].shape[-1])
            probs, routing_map = out
            CAP[idx] = {
                "x": x.cpu(),                                     # bf16 native
                "probs": probs.detach().to(torch.float32).reshape(x.shape[0], -1).cpu(),
            }
        return hook

    for idx, r in routers.items():
        r.register_forward_hook(make_hook(idx))
    return routers, experts, shared


def _expert_forward(m, x):
    out = m(x)
    return out[0] if isinstance(out, tuple) else out


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
    seq, n = args.split_seq_len, args.split_num_seqs
    need = n * (seq + 1)
    ids = None
    if args.split_data_prefix:
        try:
            ids = _read_indexed(args.split_data_prefix, need)
            print_rank_0(f"[split] loaded {len(ids)} tokens from {args.split_data_prefix}")
        except Exception as ex:
            print_rank_0(f"[split] indexed read failed ({ex}); embedded text")
            ids = None
    if ids is None or len(ids) < need:
        text = ("The mixture of experts is a machine learning technique. " * 100)
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
    model(tokens, pos, attn)


def main():
    initialize_megatron(extra_args_provider=_extra_args,
                        args_defaults={"no_load_rng": True, "no_load_optim": True})
    args = get_args()

    model = megatron_get_model(model_provider, wrap_with_ddp=False)
    load_checkpoint(model, None, None)
    model = model[0] if isinstance(model, list) else model
    model.eval()
    torch.set_grad_enabled(False)

    routers, experts, shared = register(model)
    if len(shared) == 0:
        raise SystemExit("[split] no shared experts found — this analysis needs an SE model")

    tokens, _ = make_batch(args)
    GRAB["on"] = True
    forward_once(model, tokens)
    GRAB["on"] = False

    layer_ids = sorted(CAP.keys())
    L = len(layer_ids)
    print_rank_0(f"[split] {L} MoE layers, {tokens.numel()} tokens")

    frac_s = np.full(L, np.nan, dtype=np.float32)
    frac_r = np.full(L, np.nan, dtype=np.float32)
    frac_x = np.full(L, np.nan, dtype=np.float32)
    dev = torch.cuda.current_device()
    eps = 1e-6
    chunk = args.split_tok_chunk

    for r, idx in enumerate(layer_ids):
        if idx not in shared:
            continue
        xf = CAP[idx]["x"]
        pf = CAP[idx]["probs"]
        emod, smod = experts[idx], shared[idx]
        n_e = pf.shape[1]
        T = xf.shape[0]
        ss = torch.zeros((), device=dev)   # sum_t ||h_s||^2/||h||^2
        sr = torch.zeros((), device=dev)   # sum_t ||h_r||^2/||h||^2
        sx = torch.zeros((), device=dev)   # sum_t 2<h_s,h_r>/||h||^2
        cnt = 0

        for s in range(0, T, chunk):
            x = xf[s:s + chunk].to(dev)
            probs = pf[s:s + chunk].to(dev)

            h_r = torch.zeros(x.shape[0], x.shape[1], device=dev, dtype=torch.float32)
            for e in range(n_e):
                h_r += probs[:, e:e + 1] * _expert_forward(emod.local_experts[e], x).to(torch.float32)
            h_s = _expert_forward(smod, x).to(torch.float32)
            h = h_s + h_r

            h_sq = (h * h).sum(dim=-1) + eps
            ss += ((h_s * h_s).sum(dim=-1) / h_sq).sum()
            sr += ((h_r * h_r).sum(dim=-1) / h_sq).sum()
            sx += ((2.0 * (h_s * h_r).sum(dim=-1)) / h_sq).sum()
            cnt += x.shape[0]
            del h_r, h_s, h
            torch.cuda.empty_cache()

        frac_s[r] = float(ss / cnt)
        frac_r[r] = float(sr / cnt)
        frac_x[r] = float(sx / cnt)
        print_rank_0(f"[split] layer {idx}: shared={frac_s[r]:.4f}  routed={frac_r[r]:.4f}  "
                     f"cross={frac_x[r]:.4f}  (sum={frac_s[r]+frac_r[r]+frac_x[r]:.4f})")

    if torch.distributed.get_rank() != 0:
        return

    out = args.split_output
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez(out + ".npz", frac_shared=frac_s, frac_routed=frac_r,
             frac_cross=frac_x, layer_ids=np.array(layer_ids))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = np.arange(L)
        fig, ax = plt.subplots(figsize=(max(6, L * 0.5), 4))
        ax.plot(x, frac_s, "o-", label=r"$\|h_s\|^2/\|h\|^2$ (shared)")
        ax.plot(x, frac_r, "s-", label=r"$\|h_r\|^2/\|h\|^2$ (routed)")
        ax.plot(x, frac_x, "^--", color="gray", alpha=0.6, label="cross term")
        ax.set_xticks(x); ax.set_xticklabels(layer_ids)
        ax.set_xlabel("MoE layer"); ax.set_ylabel("energy fraction of ||h||^2")
        ax.set_title("Shared vs routed energy split"); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out + ".png", dpi=150)
        print_rank_0(f"[split] saved {out}.npz and {out}.png")
    except Exception as ex:
        print_rank_0(f"[split] plot skipped: {ex}")


if __name__ == "__main__":
    main()
