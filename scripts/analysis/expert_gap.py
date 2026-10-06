"""
Expert-vs-aggregate feature gap for FLAME-MoE / no-share models.

For each MoE layer l and each routed expert e, measure how far that expert's own
feature f_e(x_t) sits from the top-k aggregated feature h(x_t), relative to
||h||, averaged over tokens:

    gap(l, e) = mean_t  || f_e(x_t) - h(x_t) ||^2  /  || h(x_t) ||^2

    h(x_t) = sum_{e' in top-k(t)} p_t^{e'} * f_{e'}(x_t)     (routed aggregate)

where x_t is the expert-block input (post pre-mlp-norm), f_e is the expert FFN,
and p_t^{e'} are the router combine weights the model actually uses (affinity,
masked to the selected top-k).

Only the ROUTED experts are considered — both in the aggregate h (the "top-k
aggregation") and in the per-expert gap. The shared expert is ignored entirely,
so the SE and no-SE numbers are directly comparable.

Run via torchrun with the model's MODEL_ARGS + --load (see the launcher .sh),
on a single GPU (NPROC=1, EP=1) so all experts are local.
"""
import os
import numpy as np
import torch

from megatron.training import get_args, get_tokenizer, print_rank_0
from megatron.training.initialize import initialize_megatron
from megatron.training.checkpointing import load_checkpoint
from megatron.training import get_model as megatron_get_model
from pretrain_gpt import model_provider


# layer_idx -> {"x": [T,H], "probs": [T,N], "routing_map": [T,N]} captured in one fwd
CAP = {}
GRAB = {"on": False}


def _extra_args(parser):
    g = parser.add_argument_group(title="expert-gap")
    g.add_argument("--gap-output", type=str, default="expert_gap")
    g.add_argument("--gap-num-seqs", type=int, default=16)
    g.add_argument("--gap-seq-len", type=int, default=1024)
    g.add_argument("--gap-data-prefix", type=str, default="",
                   help="indexed-dataset prefix (path without .bin/.idx); "
                        "empty falls back to embedded text")
    g.add_argument("--gap-selected-only", action="store_true",
                   help="average each expert's gap only over tokens where it was "
                        "selected (default: over all tokens)")
    g.add_argument("--gap-tok-chunk", type=int, default=8192,
                   help="tokens processed per chunk in the per-expert loop; lower "
                        "it if you OOM with large batches")
    g.add_argument("--gap-include-shared", action="store_true",
                   help="fold the shared expert into the aggregate h, so the gap is "
                        "measured against the full block feature (routed + shared) "
                        "the layer eventually produces; no-op for no-share models")
    g.add_argument("--gap-no-normalize", action="store_true",
                   help="report raw squared error ||f_e - h||^2 without dividing by "
                        "||h||^2 (default is the normalized ratio)")
    return parser


def register(model):
    """Hook each router to capture (x, combine-probs, routing_map) per MoE layer.

    The router input equals the expert-block input (both are the post
    pre-mlp-norm hidden state), so we can feed the captured x straight through
    each local expert to get f_e(x).
    """
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
            x = inp[0].detach()
            x = x.reshape(-1, x.shape[-1])          # [T, H], token order matches probs
            probs, routing_map = out                # combine weights (masked), bool map
            CAP[idx] = {
                # keep x in the model's native dtype (bf16) so TE experts accept it;
                # gap/norm math is promoted to fp32 downstream.
                "x": x.cpu(),
                "probs": probs.detach().to(torch.float32).reshape(x.shape[0], -1).cpu(),
                "routing_map": routing_map.detach().reshape(x.shape[0], -1).cpu(),
            }
        return hook

    for idx, r in routers.items():
        r.register_forward_hook(make_hook(idx))
    return routers, experts, shared


def _expert_forward(expert_module, x):
    """f_e(x): run a single local expert MLP on [T, H] -> [T, H]."""
    out = expert_module(x)
    out = out[0] if isinstance(out, tuple) else out
    return out


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
    seq, n = args.gap_seq_len, args.gap_num_seqs
    need = n * (seq + 1)
    ids = None
    if args.gap_data_prefix:
        try:
            ids = _read_indexed(args.gap_data_prefix, need)
            print_rank_0(f"[gap] loaded {len(ids)} tokens from {args.gap_data_prefix}")
        except Exception as ex:
            print_rank_0(f"[gap] indexed read failed ({ex}); embedded text")
            ids = None
    if ids is None or len(ids) < need:
        text = ("The mixture of experts is a machine learning technique in which "
                "multiple expert networks divide the problem space. " * 100)
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
    torch.set_grad_enabled(False)   # inference-only: covers the per-expert compute too

    routers, experts, shared = register(model)
    tokens, _ = make_batch(args)

    GRAB["on"] = True
    forward_once(model, tokens)
    GRAB["on"] = False

    layer_ids = sorted(CAP.keys())
    L = len(layer_ids)
    N = max(CAP[i]["probs"].shape[1] for i in layer_ids)
    print_rank_0(f"[gap] {L} MoE layers, {N} experts, "
                 f"{tokens.numel()} tokens, "
                 f"selected_only={args.gap_selected_only}")

    gap = np.full((L, N), np.nan, dtype=np.float32)
    dev = torch.cuda.current_device()
    eps = 1e-6
    chunk = args.gap_tok_chunk

    for r, idx in enumerate(layer_ids):
        xf = CAP[idx]["x"]                             # cpu bf16 [T, H]
        pf = CAP[idx]["probs"]                         # cpu fp32 [T, N] combine weights
        mf = CAP[idx]["routing_map"]                   # cpu bool [T, N]
        emod = experts[idx]
        n_e = pf.shape[1]
        T = xf.shape[0]

        # accumulate sum of per-token gap ratios (and token counts) over chunks,
        # so arbitrarily large batches never hold all f_e for all tokens at once.
        gsum = torch.zeros(n_e, device=dev)
        gcnt = torch.zeros(n_e, device=dev)

        for s in range(0, T, chunk):
            x = xf[s:s + chunk].to(dev)                # [c, H] bf16 (native, for TE)
            probs = pf[s:s + chunk].to(dev)
            rmap = mf[s:s + chunk].to(dev)

            # pass 1 (this chunk): routed aggregate h = sum_e probs_e * f_e(x)
            h = torch.zeros(x.shape[0], x.shape[1], device=dev, dtype=torch.float32)
            fe_cache = [None] * n_e
            for e in range(n_e):
                fe = _expert_forward(emod.local_experts[e], x).to(torch.float32)
                fe_cache[e] = fe
                h += probs[:, e:e + 1] * fe
            # optionally fold in the shared expert -> h is the full block feature
            if args.gap_include_shared and idx in shared:
                h += _expert_forward(shared[idx], x).to(torch.float32)
            hsq = (h * h).sum(dim=-1)                   # [c]  = ||h||^2

            # pass 2 (this chunk): per-expert squared gap; normalized by ||h||^2
            # unless --gap-no-normalize, in which case raw ||f_e - h||^2.
            for e in range(n_e):
                sq = ((fe_cache[e] - h) ** 2).sum(dim=-1)                   # ||f_e - h||^2
                diff = sq if args.gap_no_normalize else sq / (hsq + eps)    # [c]
                if args.gap_selected_only:
                    m = rmap[:, e]
                    gsum[e] += diff[m].sum(); gcnt[e] += m.sum()
                else:
                    gsum[e] += diff.sum(); gcnt[e] += diff.numel()
            del fe_cache, h
            torch.cuda.empty_cache()

        vals = gsum / gcnt.clamp_min(1)
        vals[gcnt == 0] = float("nan")                 # experts never selected (selected-only)
        gap[r, :n_e] = vals.float().cpu().numpy()
        print_rank_0(f"[gap] layer {idx}: mean gap = {np.nanmean(gap[r]):.4f} "
                     f"(over {int(T)} tokens)")

    if torch.distributed.get_rank() != 0:
        return

    out = args.gap_output
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez(out + ".npz", gap=gap, layer_ids=np.array(layer_ids))

    print_rank_0("[gap] layer | mean gap | min | max")
    for r, idx in enumerate(layer_ids):
        print_rank_0(f"[gap]  {idx:>3}  {np.nanmean(gap[r]):.4f}  "
                     f"{np.nanmin(gap[r]):.4f}  {np.nanmax(gap[r]):.4f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (axh, axl) = plt.subplots(
            1, 2, figsize=(max(8, N * 0.14) + 4, L * 0.45 + 1.5),
            gridspec_kw={"width_ratios": [N, 5]})

        im = axh.imshow(gap, aspect="auto", cmap="viridis")
        axh.set_xlabel("routed expert index"); axh.set_ylabel("MoE layer")
        axh.set_yticks(range(L)); axh.set_yticklabels(layer_ids)
        axh.set_title((r"$\|f_e - h\|^2$" if args.gap_no_normalize
                       else r"$\|f_e - h\|^2 / \|h\|^2$") + " per expert")
        fig.colorbar(im, ax=axh, fraction=0.03)

        mean_gap = np.nanmean(gap, axis=1)
        axl.plot(mean_gap, range(L), "o-", label="routed mean")
        axl.set_ylim(L - 0.5, -0.5); axl.set_yticks(range(L)); axl.set_yticklabels([])
        axl.set_xlabel("gap"); axl.set_title("per-layer gap"); axl.legend()
        axl.grid(alpha=0.3)

        fig.tight_layout(); fig.savefig(out + ".png", dpi=150)
        print_rank_0(f"[gap] saved {out}.npz and {out}.png")
    except Exception as ex:
        print_rank_0(f"[gap] plot skipped: {ex}")


if __name__ == "__main__":
    main()
