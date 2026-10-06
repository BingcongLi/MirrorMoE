"""
Expert-ablation importance analysis for FLAME-MoE / no-share models.

For each MoE layer and each expert (routed 0..N-1, plus the shared expert if
present), zero that expert's contribution and measure the LM loss on a small
batch. Produces a [num_moe_layers x (num_experts + 1)] matrix of Δloss
(ablated - baseline) and a heatmap.

Interpretation of Δloss:
  > 0  removing the expert HURTS  -> the expert is important / load-bearing
  ~ 0  removing it does nothing   -> redundant / prunable
  < 0  removing it IMPROVES loss   -> the expert was (net) harmful

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


# ── ablation state (mutated by the driver loop, read by hooks) ───────────────
ABLATE = {"layer_idx": None, "expert": None}   # expert: int (routed) | "shared" | None


def _extra_args(parser):
    g = parser.add_argument_group(title="expert-ablation")
    g.add_argument("--ablation-output", type=str, default="expert_ablation")
    g.add_argument("--ablation-num-seqs", type=int, default=8)
    g.add_argument("--ablation-seq-len", type=int, default=512)
    g.add_argument("--ablation-data-prefix", type=str, default="",
                   help="indexed-dataset prefix (path to a .bin/.idx without extension); "
                        "empty falls back to embedded text")
    return parser


def register_hooks(model):
    """Map decoder-layer index -> {router, shared, shared_fc2, dims} and hook them.

    ABLATE["expert"] can be:
      int          -> zero that routed expert's combine weight
      "shared"     -> zero the whole shared-expert output
      "shared_lo"  -> zero shared intermediate neurons [0:d/2]  (measures 1st half)
      "shared_hi"  -> zero shared intermediate neurons [d/2:d]  (measures 2nd half)
    """
    layers = {}   # idx -> dict
    for name, module in model.named_modules():
        if name.endswith("mlp.router"):
            idx = int(name.split(".layers.")[1].split(".")[0])
            layers.setdefault(idx, {})["router"] = module
            layers[idx]["num_experts"] = module.weight.shape[0]
        elif name.endswith("mlp.shared_experts"):
            idx = int(name.split(".layers.")[1].split(".")[0])
            layers.setdefault(idx, {})["shared"] = module
        elif name.endswith("mlp.shared_experts.linear_fc2"):
            idx = int(name.split(".layers.")[1].split(".")[0])
            layers.setdefault(idx, {})["shared_fc2"] = module
            # linear_fc2.weight is [hidden, intermediate] -> intermediate = shape[1]
            layers[idx]["shared_dim"] = module.weight.shape[1]

    def make_router_hook(idx):
        def hook(mod, inp, out):
            if ABLATE["layer_idx"] == idx and isinstance(ABLATE["expert"], int):
                scores, routing_map = out
                scores = scores.clone()
                scores[..., ABLATE["expert"]] = 0.0
                return scores, routing_map
        return hook

    def make_shared_out_hook(idx):
        def hook(mod, inp, out):
            if ABLATE["layer_idx"] == idx and ABLATE["expert"] == "shared":
                return torch.zeros_like(out)
        return hook

    def make_shared_fc2_prehook(idx, D):
        half = D // 2
        def hook(mod, inp):
            if ABLATE["layer_idx"] != idx:
                return None
            if ABLATE["expert"] not in ("shared_lo", "shared_hi"):
                return None
            x = inp[0].clone()
            if ABLATE["expert"] == "shared_lo":
                x[..., :half] = 0.0
            else:  # shared_hi
                x[..., half:] = 0.0
            return (x,) + tuple(inp[1:])
        return hook

    for idx, d in layers.items():
        if "router" in d:
            d["router"].register_forward_hook(make_router_hook(idx))
        if "shared" in d:
            d["shared"].register_forward_hook(make_shared_out_hook(idx))
        if "shared_fc2" in d:
            d["shared_fc2"].register_forward_pre_hook(
                make_shared_fc2_prehook(idx, d["shared_dim"]))
    return layers


def _read_indexed(prefix, need):
    """Read >= `need` tokens from a Megatron indexed dataset (val distribution)."""
    try:
        from megatron.core.datasets.indexed_dataset import IndexedDataset as _DS
    except Exception:
        from megatron.core.datasets.indexed_dataset import MMapIndexedDataset as _DS
    ds = _DS(prefix)
    toks = []
    i = 0
    ndoc = len(ds)
    # start a bit into the corpus (rough "held-out" slice, not the very first docs)
    start = max(0, ndoc // 2)
    while len(toks) < need and i < ndoc:
        doc = ds[(start + i) % ndoc]
        arr = doc[0] if isinstance(doc, tuple) else doc
        toks.extend(np.asarray(arr).reshape(-1).tolist())
        i += 1
    return toks


def make_batch(args):
    """Small batch of token ids from the tokenized .bin (fallback: embedded text)."""
    seq = args.ablation_seq_len
    n = args.ablation_num_seqs
    need = n * (seq + 1)

    ids = None
    if args.ablation_data_prefix:
        try:
            ids = _read_indexed(args.ablation_data_prefix, need)
            print_rank_0(f"[ablation] loaded {len(ids)} tokens from "
                         f"{args.ablation_data_prefix}")
        except Exception as ex:
            print_rank_0(f"[ablation] indexed read failed ({ex}); using embedded text")
            ids = None

    if ids is None or len(ids) < need:
        text = (
            "The history of science is the study of the development of the natural "
            "and social sciences. The mixture of experts is a machine learning "
            "technique in which multiple expert networks divide the problem space, "
            "and a gating network decides which experts to consult per input. " * 60
        )
        ids = get_tokenizer().tokenize(text)
        while len(ids) < need:
            ids = ids + ids

    ids = torch.tensor(ids[:need], dtype=torch.long).view(n, seq + 1)
    return ids[:, :-1].cuda(), ids[:, 1:].cuda()


@torch.no_grad()
def loss_on_batch(model, tokens, labels):
    b, s = tokens.shape
    position_ids = torch.arange(s, device=tokens.device).unsqueeze(0).expand(b, s)
    attn = torch.triu(torch.ones(s, s, device=tokens.device), diagonal=1).bool()
    attn = attn.view(1, 1, s, s)
    out = model(tokens, position_ids, attn, labels=labels)   # [b, s] per-token loss
    return out.float().mean().item()


def main():
    initialize_megatron(extra_args_provider=_extra_args,
                        args_defaults={"no_load_rng": True, "no_load_optim": True})
    args = get_args()

    model = megatron_get_model(model_provider, wrap_with_ddp=False)
    load_checkpoint(model, None, None)
    model = model[0] if isinstance(model, list) else model
    model.eval()

    layers = register_hooks(model)
    tokens, labels = make_batch(args)

    # baseline
    ABLATE["layer_idx"], ABLATE["expert"] = None, None
    base = loss_on_batch(model, tokens, labels)
    print_rank_0(f"[ablation] baseline loss = {base:.5f}")

    layer_ids = sorted(layers.keys())
    L = len(layer_ids)
    max_experts = max(layers[i].get("num_experts", 0) for i in layer_ids)
    has_shared = any("shared" in layers[i] for i in layer_ids)

    routed = np.full((L, max_experts), np.nan, dtype=np.float32)
    sh_full = np.full(L, np.nan, dtype=np.float32)   # zero whole shared
    sh_lo = np.full(L, np.nan, dtype=np.float32)     # zero [0:d/2]  (1st half importance)
    sh_hi = np.full(L, np.nan, dtype=np.float32)     # zero [d/2:d]  (2nd half importance)

    def dloss():
        return loss_on_batch(model, tokens, labels) - base

    for r, idx in enumerate(layer_ids):
        ne = layers[idx].get("num_experts", 0)
        for e in range(ne):
            ABLATE["layer_idx"], ABLATE["expert"] = idx, e
            routed[r, e] = dloss()
        if has_shared and "shared" in layers[idx]:
            for key, arr in (("shared", sh_full), ("shared_lo", sh_lo), ("shared_hi", sh_hi)):
                ABLATE["layer_idx"], ABLATE["expert"] = idx, key
                arr[r] = dloss()
        print_rank_0(f"[ablation] layer {idx}: routed {ne}"
                     f"{' + shared(full/lo/hi)' if has_shared else ''} done")

    ABLATE["layer_idx"], ABLATE["expert"] = None, None

    if torch.distributed.get_rank() == 0:
        out = args.ablation_output
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        np.savez(out + ".npz", routed=routed, shared_full=sh_full,
                 shared_lo=sh_lo, shared_hi=sh_hi, layer_ids=np.array(layer_ids),
                 baseline=base)

        print_rank_0("[ablation] layer | shared_full | share[0:d/2] | share[d/2:d] | routed_mean | full/(2*routed_mean)")
        lines = []
        for r, idx in enumerate(layer_ids):
            rmean = float(np.nanmean(routed[r])) if max_experts else float("nan")
            f_, lo_, hi_ = float(sh_full[r]), float(sh_lo[r]), float(sh_hi[r])
            norm = f_ / (2 * rmean) if (has_shared and rmean and rmean == rmean) else float("nan")
            line = (f"  L{idx:>2}  full={f_:+.4f}  lo={lo_:+.4f}  hi={hi_:+.4f}  "
                    f"routed_mean={rmean:+.4f}  full/(2*rmean)={norm:.1f}x")
            print_rank_0(line)
            lines.append(line)
        with open(out + "_meta.txt", "w") as f:
            f.write(f"baseline_loss {base}\nlayers {layer_ids}\n"
                    f"num_experts {max_experts}\nhas_shared {has_shared}\n\n")
            f.write("\n".join(lines) + "\n")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            rvmax = np.nanmax(np.abs(routed)) if max_experts else 1.0
            if has_shared:
                fig, (axr, axs) = plt.subplots(
                    1, 2, figsize=(max(7, max_experts * 0.13) + 3, L * 0.5 + 1),
                    gridspec_kw={"width_ratios": [max_experts, 4]})
            else:
                fig, axr = plt.subplots(figsize=(max(7, max_experts * 0.13), L * 0.5 + 1))

            im = axr.imshow(routed, aspect="auto", cmap="RdBu_r", vmin=-rvmax, vmax=rvmax)
            axr.set_xlabel("routed expert index")
            axr.set_ylabel("MoE layer")
            axr.set_yticks(range(L)); axr.set_yticklabels(layer_ids)
            axr.set_title("Routed experts: Δloss when zeroed")
            fig.colorbar(im, ax=axr, fraction=0.03)

            if has_shared:
                shared_mat = np.stack([sh_full, sh_lo, sh_hi], axis=1)  # [L, 3]
                svmax = np.nanmax(np.abs(shared_mat))
                ims = axs.imshow(shared_mat, aspect="auto", cmap="Reds", vmin=0, vmax=svmax)
                axs.set_xticks([0, 1, 2])
                axs.set_xticklabels(["full", "[0:d/2]", "[d/2:d]"], rotation=30, ha="right")
                axs.set_yticks(range(L)); axs.set_yticklabels([])
                axs.set_title("Shared expert Δloss")
                for r in range(L):
                    for c, v in enumerate((sh_full[r], sh_lo[r], sh_hi[r])):
                        axs.text(c, r, f"{v:.2f}", ha="center", va="center",
                                 fontsize=6, color="black")
                fig.colorbar(ims, ax=axs, fraction=0.05)

            fig.tight_layout()
            fig.savefig(out + ".png", dpi=150)
            print_rank_0(f"[ablation] saved {out}.npz and {out}.png")
        except Exception as ex:
            print_rank_0(f"[ablation] plot skipped: {ex}")


if __name__ == "__main__":
    main()
