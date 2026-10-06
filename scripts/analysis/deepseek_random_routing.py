"""
Routing-perturbation ablation on DeepSeekMoE-16B (or any DeepSeek MoE HF ckpt).

At EVAL ONLY, replace each MoE layer's deterministic top-k selection with one of:
  uniform : k experts drawn UNIFORMLY at random (uniform 1/k combine)
            -> removes ALL learned routing.
  sample  : k experts sampled WITHOUT replacement from softmax(router_logits)
            (affinity-normalized combine, like top-k)
            -> keeps router preferences, removes only the determinism (stochastic top-k).

Reports, for BOTH modes: baseline PPL, random-ALL PPL, per-layer PPL curve.
Plus shared-expert ablation PPL (zero shared expert output).

Usage:
  python scripts/analysis/deepseek_random_routing.py \
      --model deepseek-ai/deepseek-moe-16b-base --num-seqs 32 --seq-len 2048 \
      --repeats 3 --out logs-eval/dsmoe

Needs: transformers==4.40, datasets, accelerate, torch. ~32GB bf16 -> 1x H100.
"""
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F

# shim: DeepSeek remote code imports is_torch_fx_available from import_utils
import transformers.utils.import_utils as _iu
if not hasattr(_iu, "is_torch_fx_available"):
    try:
        from transformers.utils import is_torch_fx_available as _f
        _iu.is_torch_fx_available = _f
    except Exception:
        _iu.is_torch_fx_available = lambda: False

from transformers import AutoModelForCausalLM, AutoTokenizer

# mode: None (off) | "uniform" | "sample"; random_layers: set of MoE-block indices
STATE = {"mode": None, "random_layers": set(), "ablate_shared": False}


def find_moe_blocks(model):
    return [m for m in model.modules() if hasattr(m, "gate") and hasattr(m, "experts")]


def hook_gates(blocks, n_routed):
    def make(idx):
        def hook(mod, inp, out):
            if STATE["mode"] is None or idx not in STATE["random_layers"]:
                return None
            topk_idx, topk_weight = out[0], out[1]
            rest = tuple(out[2:]) if len(out) > 2 else ()
            B, k = topk_idx.shape
            if STATE["mode"] == "uniform":
                new_idx = torch.rand(B, n_routed, device=topk_idx.device).topk(k, dim=-1).indices
                new_w = torch.full_like(topk_weight, 1.0 / k)
            else:  # "sample": multinomial from softmax(router logits)
                hidden = inp[0].reshape(-1, inp[0].shape[-1])                 # -> [tokens, hidden]
                logits = F.linear(hidden.to(mod.weight.dtype), mod.weight)   # [tokens, n_routed]
                p = torch.softmax(logits.float(), dim=-1)
                new_idx = torch.multinomial(p, k, replacement=False)         # [tokens, k]
                w = p.gather(1, new_idx)
                new_w = (w / (w.sum(-1, keepdim=True) + 1e-9)).to(topk_weight.dtype)
            return (new_idx.to(topk_idx.dtype), new_w) + rest
        return hook
    for idx, b in enumerate(blocks):
        b.gate.register_forward_hook(make(idx))


def hook_shared(blocks):
    def hook(mod, inp, out):
        return torch.zeros_like(out) if STATE["ablate_shared"] else None
    n = 0
    for b in blocks:
        if getattr(b, "shared_experts", None) is not None:
            b.shared_experts.register_forward_hook(hook)
            n += 1
    return n


def load_data(tok, num_seqs, seq_len):
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
        text = "\n\n".join(ds["text"])
    except Exception as ex:
        print(f"[dsmoe] wikitext load failed ({ex}); using embedded text")
        text = ("The mixture of experts is a machine learning technique in which "
                "multiple expert networks divide the problem space. " * 200)
    ids = tok(text, return_tensors="pt").input_ids[0]
    need = num_seqs * (seq_len + 1)
    while ids.numel() < need:
        ids = torch.cat([ids, ids])
    return ids[:need].view(num_seqs, seq_len + 1)


@torch.no_grad()
def ppl(model, ids, micro=2):
    tot_loss, tot_tok = 0.0, 0
    for i in range(0, ids.shape[0], micro):
        chunk = ids[i:i + micro].to(model.device)
        inp, lbl = chunk[:, :-1], chunk[:, 1:]
        logits = model(inp).logits.float()
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               lbl.reshape(-1), reduction="sum")
        tot_loss += loss.item()
        tot_tok += lbl.numel()
    mean = tot_loss / tot_tok
    return float(np.exp(mean)), mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/deepseek-moe-16b-base")
    ap.add_argument("--num-seqs", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--micro", type=int, default=2)
    ap.add_argument("--out", default="logs-eval/dsmoe")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    n_routed = model.config.n_routed_experts
    topk = model.config.num_experts_per_tok
    n_shared = getattr(model.config, "n_shared_experts", None)
    blocks = find_moe_blocks(model)
    L = len(blocks)
    print(f"[dsmoe] {args.model}: n_routed={n_routed} top_k={topk} n_shared={n_shared} moe_layers={L}")

    hook_gates(blocks, n_routed)
    n_sh = hook_shared(blocks)

    ids = load_data(tok, args.num_seqs, args.seq_len)

    def avg_ppl():
        return float(np.mean([ppl(model, ids, args.micro)[0] for _ in range(args.repeats)]))

    STATE["mode"], STATE["ablate_shared"] = None, False
    base_ppl, base_loss = ppl(model, ids, args.micro)
    print(f"[dsmoe] baseline PPL = {base_ppl:.4f} (loss {base_loss:.4f})")

    results = {}
    for mode in ("uniform", "sample"):
        STATE["mode"] = mode
        STATE["random_layers"] = set(range(L))
        all_ppl = avg_ppl()
        print(f"[dsmoe] [{mode}] random-ALL PPL = {all_ppl:.4f}  "
              f"increase = {all_ppl - base_ppl:+.4f} ({100*(all_ppl/base_ppl-1):+.1f}%)")
        per = []
        for i in range(L):
            STATE["random_layers"] = {i}
            p = avg_ppl()
            per.append(p)
            print(f"[dsmoe] [{mode}]   layer {i}: PPL = {p:.4f}  Δ = {p - base_ppl:+.4f}")
        results[mode] = {"all": all_ppl, "per": np.array(per)}
    STATE["mode"], STATE["random_layers"] = None, set()

    sh_ppl = float("nan")
    if n_sh:
        STATE["ablate_shared"] = True
        sh_ppl, _ = ppl(model, ids, args.micro)
        STATE["ablate_shared"] = False
        print(f"[dsmoe] shared-ablated PPL = {sh_ppl:.4f}  "
              f"increase = {sh_ppl - base_ppl:+.4f} ({100*(sh_ppl/base_ppl-1):+.1f}%)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out + ".npz", base_ppl=base_ppl,
             uniform_all=results["uniform"]["all"], uniform_per=results["uniform"]["per"],
             sample_all=results["sample"]["all"], sample_per=results["sample"]["per"],
             shared_ablated=sh_ppl, n_routed=n_routed, top_k=topk)
    with open(args.out + "_summary.txt", "w") as f:
        f.write(f"model {args.model}\nn_routed {n_routed}\ntop_k {topk}\nn_shared {n_shared}\n"
                f"baseline_ppl {base_ppl}\n"
                f"uniform_all {results['uniform']['all']}\n"
                f"sample_all {results['sample']['all']}\n"
                f"shared_ablated {sh_ppl}\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = np.arange(L)
        fig, ax = plt.subplots(figsize=(max(6, L * 0.4), 4))
        ax.bar(x - 0.2, results["uniform"]["per"] - base_ppl, width=0.4, label="uniform")
        ax.bar(x + 0.2, results["sample"]["per"] - base_ppl, width=0.4, label="sample")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("MoE layer"); ax.set_ylabel("PPL increase (randomize one layer)")
        ax.set_title(f"DeepSeekMoE routing perturbation per layer\n"
                     f"base {base_ppl:.2f} | uniform-all {results['uniform']['all']:.2f} | "
                     f"sample-all {results['sample']['all']:.2f}")
        ax.legend()
        fig.tight_layout(); fig.savefig(args.out + ".png", dpi=150)
        print(f"[dsmoe] saved {args.out}.npz / .png / _summary.txt")
    except Exception as ex:
        print(f"[dsmoe] plot skipped: {ex}")


if __name__ == "__main__":
    main()
