"""
Routing-induced variance on DeepSeekMoE-16B (shared-expert model).

Resample routing M times (stochastic top-k ~ softmax(router logits)); for each
MoE block capture the output out(h)=shared+routed; per token compute variance
across the M draws:
  V_abs = mean_m||out_m - mean||^2 ; V_rel = V_abs/||mean||^2 ; shared/out ratio.
Per-layer, averaged over tokens.

  python scripts/analysis/deepseek_routing_variance.py --model deepseek-ai/deepseek-moe-16b-base \
      --num-seqs 8 --seq-len 1024 --draws 64 --micro 2 --out logs-eval/dsvar/dsmoe
"""
import argparse, os
import numpy as np
import torch
import torch.nn.functional as F

import transformers.utils.import_utils as _iu
if not hasattr(_iu, "is_torch_fx_available"):
    try:
        from transformers.utils import is_torch_fx_available as _f
        _iu.is_torch_fx_available = _f
    except Exception:
        _iu.is_torch_fx_available = lambda: False
from transformers import AutoModelForCausalLM, AutoTokenizer

SAMPLE = {"on": False}
CAPTURE, SHARED = {}, {}


def find_blocks(model):
    return [m for m in model.modules() if hasattr(m, "gate") and hasattr(m, "experts")]


def install_hooks(blocks):
    def make_gate(module):
        def h(mod, inp, out):
            if not SAMPLE["on"]:
                return None
            topk_idx, topk_weight = out[0], out[1]
            rest = tuple(out[2:]) if len(out) > 2 else ()
            hidden = inp[0].reshape(-1, inp[0].shape[-1])
            logits = F.linear(hidden.to(mod.weight.dtype), mod.weight)
            p = torch.softmax(logits.float(), dim=-1)
            k = topk_idx.shape[1]
            new_idx = torch.multinomial(p, k, replacement=False)
            w = p.gather(1, new_idx)
            w = (w / (w.sum(-1, keepdim=True) + 1e-9)).to(topk_weight.dtype)
            return (new_idx.to(topk_idx.dtype), w) + rest
        return h

    def make_out(idx):
        def h(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            CAPTURE[idx] = o.detach()
        return h

    def make_sh(idx):
        def h(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            SHARED[idx] = o.detach().float().reshape(-1, o.shape[-1]).norm(dim=-1)
        return h

    for i, b in enumerate(blocks):
        b.gate.register_forward_hook(make_gate(b.gate))
        b.register_forward_hook(make_out(i))
        if getattr(b, "shared_experts", None) is not None:
            b.shared_experts.register_forward_hook(make_sh(i))


def load_data(tok, n, s):
    try:
        from datasets import load_dataset
        text = "\n\n".join(load_dataset("wikitext", "wikitext-103-raw-v1", split="test")["text"])
    except Exception as ex:
        print(f"[dsvar] wikitext failed ({ex}); embedded text")
        text = "The mixture of experts is a machine learning technique. " * 200
    ids = tok(text, return_tensors="pt").input_ids[0]
    need = n * s
    while ids.numel() < need:
        ids = torch.cat([ids, ids])
    return ids[:need].view(n, s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/deepseek-moe-16b-base")
    ap.add_argument("--num-seqs", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--draws", type=int, default=64)
    ap.add_argument("--micro", type=int, default=2)
    ap.add_argument("--out", default="logs-eval/dsvar/dsmoe")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    blocks = find_blocks(model)
    install_hooks(blocks)
    L = len(blocks)
    ids = load_data(tok, args.num_seqs, args.seq_len).to(model.device)
    N, S = ids.shape
    Ttot = N * S
    print(f"[dsvar] {args.model}: {L} MoE layers, {Ttot} tokens, {args.draws} draws")

    acc_sum, acc_sq, sh_full = {}, {}, {}
    SAMPLE["on"] = True
    with torch.no_grad():
        for m in range(args.draws):
            for c in range(0, N, args.micro):
                CAPTURE.clear(); SHARED.clear()
                model(ids[c:c + args.micro])
                off = c * S
                for idx, o in CAPTURE.items():
                    o = o.float().reshape(-1, o.shape[-1])
                    if idx not in acc_sum:
                        acc_sum[idx] = torch.zeros(Ttot, o.shape[-1], device=o.device)
                        acc_sq[idx] = torch.zeros(Ttot, device=o.device)
                    acc_sum[idx][off:off + o.shape[0]] += o
                    acc_sq[idx][off:off + o.shape[0]] += (o * o).sum(-1)
                # shared output is deterministic -> fill the full per-token buffer once
                for idx, sn in SHARED.items():
                    if idx not in sh_full:
                        sh_full[idx] = torch.zeros(Ttot, device=sn.device)
                    sh_full[idx][off:off + sn.shape[0]] = sn
            if (m + 1) % 8 == 0:
                print(f"[dsvar]   draw {m+1}/{args.draws}")
    SAMPLE["on"] = False

    Mn = args.draws
    v_abs, v_rel, sh = (np.full(L, np.nan) for _ in range(3))
    for idx in sorted(acc_sum):
        mean = acc_sum[idx] / Mn
        msq = (mean * mean).sum(-1)
        va = (acc_sq[idx] / Mn - msq).clamp_min(0)
        v_abs[idx] = float(va.mean())
        v_rel[idx] = float((va / (msq + 1e-9)).mean())
        if idx in sh_full:
            sh[idx] = float((sh_full[idx] / (mean.norm(dim=-1) + 1e-9)).mean())

    print("[dsvar] layer |   V_abs   |   V_rel   | shared/out")
    for idx in range(L):
        print(f"[dsvar]  {idx:>3}  {v_abs[idx]:.4e}  {v_rel[idx]:.4e}  {sh[idx]:.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out + ".npz", v_abs=v_abs, v_rel=v_rel, shared_ratio=sh)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = np.arange(L)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(x, v_rel, "o-"); ax[0].set_title("V_rel (relative routing variance)")
        ax[0].set_xlabel("MoE layer")
        if not np.all(np.isnan(sh)):
            a2 = ax[0].twinx(); a2.plot(x, sh, "^--", color="gray", alpha=0.6); a2.set_ylabel("shared/out")
        ax[1].plot(x, v_abs, "s-", color="tab:orange"); ax[1].set_title("V_abs (absolute)")
        ax[1].set_xlabel("MoE layer")
        fig.tight_layout(); fig.savefig(args.out + ".png", dpi=150)
        print(f"[dsvar] saved {args.out}.npz / .png")
    except Exception as ex:
        print(f"[dsvar] plot skipped: {ex}")


if __name__ == "__main__":
    main()
