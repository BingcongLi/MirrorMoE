"""
Routing-induced variance on OLMoE-1B-7B (no shared expert) — the no-share
comparison for the DeepSeekMoE variance experiment.

Resample routing M times (stochastic top-k); per MoE block capture the output;
per token compute V_abs / V_rel across draws, averaged over tokens, per layer.

  python scripts/analysis/olmoe_routing_variance.py --model allenai/OLMoE-1B-7B-0924 \
      --num-seqs 8 --seq-len 1024 --draws 64 --micro 2 --out logs-eval/olvar/olmoe
"""
import argparse, os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock

SAMPLE = {"on": False}
CAPTURE = {}

_orig_forward = OlmoeSparseMoeBlock.forward


def _patched_forward(self, hidden_states):
    if not SAMPLE["on"]:
        return _orig_forward(self, hidden_states)
    b, s, h = hidden_states.shape
    hs = hidden_states.view(-1, h)
    router_logits = self.gate(hs)
    E, k, T = self.num_experts, self.top_k, hs.shape[0]
    p = F.softmax(router_logits, dim=1, dtype=torch.float)
    selected = torch.multinomial(p, k, replacement=False)
    weights = p.gather(1, selected)
    if getattr(self, "norm_topk_prob", True):
        weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = weights.to(hs.dtype)
    final = torch.zeros((T, h), dtype=hs.dtype, device=hs.device)
    mask = F.one_hot(selected, num_classes=E).permute(2, 1, 0)
    for e in range(E):
        i_idx, top_x = torch.where(mask[e])
        if top_x.numel() == 0:
            continue
        out = self.experts[e](hs[top_x]) * weights[top_x, i_idx, None]
        final.index_add_(0, top_x, out.to(hs.dtype))
    final3 = final.view(b, s, h)
    CAPTURE[self._idx] = final3.detach()
    return final3, router_logits


OlmoeSparseMoeBlock.forward = _patched_forward


def load_data(tok, n, s):
    try:
        from datasets import load_dataset
        text = "\n\n".join(load_dataset("wikitext", "wikitext-103-raw-v1", split="test")["text"])
    except Exception as ex:
        print(f"[olvar] wikitext failed ({ex}); embedded text")
        text = "The mixture of experts is a machine learning technique. " * 200
    ids = tok(text, return_tensors="pt").input_ids[0]
    need = n * s
    while ids.numel() < need:
        ids = torch.cat([ids, ids])
    return ids[:need].view(n, s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--num-seqs", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--draws", type=int, default=64)
    ap.add_argument("--micro", type=int, default=2)
    ap.add_argument("--out", default="logs-eval/olvar/olmoe")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    blocks = [m for m in model.modules() if isinstance(m, OlmoeSparseMoeBlock)]
    for i, b in enumerate(blocks):
        b._idx = i
    L = len(blocks)
    ids = load_data(tok, args.num_seqs, args.seq_len).to(model.device)
    N, S = ids.shape
    Ttot = N * S
    print(f"[olvar] {args.model}: {L} MoE layers, {Ttot} tokens, {args.draws} draws (no shared)")

    acc_sum, acc_sq = {}, {}
    SAMPLE["on"] = True
    with torch.no_grad():
        for m in range(args.draws):
            for c in range(0, N, args.micro):
                CAPTURE.clear()
                model(ids[c:c + args.micro])
                off = c * S
                for idx, o in CAPTURE.items():
                    o = o.float().reshape(-1, o.shape[-1])
                    if idx not in acc_sum:
                        acc_sum[idx] = torch.zeros(Ttot, o.shape[-1], device=o.device)
                        acc_sq[idx] = torch.zeros(Ttot, device=o.device)
                    acc_sum[idx][off:off + o.shape[0]] += o
                    acc_sq[idx][off:off + o.shape[0]] += (o * o).sum(-1)
            if (m + 1) % 8 == 0:
                print(f"[olvar]   draw {m+1}/{args.draws}")
    SAMPLE["on"] = False

    Mn = args.draws
    v_abs, v_rel = np.full(L, np.nan), np.full(L, np.nan)
    for idx in sorted(acc_sum):
        mean = acc_sum[idx] / Mn
        msq = (mean * mean).sum(-1)
        va = (acc_sq[idx] / Mn - msq).clamp_min(0)
        v_abs[idx] = float(va.mean())
        v_rel[idx] = float((va / (msq + 1e-9)).mean())

    print("[olvar] layer |   V_abs   |   V_rel")
    for idx in range(L):
        print(f"[olvar]  {idx:>3}  {v_abs[idx]:.4e}  {v_rel[idx]:.4e}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out + ".npz", v_abs=v_abs, v_rel=v_rel)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = np.arange(L)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(x, v_rel, "o-"); ax[0].set_title("V_rel (relative routing variance)")
        ax[0].set_xlabel("MoE layer")
        ax[1].plot(x, v_abs, "s-", color="tab:orange"); ax[1].set_title("V_abs (absolute)")
        ax[1].set_xlabel("MoE layer")
        fig.tight_layout(); fig.savefig(args.out + ".png", dpi=150)
        print(f"[olvar] saved {args.out}.npz / .png")
    except Exception as ex:
        print(f"[olvar] plot skipped: {ex}")


if __name__ == "__main__":
    main()
