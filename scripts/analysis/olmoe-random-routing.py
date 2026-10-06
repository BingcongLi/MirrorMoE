"""
Routing-perturbation ablation on OLMoE-1B-7B (no shared expert) — the no-share
comparison for the DeepSeekMoE experiment.

At EVAL ONLY, replace each MoE block's deterministic top-k with:
  uniform : k experts UNIFORMLY at random, uniform 1/k combine
  sample  : k experts ~ multinomial(softmax(router_logits)) w/o replacement,
            affinity-normalized combine (stochastic top-k)

Reports, for BOTH modes: baseline PPL, random-ALL PPL, per-layer PPL curve.

Usage:
  python scripts/analysis/olmoe_random_routing.py \
      --model allenai/OLMoE-1B-7B-0924 --num-seqs 32 --seq-len 2048 \
      --repeats 3 --out logs-eval/olmoe

Needs transformers>=4.45 (native OLMoE), datasets, accelerate, torch.
"""
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock

STATE = {"mode": None, "layers": set()}

_orig_forward = OlmoeSparseMoeBlock.forward


def _patched_forward(self, hidden_states):
    idx = getattr(self, "_moe_idx", None)
    if STATE["mode"] is None or idx not in STATE["layers"]:
        return _orig_forward(self, hidden_states)

    b, s, h = hidden_states.shape
    hs = hidden_states.view(-1, h)
    router_logits = self.gate(hs)                       # [tokens, E]
    E, k = self.num_experts, self.top_k
    T = hs.shape[0]

    if STATE["mode"] == "uniform":
        selected = torch.rand(T, E, device=hs.device).topk(k, dim=-1).indices
        weights = torch.full((T, k), 1.0 / k, device=hs.device, dtype=hs.dtype)
    else:  # sample: stochastic top-k from the router distribution
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
        cur = hs[top_x]
        out = self.experts[e](cur) * weights[top_x, i_idx, None]
        final.index_add_(0, top_x, out.to(hs.dtype))
    return final.view(b, s, h), router_logits


OlmoeSparseMoeBlock.forward = _patched_forward


def index_blocks(model):
    blocks = [m for m in model.modules() if isinstance(m, OlmoeSparseMoeBlock)]
    for i, blk in enumerate(blocks):
        blk._moe_idx = i
    return blocks


def load_data(tok, num_seqs, seq_len):
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
        text = "\n\n".join(ds["text"])
    except Exception as ex:
        print(f"[olmoe] wikitext load failed ({ex}); using embedded text")
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
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--num-seqs", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--micro", type=int, default=2)
    ap.add_argument("--out", default="logs-eval/olmoe")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    blocks = index_blocks(model)
    E = model.config.num_experts
    k = model.config.num_experts_per_tok
    L = len(blocks)
    print(f"[olmoe] {args.model}: num_experts={E} top_k={k} moe_layers={L} (no shared expert)")

    ids = load_data(tok, args.num_seqs, args.seq_len)

    def avg_ppl():
        return float(np.mean([ppl(model, ids, args.micro)[0] for _ in range(args.repeats)]))

    STATE["mode"], STATE["layers"] = None, set()
    base_ppl, base_loss = ppl(model, ids, args.micro)
    print(f"[olmoe] baseline PPL = {base_ppl:.4f} (loss {base_loss:.4f})")

    results = {}
    for mode in ("uniform", "sample"):
        STATE["mode"] = mode
        STATE["layers"] = set(range(L))
        all_ppl = avg_ppl()
        print(f"[olmoe] [{mode}] random-ALL PPL = {all_ppl:.4f}  "
              f"increase = {all_ppl - base_ppl:+.4f} ({100*(all_ppl/base_ppl-1):+.1f}%)")
        per = []
        for i in range(L):
            STATE["layers"] = {i}
            p = avg_ppl()
            per.append(p)
            print(f"[olmoe] [{mode}]   layer {i}: PPL {p:.4f}  Δ {p - base_ppl:+.4f}")
        results[mode] = {"all": all_ppl, "per": np.array(per)}
    STATE["mode"], STATE["layers"] = None, set()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out + ".npz", base_ppl=base_ppl,
             uniform_all=results["uniform"]["all"], uniform_per=results["uniform"]["per"],
             sample_all=results["sample"]["all"], sample_per=results["sample"]["per"],
             num_experts=E, top_k=k)
    with open(args.out + "_summary.txt", "w") as f:
        f.write(f"model {args.model}\nnum_experts {E}\ntop_k {k}\nbaseline_ppl {base_ppl}\n"
                f"uniform_all {results['uniform']['all']}\nsample_all {results['sample']['all']}\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = np.arange(L)
        fig, ax = plt.subplots(figsize=(max(6, L * 0.4), 4))
        ax.bar(x - 0.2, results["uniform"]["per"] - base_ppl, width=0.4, label="uniform")
        ax.bar(x + 0.2, results["sample"]["per"] - base_ppl, width=0.4, label="sample (stochastic top-k)")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("MoE layer"); ax.set_ylabel("PPL increase (randomize one layer)")
        ax.set_title(f"OLMoE (no shared) routing perturbation per layer\n"
                     f"base {base_ppl:.2f} | uniform-all {results['uniform']['all']:.2f} | "
                     f"sample-all {results['sample']['all']:.2f}")
        ax.legend()
        fig.tight_layout(); fig.savefig(args.out + ".png", dpi=150)
        print(f"[olmoe] saved {args.out}.npz / .png / _summary.txt")
    except Exception as ex:
        print(f"[olmoe] plot skipped: {ex}")


if __name__ == "__main__":
    main()
