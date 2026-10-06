"""
Uniform-blend routing ablation for mirror-moe.

Ablation baseline:
  p_selection = (1-ρ_t) · softmax(Wh) + ρ_t · (1/N)
  ρ_t = ρ_max · ½ · (1 + cos(π · t / T))   [cosine annealing to 0]

Output weighting and aux-loss use plain softmax p1, matching mirror-moe exactly.
Activated by --moe-router-second-component uniform [--moe-router-cosine-rho].
"""
import math
from functools import partial

import torch


def log_selected_xi_stats(router, xi: torch.Tensor, routing_map: torch.Tensor):
    """
    Ablation logging. Let xi-bar be the router affinity xi renormalised over the
    experts SELECTED by p = (1-rho)*xi + rho*zeta (routing_map). Define the

        effective path = exp( - sum_{e in selected} xi-bar_e * log xi-bar_e )
                       = exp( entropy of xi-bar )

    i.e. the perplexity / effective number of experts of the affinity
    distribution among the selected set (ranges in [1, k]). ~1 means one expert
    carries all the affinity (the other k-1 are effectively pruned); ~k means
    affinity is spread evenly over the selected experts.

    Logged per MoE layer, per iteration, averaged over tokens, via the aux-loss
    tracker (-> track_moe_metrics -> wandb). Companion small-xi stats are also
    logged for the "pruned" analysis.

    Args:
        router: TopKRouter instance (provides .topk, .layer_number, .config).
        xi: router affinity softmax(Wh), shape [num_tokens, num_experts].
        routing_map: bool mask of experts selected by p, same shape.
    """
    from megatron.core.transformer.moe.moe_utils import save_to_aux_losses_tracker

    if router.layer_number is None:
        return
    with torch.no_grad():
        k = router.topk
        xi_sel = (xi.float() * routing_map)                       # [T, E], 0 off-selection
        denom = xi_sel.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        xi_hat = xi_sel / denom                                   # xi-bar, normalised over selected

        # entropy of xi-bar over the selected set (0*log0 -> 0 via the clamp; the
        # zero xi_hat on non-selected experts contributes 0 to the sum).
        ent = -(xi_hat * xi_hat.clamp_min(1e-9).log()).sum(dim=-1)  # [T]
        effective_path = torch.exp(ent)                            # [T], in [1, k]

        # Companion small-xi stats.
        min_xi = xi_hat.masked_fill(~routing_map, float('inf')).min(dim=-1).values
        pruned = ((xi_hat < 0.5 / k) & routing_map).float().sum(dim=-1) / k

        nl = router.config.num_layers
        ln = router.layer_number
        save_to_aux_losses_tracker('effective_path', effective_path.mean(), ln, nl)
        save_to_aux_losses_tracker('sel_xi_min', min_xi.mean(), ln, nl)
        save_to_aux_losses_tracker('sel_xi_frac_pruned', pruned.mean(), ln, nl)


def cosine_rho(rho_max: float, curr_iter: int, total_iters: int) -> float:
    """Cosine-anneal ρ from rho_max at step 0 to 0 at total_iters."""
    if total_iters <= 0:
        return 0.0
    t = min(curr_iter, total_iters)
    return rho_max * 0.5 * (1.0 + math.cos(math.pi * t / total_iters))


def sampled_topk_routing(router, logits: torch.Tensor):
    """
    Stochastic routing: sample k experts WITHOUT replacement from softmax(Wh),
    instead of taking the deterministic top-k. For comparison against top-k.

    - Training: multinomial sampling (k distinct experts per token, drawn
      proportional to the softmax routing probabilities).
    - Eval: deterministic top-k, for reproducible numbers.

    Output weighting and aux-loss use the full softmax probs (same as flame).

    Args:
        router: TopKRouter instance.
        logits: [num_tokens, num_experts].

    Returns:
        probs: softmax probs masked to the selected experts.
        routing_map: bool mask of selected experts.
    """
    from megatron.core.transformer.moe.moe_utils import switch_load_balancing_loss_func

    probs_full = torch.softmax(logits, dim=-1, dtype=torch.float32).to(logits.dtype)

    if router.training:
        # Sample k distinct experts per token ~ routing probability.
        sample_idx = torch.multinomial(
            probs_full.float(), router.topk, replacement=False
        )  # [T, k]
    else:
        # Deterministic top-k at eval for reproducibility.
        _, sample_idx = torch.topk(probs_full, router.topk, dim=-1)

    routing_map = torch.zeros_like(probs_full, dtype=torch.bool).scatter_(1, sample_idx, True)
    probs = probs_full * routing_map

    if router.training:
        tokens_per_expert = routing_map.sum(dim=0).float()
        aux_loss_func = partial(
            switch_load_balancing_loss_func,
            probs=probs_full,
            tokens_per_expert=tokens_per_expert,
            topk=router.topk,
        )
        probs = router.apply_load_balancing_loss(probs, aux_loss_func)

    return probs, routing_map


def random_replace_topk_routing(router, logits: torch.Tensor, n_replace: int):
    """
    Top-k routing where the lowest-scoring n_replace of the selected top-k experts
    are replaced with experts drawn uniformly at random from the NON-selected pool.

    - Training: keep top (k - n_replace) by score; fill the remaining n_replace slots
      with distinct uniformly-random experts sampled from those NOT in the top-k.
    - Eval: deterministic top-k (no replacement), for reproducibility.

    Output weighting and aux-loss use the full softmax probs (same as flame).

    Args:
        router: TopKRouter instance.
        logits: [num_tokens, num_experts].
        n_replace: number of low-rank top-k slots to replace with random experts.

    Returns:
        probs: softmax probs masked to the selected experts.
        routing_map: bool mask of selected experts.
    """
    from megatron.core.transformer.moe.moe_utils import switch_load_balancing_loss_func

    probs_full = torch.softmax(logits, dim=-1, dtype=torch.float32).to(logits.dtype)
    k = router.topk

    # top-k indices, sorted by score descending: col 0 is the strongest.
    _, topk_idx = torch.topk(probs_full, k, dim=-1)  # [T, k]

    if router.training and n_replace > 0:
        keep = k - n_replace
        keep_idx = topk_idx[:, :keep]  # top (k - n_replace) experts we exploit

        # Mask of the original top-k so we sample replacements from outside it.
        topk_mask = torch.zeros_like(probs_full, dtype=torch.bool).scatter_(1, topk_idx, True)
        # Uniform random scores over non-selected experts; top-n_replace of those
        # = n_replace distinct uniform-random experts from the non-selected pool.
        rand = torch.rand_like(probs_full).masked_fill(topk_mask, float('-inf'))
        _, repl_idx = torch.topk(rand, n_replace, dim=-1)  # [T, n_replace]

        sel_idx = torch.cat([keep_idx, repl_idx], dim=1)  # [T, k]
    else:
        sel_idx = topk_idx

    routing_map = torch.zeros_like(probs_full, dtype=torch.bool).scatter_(1, sel_idx, True)
    probs = probs_full * routing_map

    if router.training:
        tokens_per_expert = routing_map.sum(dim=0).float()
        aux_loss_func = partial(
            switch_load_balancing_loss_func,
            probs=probs_full,
            tokens_per_expert=tokens_per_expert,
            topk=router.topk,
        )
        probs = router.apply_load_balancing_loss(probs, aux_loss_func)

    return probs, routing_map


def uniform_blend_routing(router, logits: torch.Tensor):
    """
    Top-k routing with p = (1-ρ)*softmax(Wh) + ρ*(1/N) as selection scores.

    Output weighting and aux-loss use plain softmax p1 (identical to mirror-moe),
    so only the *selection* criterion differs from the antisoftmax baseline.

    Args:
        router: TopKRouter instance — provides .topk, .num_experts, .training,
                .rho, and .apply_load_balancing_loss().
        logits: [num_tokens, num_experts], any float dtype.

    Returns:
        probs: p1 masked to top-k selected experts.
        routing_map: bool mask of selected experts.
    """
    from megatron.core.transformer.moe.moe_utils import switch_load_balancing_loss_func

    p1 = torch.softmax(logits, dim=-1, dtype=torch.float32).to(logits.dtype)
    uniform = torch.full_like(p1, 1.0 / router.num_experts)
    scores = (1.0 - router.rho) * p1 + router.rho * uniform  # selection only

    _, topk_indices = torch.topk(scores, k=router.topk, dim=-1)
    routing_map = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, topk_indices, True)

    probs = p1 * routing_map  # output weighting = plain softmax

    if router.training and getattr(router.config, 'moe_router_log_selected_xi', False):
        log_selected_xi_stats(router, p1, routing_map)

    if router.training:
        tokens_per_expert = routing_map.sum(dim=0).float()
        # Load-balancing probability term: plain affinity p1, or the blended
        # selection distribution (scores=xi) that actually drives dispatch.
        lb_probs = scores if router.config.moe_router_lb_probs == 'selection' else p1
        aux_loss_func = partial(
            switch_load_balancing_loss_func,
            probs=lb_probs,
            tokens_per_expert=tokens_per_expert,
            topk=router.topk,
        )
        probs = router.apply_load_balancing_loss(probs, aux_loss_func)

    return probs, routing_map
