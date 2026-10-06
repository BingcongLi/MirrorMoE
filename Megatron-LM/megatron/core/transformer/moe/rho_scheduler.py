"""Cosine schedule for the MirrorMoE mixed-softmax routing coefficient ρ.

Usage (automatic — called inside TopKRouter.forward every step):
    The scheduler is transparent to the training loop.  Set --moe-router-rho
    to the *peak* value; ρ will decay from that value to 0 over training.

Standalone usage (e.g. plotting / unit tests):
    from megatron.core.transformer.moe.rho_scheduler import cosine_rho
    rho_t = cosine_rho(step=500, total_steps=2121, rho_max=0.1)
"""

import math


def cosine_rho(step: int, total_steps: int, rho_max: float, rho_min: float = 0.0) -> float:
    """Cosine decay schedule for ρ.

    Starts at *rho_max* at step 0 and smoothly decays to *rho_min* at
    *total_steps*, following a half-cosine curve:

        ρ(t) = rho_min + (rho_max - rho_min) * 0.5 * (1 + cos(π * t/T))

    Args:
        step:        Current training step (0-indexed).
        total_steps: Total number of training steps T.
        rho_max:     Peak value of ρ (at step 0).
        rho_min:     Final value of ρ (at step >= total_steps). Default 0.0.

    Returns:
        float: ρ at the given step, clamped to [rho_min, rho_max].
    """
    if total_steps <= 0:
        return rho_max
    progress = min(step / total_steps, 1.0)
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return rho_min + (rho_max - rho_min) * cosine_factor
