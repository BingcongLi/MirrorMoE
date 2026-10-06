# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from functools import partial
from typing import Callable

import torch

from megatron.core import parallel_state
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import (
    MoEAuxLossAutoScaler,
    mixed_softmax_scores,
    save_to_aux_losses_tracker,
    sequence_load_balancing_loss_func,
    sinkhorn,
    switch_load_balancing_loss_func,
    topk_softmax_with_capacity,
    z_loss_func,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.moe.router_ablation import (
    cosine_rho,
    log_selected_xi_stats,
    random_replace_topk_routing,
    sampled_topk_routing,
    uniform_blend_routing,
)


class Router(ABC, MegatronModule):
    """Base Router class"""

    def __init__(self, config: TransformerConfig) -> None:
        """
        Initialize the Router module.

        Args:
            config (TransformerConfig): Configuration object for the Transformer model.
        """
        super().__init__(config)
        self.config = config
        self.num_experts = self.config.num_moe_experts
        self.moe_aux_loss_func = None
        self.layer_number = None

        # Initialize the gate weights.
        # TODO: Add support for GPU initialization, which requires updating the golden values.
        self.weight = torch.nn.Parameter(
            torch.empty((self.config.num_moe_experts, self.config.hidden_size), dtype=torch.float32)
        )
        if config.perform_initialization:
            config.init_method(self.weight)
        self.weight.data = self.weight.data.to(dtype=config.params_dtype)
        setattr(self.weight, 'sequence_parallel', config.sequence_parallel)

    def gating(self, input: torch.Tensor):
        """Forward pass of the router gate.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Logits tensor.
        """
        if self.weight.device.type == 'cpu':
            # move weights to GPU
            self.weight.data = self.weight.data.to(device=torch.cuda.current_device())
        # Convert to specified datatype for routing computation if enabled
        router_dtype = input.dtype
        if self.config.moe_router_dtype == 'fp32':
            router_dtype = torch.float32
        elif self.config.moe_router_dtype == 'fp64':
            router_dtype = torch.float64
        logits = torch.nn.functional.linear(input.to(router_dtype), self.weight.to(router_dtype))
        return logits

    @abstractmethod
    def routing(self, logits: torch.Tensor):
        """Routing function.

        Args:
            logits (torch.Tensor): Logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing token assignment
            probabilities and mapping.
        """
        raise NotImplementedError("Routing function not implemented.")

    @abstractmethod
    def forward(self, input: torch.Tensor):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
        """
        raise NotImplementedError("Forward function not implemented.")

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the router."""
        self.layer_number = layer_number


class TopKRouter(Router):
    """Route each token to the top-k experts."""

    def __init__(self, config: TransformerConfig) -> None:
        """Initialize the zero token dropping router.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
        """
        super().__init__(config=config)
        self.topk = self.config.moe_router_topk
        self.routing_type = self.config.moe_router_load_balancing_type
        self.score_function = self.config.moe_router_score_function
        self.input_jitter = None

        # rho_max: the configured peak value (fixed); rho: current value
        # (updated each forward pass by the cosine schedule).
        # Note: no gradient hook needed. Adam is scale-invariant under constant
        # gradient scaling (m and v scale together, m/√v is unchanged), so the
        # (1-2ρ) attenuation does not affect the Adam update direction.
        # The residual variance from ρ·(j₁+j₂) is O(ρ²) and Adam adapts naturally.
        self.rho_max = getattr(self.config, 'moe_router_rho', 0.0)
        self.rho = self.rho_max

        # Stores the full softmax distribution over all experts for the most
        # recent forward pass. Set by routing(), read by routing_consistency_loss().
        self.routing_probs: torch.Tensor = None

        self.enable_expert_bias = self.config.moe_router_enable_expert_bias
        if self.enable_expert_bias:
            self.register_buffer(
                'local_tokens_per_expert',
                torch.zeros(self.config.num_moe_experts, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                'expert_bias', torch.zeros(self.config.num_moe_experts, dtype=torch.float32)
            )
        else:
            self.local_tokens_per_expert = None
            self.expert_bias = None

    def sinkhorn_load_balancing(self, logits: torch.Tensor):
        """Apply sinkhorn routing to the logits tensor.

        Args:
            logits (torch.Tensor): The logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing token assignment
            probabilities and mask.
        """

        def _sinkhorn_activation(logits):
            if self.topk == 1:
                logits = torch.sigmoid(logits)
            else:  # k > 1
                logits = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
            return logits

        assert self.config.moe_aux_loss_coeff == 0, "Sinkhorn routing does not support aux loss."
        if self.training:
            with torch.no_grad():
                norm_logits = sinkhorn(
                    logits.to(dtype=torch.float32)
                )  # explicit fp32 conversion for stability
                _, indices = torch.topk(norm_logits, k=self.topk, dim=1)
            logits = _sinkhorn_activation(logits)
        else:
            logits = _sinkhorn_activation(logits)
            _, indices = torch.topk(logits, k=self.topk, dim=1)
        map = torch.zeros_like(logits).int().scatter(1, indices, 1).bool()
        scores = logits * map
        return scores, map

    def aux_loss_load_balancing(self, logits: torch.Tensor):
        """Apply loss-based load balancing to the logits tensor.

        Args:
            logits (torch.Tensor): the logits tensor after gating, shape: [num_tokens, num_experts].

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment.
            routing_map (torch.Tensor): The mask of token to experts assignment.
        """
        probs, routing_map, tokens_per_expert = topk_softmax_with_capacity(
            logits,
            self.topk,
            capacity_factor=self.config.moe_expert_capacity_factor,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            drop_policy=self.config.moe_token_drop_policy,
            use_pre_softmax=self.config.moe_router_pre_softmax,
            num_groups=self.config.moe_router_num_groups,
            group_topk=self.config.moe_router_group_topk,
            scaling_factor=self.config.moe_router_topk_scaling_factor,
            deterministic_mode=self.config.deterministic_mode,
            score_function=self.score_function,
            expert_bias=self.expert_bias,
        )

        if self.training:
            # Apply load balancing loss
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
            aux_loss_func = partial(
                switch_load_balancing_loss_func,
                probs=scores,
                tokens_per_expert=tokens_per_expert,
                topk=self.topk,
            )
            probs = self.apply_load_balancing_loss(
                activation=probs, load_balancing_loss_func=aux_loss_func
            )
        return probs, routing_map

    def seq_aux_loss_load_balancing(self, logits: torch.Tensor, bsz: int, seq_length: int):
        """Apply loss-based load balancing to the logits tensor."""

        probs, routing_map, tokens_per_expert = topk_softmax_with_capacity(
            logits,
            self.topk,
            capacity_factor=self.config.moe_expert_capacity_factor,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            drop_policy=self.config.moe_token_drop_policy,
            use_pre_softmax=self.config.moe_router_pre_softmax,
            num_groups=self.config.moe_router_num_groups,
            group_topk=self.config.moe_router_group_topk,
            scaling_factor=self.config.moe_router_topk_scaling_factor,
            deterministic_mode=self.config.deterministic_mode,
            score_function=self.score_function,
            expert_bias=self.expert_bias,
        )

        if self.training:
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
            aux_loss_func = partial(
                sequence_load_balancing_loss_func,
                probs=scores,
                routing_map=routing_map,
                batch_size=bsz,
                seq_length=seq_length,
                topk=self.topk,
            )
            probs = self.apply_load_balancing_loss(
                activation=probs, load_balancing_loss_func=aux_loss_func
            )

        return probs, routing_map

    def apply_load_balancing_loss(
        self, activation: torch.Tensor, load_balancing_loss_func: Callable
    ):
        """Calculate auxiliary loss, attach gradient function to activation and add to logging."""
        moe_aux_loss_coeff = self.config.moe_aux_loss_coeff
        if moe_aux_loss_coeff == 0:
            return activation
        sequence_partition_group = None
        if self.config.moe_token_dispatcher_type == "alltoall_seq":
            sequence_partition_group = parallel_state.get_context_parallel_group()
            moe_aux_loss_coeff /= parallel_state.get_tensor_model_parallel_world_size()
        elif parallel_state.get_tensor_and_context_parallel_world_size() > 1:
            sequence_partition_group = parallel_state.get_tensor_and_context_parallel_group()

        aux_loss = load_balancing_loss_func(
            moe_aux_loss_coeff=moe_aux_loss_coeff, sequence_partition_group=sequence_partition_group
        )
        save_to_aux_losses_tracker(
            "load_balancing_loss",
            aux_loss / moe_aux_loss_coeff,
            self.layer_number,
            self.config.num_layers,
            reduce_group=sequence_partition_group,
        )
        activation = MoEAuxLossAutoScaler.apply(activation, aux_loss)
        return activation

    def apply_z_loss(self, logits):
        """Encourages the router's logits to remain small to enhance stability.
        Please refer to the ST-MoE paper (https://arxiv.org/pdf/2202.08906.pdf) for details.

        Args:
            logits (torch.Tensor): The logits of the router.

        Returns:
            torch.Tensor: The logits after applying the z-loss.
        """
        if self.config.moe_z_loss_coeff is not None and self.training:
            moe_z_loss_coeff = (
                self.config.moe_z_loss_coeff
                / parallel_state.get_tensor_and_context_parallel_world_size()
            )
            z_loss = z_loss_func(logits, moe_z_loss_coeff)
            logits = MoEAuxLossAutoScaler.apply(logits, z_loss)
            save_to_aux_losses_tracker(
                "z_loss", z_loss / moe_z_loss_coeff, self.layer_number, self.config.num_layers
            )
        return logits

    def apply_input_jitter(self, input: torch.Tensor):
        """Add noise to the input tensor.
        Refer to https://arxiv.org/abs/2101.03961.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Jittered input.
        """
        if self.config.moe_input_jitter_eps is not None:
            eps = self.config.moe_input_jitter_eps
            if self.input_jitter is None:
                self.input_jitter = torch.distributions.uniform.Uniform(
                    torch.tensor(1.0 - eps, device=input.device),
                    torch.tensor(1.0 + eps, device=input.device),
                ).rsample
            return input * self.input_jitter(input.shape)
        else:
            return input

    def mixed_softmax_routing(self, logits: torch.Tensor):
        """Top-k routing using mixed-softmax scores (calls mixed_softmax_scores).

        Mixed scores are used only for top-k *selection* (diversity / anti-affinity).
        Output weighting uses plain softmax p1 so that the contribution of each
        selected expert reflects true affinity, not the anti-affinity correction.
        Aux load-balancing loss also uses p1 (unchanged).

        Args:
            logits: [num_tokens, num_experts], fp32.

        Returns:
            probs: plain softmax scores masked to top-k, shape [num_tokens, num_experts].
            routing_map: bool mask, shape [num_tokens, num_experts].
        """
        # Compute p1 once and reuse for both mixed-score selection and output weighting.
        p1 = torch.softmax(logits, dim=-1, dtype=torch.float32).to(logits.dtype)
        p2 = torch.softmax(-logits, dim=-1, dtype=torch.float32).to(logits.dtype)
        scores = (1.0 - self.rho) * p1 + self.rho * p2  # [T, E] — selection only

        # DeepSeek loss-free load balancing: add the per-expert bias to the
        # SELECTION score only (never to p1, the output weight). expert_bias is
        # updated each step by the sign rule in finalize_model_grads.
        sel_scores = scores if self.expert_bias is None else scores + self.expert_bias
        _, topk_indices = torch.topk(sel_scores, k=self.topk, dim=-1)
        routing_map = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, topk_indices, True)

        # p1 reused for output weighting — no extra softmax call.
        probs = p1 * routing_map

        # Ablation logging: distribution of affinity xi (=p1) over selected experts.
        if self.training and getattr(self.config, 'moe_router_log_selected_xi', False):
            log_selected_xi_stats(self, p1, routing_map)

        if self.training:
            tokens_per_expert = routing_map.sum(dim=0).float()
            # Load-balancing probability term: plain affinity p1, or the blended
            # selection distribution (scores=xi) that actually drives dispatch.
            lb_probs = scores if self.config.moe_router_lb_probs == 'selection' else p1
            aux_loss_func = partial(
                switch_load_balancing_loss_func,
                probs=lb_probs,
                tokens_per_expert=tokens_per_expert,
                topk=self.topk,
            )
            probs = self.apply_load_balancing_loss(probs, aux_loss_func)

        return probs, routing_map

    def routing(self, logits: torch.Tensor):
        """Top-k routing function

        Args:
            logits (torch.Tensor): Logits tensor after gating.

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment.
            routing_map (torch.Tensor): The mapping of token to experts assignment,
                with shape [num_tokens, num_experts].
        """
        seq_length, bsz = logits.shape[:2]
        logits = logits.view(-1, self.config.num_moe_experts)

        # Apply Z-Loss
        logits = self.apply_z_loss(logits)

        if self.config.moe_token_dispatcher_type == "alltoall_seq":
            # Gather the logits from the TP region
            logits = gather_from_sequence_parallel_region(logits)

        if getattr(self.config, 'moe_router_sampled_topk', False):
            scores, routing_map = sampled_topk_routing(self, logits)
            if self.enable_expert_bias and torch.is_grad_enabled():
                with torch.no_grad():
                    self.local_tokens_per_expert += routing_map.sum(dim=0)
            return scores, routing_map

        if getattr(self.config, 'moe_router_random_replace', 0) > 0:
            scores, routing_map = random_replace_topk_routing(
                self, logits, self.config.moe_router_random_replace
            )
            if self.enable_expert_bias and torch.is_grad_enabled():
                with torch.no_grad():
                    self.local_tokens_per_expert += routing_map.sum(dim=0)
            return scores, routing_map

        # Force standard (flame) top-k routing on selected MoE layers, even when
        # rho > 0: the first N layers (moe_router_standard_first_n_layers) and/or
        # an explicit interleaved set (moe_router_standard_layers). Remaining
        # layers use mirror-moe mixed routing. layer_number is global 1-based.
        if not hasattr(self, '_standard_layer_set'):
            _s = getattr(self.config, 'moe_router_standard_layers', '') or ''
            self._standard_layer_set = {int(x) for x in _s.replace(' ', '').split(',') if x}
        _std_first_n = getattr(self.config, 'moe_router_standard_first_n_layers', 0)
        _force_standard = self.layer_number is not None and (
            self.layer_number <= _std_first_n or self.layer_number in self._standard_layer_set
        )

        if self.rho > 0 and not _force_standard:
            _second = getattr(self.config, 'moe_router_second_component', 'antisoftmax')
            if _second == 'uniform':
                scores, routing_map = uniform_blend_routing(self, logits)
            else:
                scores, routing_map = self.mixed_softmax_routing(logits)
            if self.enable_expert_bias and torch.is_grad_enabled():
                with torch.no_grad():
                    self.local_tokens_per_expert += routing_map.sum(dim=0)
            return scores, routing_map

        if self.routing_type == "sinkhorn":
            scores, routing_map = self.sinkhorn_load_balancing(logits)
        elif self.routing_type == "aux_loss":
            scores, routing_map = self.aux_loss_load_balancing(logits)
        elif self.routing_type == "seq_aux_loss":
            scores, routing_map = self.seq_aux_loss_load_balancing(logits, bsz, seq_length)
        elif self.routing_type == "none":
            # A naive top-k routing without load balancing
            scores, routing_map, _ = topk_softmax_with_capacity(
                logits,
                self.topk,
                capacity_factor=self.config.moe_expert_capacity_factor,
                pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
                drop_policy=self.config.moe_token_drop_policy,
                use_pre_softmax=self.config.moe_router_pre_softmax,
                num_groups=self.config.moe_router_num_groups,
                group_topk=self.config.moe_router_group_topk,
                scaling_factor=self.config.moe_router_topk_scaling_factor,
                deterministic_mode=self.config.deterministic_mode,
                score_function=self.score_function,
                expert_bias=self.expert_bias,
            )
        else:
            raise ValueError(f"Unsupported MoE routing type: {self.routing_type}")
        # Prevent extra local tokens accumulation on evaluation or activation recomputation
        if self.enable_expert_bias and torch.is_grad_enabled():
            with torch.no_grad():
                self.local_tokens_per_expert += routing_map.sum(dim=0)

        # Ablation logging for the standard (flame) top-k path: distribution of
        # affinity xi = softmax(logits) over the selected experts. (The rho>0 /
        # sampled / random-replace paths return earlier and log themselves.)
        if self.training and getattr(self.config, 'moe_router_log_selected_xi', False):
            xi = torch.softmax(logits, dim=-1, dtype=torch.float32).to(logits.dtype)
            log_selected_xi_stats(self, xi, routing_map)

        return scores, routing_map

    def forward(self, input: torch.Tensor):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
        """

        # Update ρ: cosine schedule if enabled, else constant.
        if getattr(self.config, 'moe_router_cosine_rho', False):
            try:
                from megatron.training.global_vars import get_args
                _args = get_args()
                self.rho = cosine_rho(self.rho_max, _args.curr_iteration, _args.train_iters)
            except Exception:
                self.rho = self.rho_max
        else:
            self.rho = self.rho_max

        # Apply input jitter
        input = self.apply_input_jitter(input)
        logits = self.gating(input)

        scores, routing_map = self.routing(logits)

        return scores, routing_map
