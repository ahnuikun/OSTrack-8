import math

import torch
from torch import nn
import torch.nn.functional as F


class TargetScaleEstimator(nn.Module):
    """Estimate target scale state from template tokens.

    The estimator pools center/mid/outer template regions and predicts a soft
    state distribution over small, medium, large, and uncertain. It does not use
    ground-truth boxes at inference time.
    """

    def __init__(self, dim, hidden_dim, sigma=0.35):
        super().__init__()
        self.sigma = max(float(sigma), 1e-3)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim * 5),
            nn.Linear(dim * 5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        with torch.no_grad():
            # Start conservative: medium/uncertain dominate until data says otherwise.
            self.mlp[-1].bias[1].fill_(0.35)
            self.mlp[-1].bias[3].fill_(0.25)

    def forward(self, template_tokens):
        if template_tokens is None or template_tokens.numel() == 0:
            return None

        bsz, token_count, channels = template_tokens.shape
        side = int(math.sqrt(token_count))
        if side * side != token_count:
            pooled = template_tokens.mean(dim=1)
            zeros = pooled.new_zeros(pooled.shape)
            descriptor = torch.cat([pooled, pooled, pooled, zeros, zeros], dim=1)
            return torch.softmax(self.mlp(descriptor), dim=1)

        dtype = template_tokens.dtype
        device = template_tokens.device
        coords = torch.linspace(-1.0, 1.0, side, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        dist = torch.sqrt(xx.square() + yy.square()).reshape(1, token_count, 1)

        center_mask = torch.exp(-dist.square() / (2.0 * self.sigma * self.sigma))
        mid_mask = torch.exp(-(dist - 0.55).square() / (2.0 * 0.22 * 0.22))
        outer_mask = torch.exp(-(dist - 0.95).square() / (2.0 * 0.25 * 0.25))

        center = self._weighted_pool(template_tokens, center_mask)
        mid = self._weighted_pool(template_tokens, mid_mask)
        outer = self._weighted_pool(template_tokens, outer_mask)
        descriptor = torch.cat([center, mid, outer, center - mid, mid - outer], dim=1)
        return torch.softmax(self.mlp(descriptor), dim=1)

    @staticmethod
    def _weighted_pool(tokens, weights):
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return (tokens * weights).sum(dim=1)


class MultiScaleTokenFusion(nn.Module):
    """Multi-scale search-token fusion that preserves OSTrack token layout.

    The module receives reshaped search tokens as B x C x H x W and returns the
    same shape. It is inserted before the center head, so the ViT input size and
    token count remain unchanged.
    """

    def __init__(
        self,
        dim,
        local_kernel=3,
        mid_kernel=5,
        dilated_kernel=3,
        dilation=2,
        reduction=4,
        adaptive_scale=True,
        gate_reduction=16,
        gate_temperature=1.0,
        local_bias=0.40,
        mid_bias=0.20,
        context_bias=-0.20,
        context_gate=True,
        context_gate_bias=-1.50,
        residual_confidence_gate=True,
        residual_floor=0.65,
        residual_init=0.10,
        dropout=0.0,
        target_guided=False,
        spatial_adaptive=False,
        spatial_blend=-1.0,
        target_sigma=0.35,
        target_temperature=4.0,
        target_residual_floor=0.30,
        target_response_clamp=0.0,
        target_confidence_gate=False,
        target_confidence_mode="softmax_peak",
        target_confidence_temperature=1.0,
        target_confidence_floor=0.80,
        target_confidence_power=1.0,
        context_confidence_gate=False,
        context_confidence_floor=0.70,
        scale_estimation=False,
        scale_prior_strength=0.25,
        scale_prior_confidence_gate=False,
        scale_uncertain_residual_floor=0.70,
        branch_weight_floor=0.0,
    ):
        super().__init__()
        hidden_dim = max(dim // int(reduction), 64)
        self.adaptive_scale = bool(adaptive_scale)
        self.gate_temperature = max(float(gate_temperature), 1e-3)
        self.use_context_gate = bool(context_gate)
        self.residual_confidence_gate = bool(residual_confidence_gate)
        self.residual_floor = min(max(float(residual_floor), 0.0), 1.0)
        self.target_guided = bool(target_guided)
        self.spatial_adaptive = bool(spatial_adaptive)
        self.spatial_blend = float(spatial_blend)
        self.target_sigma = max(float(target_sigma), 1e-3)
        self.target_temperature = float(target_temperature)
        self.target_residual_floor = min(max(float(target_residual_floor), 0.0), 1.0)
        self.target_response_clamp = max(float(target_response_clamp), 0.0)
        self.target_confidence_gate = bool(target_confidence_gate)
        self.target_confidence_mode = str(target_confidence_mode).lower()
        self.target_confidence_temperature = max(float(target_confidence_temperature), 1e-3)
        self.target_confidence_floor = min(max(float(target_confidence_floor), 0.0), 1.0)
        self.target_confidence_power = max(float(target_confidence_power), 1e-3)
        self.context_confidence_gate = bool(context_confidence_gate)
        self.context_confidence_floor = min(max(float(context_confidence_floor), 0.0), 1.0)
        self.scale_estimation = bool(scale_estimation)
        self.scale_prior_strength = min(max(float(scale_prior_strength), 0.0), 1.0)
        self.scale_prior_confidence_gate = bool(scale_prior_confidence_gate)
        self.scale_uncertain_residual_floor = min(max(float(scale_uncertain_residual_floor), 0.0), 1.0)
        self.branch_weight_floor = min(max(float(branch_weight_floor), 0.0), 0.32)
        self.reduce = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=True)
        self.local_branch = self._depthwise_branch(hidden_dim, int(local_kernel), dilation=1)
        self.mid_branch = self._depthwise_branch(hidden_dim, int(mid_kernel), dilation=1)
        self.context_branch = self._depthwise_branch(hidden_dim, int(dilated_kernel), dilation=int(dilation))
        if self.target_guided:
            self.target_proj = nn.Linear(dim, hidden_dim)
        else:
            self.target_proj = None
        if self.scale_estimation:
            self.scale_estimator = TargetScaleEstimator(dim, hidden_dim, sigma=self.target_sigma)
            self.register_buffer(
                "scale_state_to_branch",
                torch.tensor(
                    [
                        [0.70, 0.25, 0.05],  # small -> local
                        [0.25, 0.60, 0.15],  # medium -> mid
                        [0.12, 0.33, 0.55],  # large -> context
                        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],  # uncertain -> neutral
                    ],
                    dtype=torch.float32,
                ),
            )
        else:
            self.scale_estimator = None
        if self.adaptive_scale:
            gate_hidden_dim = max(hidden_dim // int(gate_reduction), 16)
            self.scale_selector = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden_dim, gate_hidden_dim, kernel_size=1, bias=True),
                nn.GELU(),
                nn.Conv2d(gate_hidden_dim, 3, kernel_size=1, bias=True),
            )
            nn.init.zeros_(self.scale_selector[-1].weight)
            nn.init.zeros_(self.scale_selector[-1].bias)
            with torch.no_grad():
                self.scale_selector[-1].bias[0].fill_(float(local_bias))
                self.scale_selector[-1].bias[1].fill_(float(mid_bias))
                self.scale_selector[-1].bias[2].fill_(float(context_bias))
        else:
            self.scale_selector = None
        if self.use_context_gate:
            gate_hidden_dim = max(hidden_dim // int(gate_reduction), 16)
            self.context_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden_dim, gate_hidden_dim, kernel_size=1, bias=True),
                nn.GELU(),
                nn.Conv2d(gate_hidden_dim, 1, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )
            nn.init.zeros_(self.context_gate[-2].weight)
            nn.init.constant_(self.context_gate[-2].bias, float(context_gate_bias))
        else:
            self.context_gate = None
        if self.spatial_adaptive:
            gate_hidden_dim = max(hidden_dim // int(gate_reduction), 16)
            self.spatial_selector = nn.Sequential(
                nn.Conv2d(hidden_dim + 1, gate_hidden_dim, kernel_size=1, bias=True),
                nn.GELU(),
                nn.Conv2d(gate_hidden_dim, 3, kernel_size=1, bias=True),
            )
            nn.init.zeros_(self.spatial_selector[-1].weight)
            nn.init.zeros_(self.spatial_selector[-1].bias)
            with torch.no_grad():
                self.spatial_selector[-1].bias[0].fill_(float(local_bias))
                self.spatial_selector[-1].bias[1].fill_(float(mid_bias))
                self.spatial_selector[-1].bias[2].fill_(float(context_bias))
        else:
            self.spatial_selector = None
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_dim * 4, dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Dropout2d(float(dropout)) if float(dropout) > 0 else nn.Identity(),
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_init)))

        nn.init.zeros_(self.fuse[0].weight)
        nn.init.zeros_(self.fuse[0].bias)

    @staticmethod
    def _depthwise_branch(dim, kernel_size, dilation=1):
        padding = (kernel_size // 2) * dilation
        return nn.Sequential(
            nn.Conv2d(
                dim,
                dim,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=dim,
                bias=False,
            ),
            nn.GELU(),
        )

    def forward(self, search_feat, template_tokens=None, return_aux=False):
        reduced = self.reduce(search_feat)
        gate_feat, target_response, scale_probs, scale_prior = self._target_guided_gate_feature(
            reduced, template_tokens
        )
        target_confidence = None
        shaped_target_confidence = None
        if target_response is not None and (self.target_confidence_gate or self.context_confidence_gate):
            target_confidence = self._target_confidence(target_response)
            shaped_target_confidence = target_confidence.pow(self.target_confidence_power)
        local_feat = self.local_branch(reduced)
        mid_feat = self.mid_branch(reduced)
        context_feat = self.context_branch(reduced)
        branches = [reduced, local_feat, mid_feat, context_feat]
        weights = None
        if self.scale_selector is not None or self.spatial_selector is not None:
            weights = self._branch_weights(gate_feat, target_response, scale_prior, scale_probs)
            context_weight = weights[:, 2:3]
            if self.context_gate is not None:
                context_weight = context_weight * self.context_gate(gate_feat)
            if self.context_confidence_gate and shaped_target_confidence is not None:
                context_confidence = self.context_confidence_floor + (
                    1.0 - self.context_confidence_floor
                ) * shaped_target_confidence
                context_weight = context_weight * context_confidence
            branches = [
                reduced,
                local_feat * weights[:, 0:1],
                mid_feat * weights[:, 1:2],
                context_feat * context_weight,
            ]
        elif self.context_gate is not None:
            branches[-1] = context_feat * self.context_gate(gate_feat)
        fused = self.fuse(torch.cat(branches, dim=1))
        residual_strength = self.residual_scale
        if self.residual_confidence_gate and weights is not None:
            confidence = weights.max(dim=1, keepdim=True).values
            residual_strength = residual_strength * (
                self.residual_floor + (1.0 - self.residual_floor) * confidence
            )
        if self.target_guided and target_response is not None:
            target_prob = torch.sigmoid(self.target_temperature * target_response)
            target_residual = self.target_residual_floor + (1.0 - self.target_residual_floor) * target_prob
            residual_strength = residual_strength * target_residual
            if self.target_confidence_gate:
                if shaped_target_confidence is None:
                    target_confidence = self._target_confidence(target_response)
                    shaped_target_confidence = target_confidence.pow(self.target_confidence_power)
                confidence_residual = self.target_confidence_floor + (
                    1.0 - self.target_confidence_floor
                ) * shaped_target_confidence
                residual_strength = residual_strength * confidence_residual
        if scale_probs is not None:
            scale_confidence = (1.0 - scale_probs[:, 3:4]).view(scale_probs.shape[0], 1, 1, 1)
            scale_residual = self.scale_uncertain_residual_floor + (
                1.0 - self.scale_uncertain_residual_floor
            ) * scale_confidence
            residual_strength = residual_strength * scale_residual

        out = search_feat + residual_strength * fused
        if not return_aux:
            return out

        aux = {}
        if weights is not None:
            aux["mstf_branch_weights"] = self._mean_branch_weights(weights)
        if scale_probs is not None:
            aux["mstf_scale_probs"] = scale_probs
        if target_response is not None:
            if target_confidence is None:
                target_confidence = self._target_confidence(target_response)
            aux["mstf_target_confidence"] = target_confidence.flatten(1)
        aux["mstf_residual_strength"] = self._batch_mean_scalar(residual_strength, search_feat.shape[0])
        return out, aux

    @staticmethod
    def _batch_mean_scalar(value, batch_size):
        if value.dim() == 0:
            return value.detach().view(1, 1).expand(batch_size, 1)
        return value.detach().flatten(1).mean(dim=1, keepdim=True)

    def _branch_weights(self, gate_feat, target_response=None, scale_prior=None, scale_probs=None):
        global_weights = None
        if self.scale_selector is not None:
            global_weights = torch.softmax(self.scale_selector(gate_feat) / self.gate_temperature, dim=1)

        spatial_weights = None
        if self.spatial_selector is not None:
            if target_response is None:
                target_response = gate_feat.new_zeros(gate_feat.shape[0], 1, gate_feat.shape[2], gate_feat.shape[3])
            spatial_input = torch.cat([gate_feat, target_response], dim=1)
            spatial_weights = torch.softmax(self.spatial_selector(spatial_input) / self.gate_temperature, dim=1)

        if global_weights is None:
            return self._apply_scale_prior(spatial_weights, scale_prior, scale_probs)
        if spatial_weights is None:
            return self._apply_scale_prior(global_weights, scale_prior, scale_probs)

        if self.spatial_blend >= 0.0:
            blend = min(max(self.spatial_blend, 0.0), 1.0)
            weights = (1.0 - blend) * global_weights + blend * spatial_weights
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
            return self._apply_scale_prior(weights, scale_prior, scale_probs)

        weights = spatial_weights * global_weights
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return self._apply_scale_prior(weights, scale_prior, scale_probs)

    def _target_guided_gate_feature(self, reduced, template_tokens=None):
        scale_probs, scale_prior = self._scale_prior(template_tokens, reduced)
        if self.target_proj is None or template_tokens is None or template_tokens.numel() == 0:
            return reduced, None, scale_probs, scale_prior

        target_proto = self._target_prototype(template_tokens)
        target = self.target_proj(target_proto).view(reduced.shape[0], -1, 1, 1)
        gate_feat = reduced + target
        target_response = self._target_response(reduced, target)
        return gate_feat, target_response, scale_probs, scale_prior

    def _scale_prior(self, template_tokens, reduced):
        if self.scale_estimator is None or template_tokens is None or template_tokens.numel() == 0:
            return None, None
        scale_probs = self.scale_estimator(template_tokens)
        state_to_branch = self.scale_state_to_branch.to(device=reduced.device, dtype=reduced.dtype)
        scale_prior = torch.matmul(scale_probs.to(dtype=reduced.dtype), state_to_branch)
        return scale_probs, scale_prior

    def _apply_scale_prior(self, weights, scale_prior, scale_probs=None):
        if weights is None:
            return None
        if scale_prior is not None and self.scale_prior_strength > 0.0:
            if weights.dim() == 4:
                prior = scale_prior.view(scale_prior.shape[0], scale_prior.shape[1], 1, 1)
                strength = weights.new_full((scale_prior.shape[0], 1, 1, 1), self.scale_prior_strength)
                if self.scale_prior_confidence_gate and scale_probs is not None:
                    confidence = (1.0 - scale_probs[:, 3:4]).to(dtype=weights.dtype).view(scale_prior.shape[0], 1, 1, 1)
                    strength = strength * confidence
            else:
                prior = scale_prior
                strength = weights.new_full((scale_prior.shape[0], 1), self.scale_prior_strength)
                if self.scale_prior_confidence_gate and scale_probs is not None:
                    confidence = (1.0 - scale_probs[:, 3:4]).to(dtype=weights.dtype)
                    strength = strength * confidence
            weights = (1.0 - strength) * weights + strength * prior
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return self._apply_branch_floor(weights)

    def _apply_branch_floor(self, weights):
        if self.branch_weight_floor <= 0.0:
            return weights
        floor = min(self.branch_weight_floor, 0.32)
        weights = (1.0 - 3.0 * floor) * weights + floor
        return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

    @staticmethod
    def _mean_branch_weights(weights):
        if weights.dim() == 4:
            return weights.mean(dim=(2, 3))
        return weights

    def _target_response(self, reduced, target):
        response = (F.normalize(reduced, dim=1) * F.normalize(target, dim=1)).sum(dim=1, keepdim=True)
        mean = response.mean(dim=(2, 3), keepdim=True)
        std = response.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        response = (response - mean) / std
        if self.target_response_clamp > 0.0:
            response = response.clamp(-self.target_response_clamp, self.target_response_clamp)
        return response

    def _target_confidence(self, target_response):
        flat_response = target_response.flatten(2)
        if self.target_confidence_mode in ("peak_entropy", "calibrated"):
            peak_response = flat_response.max(dim=-1).values.view(target_response.shape[0], 1, 1, 1)
            peak_confidence = torch.sigmoid((peak_response - 1.0) / self.target_confidence_temperature)

            probabilities = torch.softmax(flat_response / self.target_confidence_temperature, dim=-1)
            entropy = -(probabilities * probabilities.clamp_min(1e-6).log()).sum(dim=-1, keepdim=True)
            max_entropy = math.log(max(flat_response.shape[-1], 2))
            entropy_confidence = 1.0 - entropy / max_entropy
            entropy_confidence = entropy_confidence.view(target_response.shape[0], 1, 1, 1)

            confidence = 0.70 * peak_confidence + 0.30 * entropy_confidence
            return confidence.clamp(0.0, 1.0)

        probabilities = torch.softmax(flat_response / self.target_confidence_temperature, dim=-1)
        peak_probability = probabilities.max(dim=-1).values.view(target_response.shape[0], 1, 1, 1)
        uniform_probability = 1.0 / max(flat_response.shape[-1], 1)
        confidence = (peak_probability - uniform_probability) / (1.0 - uniform_probability)
        return confidence.clamp(0.0, 1.0)

    def _target_prototype(self, template_tokens):
        bsz, token_count, _ = template_tokens.shape
        side = int(math.sqrt(token_count))
        if side * side != token_count:
            return template_tokens.mean(dim=1)

        dtype = template_tokens.dtype
        device = template_tokens.device
        coords = torch.linspace(-1.0, 1.0, side, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        dist = xx.square() + yy.square()
        weights = torch.exp(-dist / (2.0 * self.target_sigma * self.target_sigma)).reshape(1, token_count, 1)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return (template_tokens * weights).sum(dim=1).view(bsz, -1)


def build_multi_scale_token_fusion(cfg, hidden_dim):
    mstf_cfg = getattr(cfg.MODEL, "MSTF", None)
    if mstf_cfg is None or not getattr(mstf_cfg, "ENABLE", False):
        return None
    return MultiScaleTokenFusion(
        dim=hidden_dim,
        local_kernel=int(getattr(mstf_cfg, "LOCAL_KERNEL", 3)),
        mid_kernel=int(getattr(mstf_cfg, "MID_KERNEL", 5)),
        dilated_kernel=int(getattr(mstf_cfg, "DILATED_KERNEL", 3)),
        dilation=int(getattr(mstf_cfg, "DILATION", 2)),
        reduction=int(getattr(mstf_cfg, "REDUCTION", 4)),
        adaptive_scale=bool(getattr(mstf_cfg, "ADAPTIVE_SCALE", True)),
        gate_reduction=int(getattr(mstf_cfg, "GATE_REDUCTION", 16)),
        gate_temperature=float(getattr(mstf_cfg, "GATE_TEMPERATURE", 1.0)),
        local_bias=float(getattr(mstf_cfg, "LOCAL_BIAS", 0.40)),
        mid_bias=float(getattr(mstf_cfg, "MID_BIAS", 0.20)),
        context_bias=float(getattr(mstf_cfg, "CONTEXT_BIAS", -0.20)),
        context_gate=bool(getattr(mstf_cfg, "CONTEXT_GATE", True)),
        context_gate_bias=float(getattr(mstf_cfg, "CONTEXT_GATE_BIAS", -1.50)),
        residual_confidence_gate=bool(getattr(mstf_cfg, "RESIDUAL_CONFIDENCE_GATE", True)),
        residual_floor=float(getattr(mstf_cfg, "RESIDUAL_FLOOR", 0.65)),
        residual_init=float(getattr(mstf_cfg, "RESIDUAL_INIT", 0.10)),
        dropout=float(getattr(mstf_cfg, "DROPOUT", 0.0)),
        target_guided=bool(getattr(mstf_cfg, "TARGET_GUIDED", False)),
        spatial_adaptive=bool(getattr(mstf_cfg, "SPATIAL_ADAPTIVE", False)),
        spatial_blend=float(getattr(mstf_cfg, "SPATIAL_BLEND", -1.0)),
        target_sigma=float(getattr(mstf_cfg, "TARGET_SIGMA", 0.35)),
        target_temperature=float(getattr(mstf_cfg, "TARGET_TEMPERATURE", 4.0)),
        target_residual_floor=float(getattr(mstf_cfg, "TARGET_RESIDUAL_FLOOR", 0.30)),
        target_response_clamp=float(getattr(mstf_cfg, "TARGET_RESPONSE_CLAMP", 0.0)),
        target_confidence_gate=bool(getattr(mstf_cfg, "TARGET_CONFIDENCE_GATE", False)),
        target_confidence_mode=str(getattr(mstf_cfg, "TARGET_CONFIDENCE_MODE", "softmax_peak")),
        target_confidence_temperature=float(getattr(mstf_cfg, "TARGET_CONFIDENCE_TEMPERATURE", 1.0)),
        target_confidence_floor=float(getattr(mstf_cfg, "TARGET_CONFIDENCE_FLOOR", 0.80)),
        target_confidence_power=float(getattr(mstf_cfg, "TARGET_CONFIDENCE_POWER", 1.0)),
        context_confidence_gate=bool(getattr(mstf_cfg, "CONTEXT_CONFIDENCE_GATE", False)),
        context_confidence_floor=float(getattr(mstf_cfg, "CONTEXT_CONFIDENCE_FLOOR", 0.70)),
        scale_estimation=bool(getattr(mstf_cfg, "SCALE_ESTIMATION", False)),
        scale_prior_strength=float(getattr(mstf_cfg, "SCALE_PRIOR_STRENGTH", 0.25)),
        scale_prior_confidence_gate=bool(getattr(mstf_cfg, "SCALE_PRIOR_CONFIDENCE_GATE", False)),
        scale_uncertain_residual_floor=float(getattr(mstf_cfg, "SCALE_UNCERTAIN_RESIDUAL_FLOOR", 0.70)),
        branch_weight_floor=float(getattr(mstf_cfg, "BRANCH_WEIGHT_FLOOR", 0.0)),
    )
