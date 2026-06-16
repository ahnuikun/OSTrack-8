import torch
from torch import nn


class MultiScaleTokenFusion(nn.Module):
    """Multi-scale search-token fusion that preserves OSTrack token layout.

    The module receives reshaped search tokens as B x C x H x W and returns the
    same shape. It is inserted before the OSTrack box head, so the ViT token
    count, head input size, and test pipeline remain unchanged.
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
        local_bias=0.55,
        mid_bias=0.35,
        context_bias=-1.20,
        context_gate=True,
        context_gate_bias=-1.50,
        residual_confidence_gate=True,
        residual_floor=0.65,
        residual_init=0.07,
        dropout=0.0,
    ):
        super().__init__()
        hidden_dim = max(dim // int(reduction), 64)
        self.adaptive_scale = bool(adaptive_scale)
        self.gate_temperature = max(float(gate_temperature), 1e-3)
        self.use_context_gate = bool(context_gate)
        self.residual_confidence_gate = bool(residual_confidence_gate)
        self.residual_floor = min(max(float(residual_floor), 0.0), 1.0)

        self.reduce = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=True)
        self.local_branch = self._depthwise_branch(hidden_dim, int(local_kernel), dilation=1)
        self.mid_branch = self._depthwise_branch(hidden_dim, int(mid_kernel), dilation=1)
        self.context_branch = self._depthwise_branch(hidden_dim, int(dilated_kernel), dilation=int(dilation))

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

    def forward(self, search_feat):
        reduced = self.reduce(search_feat)
        local_feat = self.local_branch(reduced)
        mid_feat = self.mid_branch(reduced)
        context_feat = self.context_branch(reduced)
        branches = [reduced, local_feat, mid_feat, context_feat]
        weights = None

        if self.scale_selector is not None:
            weights = self._branch_weights(reduced)
            context_weight = weights[:, 2:3]
            if self.context_gate is not None:
                context_weight = context_weight * self.context_gate(reduced)
            branches = [
                reduced,
                local_feat * weights[:, 0:1],
                mid_feat * weights[:, 1:2],
                context_feat * context_weight,
            ]
        elif self.context_gate is not None:
            branches[-1] = context_feat * self.context_gate(reduced)

        fused = self.fuse(torch.cat(branches, dim=1))
        residual_strength = self.residual_scale
        if self.residual_confidence_gate and weights is not None:
            confidence = weights.max(dim=1, keepdim=True).values
            residual_strength = residual_strength * (
                self.residual_floor + (1.0 - self.residual_floor) * confidence
            )
        return search_feat + residual_strength * fused

    def _branch_weights(self, reduced):
        logits = self.scale_selector(reduced) / self.gate_temperature
        return torch.softmax(logits, dim=1)


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
        local_bias=float(getattr(mstf_cfg, "LOCAL_BIAS", 0.55)),
        mid_bias=float(getattr(mstf_cfg, "MID_BIAS", 0.35)),
        context_bias=float(getattr(mstf_cfg, "CONTEXT_BIAS", -1.20)),
        context_gate=bool(getattr(mstf_cfg, "CONTEXT_GATE", True)),
        context_gate_bias=float(getattr(mstf_cfg, "CONTEXT_GATE_BIAS", -1.50)),
        residual_confidence_gate=bool(getattr(mstf_cfg, "RESIDUAL_CONFIDENCE_GATE", True)),
        residual_floor=float(getattr(mstf_cfg, "RESIDUAL_FLOOR", 0.65)),
        residual_init=float(getattr(mstf_cfg, "RESIDUAL_INIT", 0.07)),
        dropout=float(getattr(mstf_cfg, "DROPOUT", 0.0)),
    )
