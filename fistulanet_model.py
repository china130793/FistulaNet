"""
Fistula-Net model implementation.

This module contains the multi-sequence volumetric network used by the
execution pipeline. It implements sequence-specific 3D feature extraction,
local reliability-gated fusion, anatomical coordinate encoding, dual disease
and topology decoding, and graph-oriented feature heads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


@dataclass
class FistulaNetConfig:
    input_channels: int
    base_channels: int
    bottleneck_channels: int
    topology_channels: int
    segmentation_channels: int
    graph_hidden_dim: int
    attention_heads: int
    dropout: float

    @classmethod
    def from_dict(cls, cfg: Dict) -> "FistulaNetConfig":
        return cls(
            input_channels=int(cfg.get("input_channels", 5)),
            base_channels=int(cfg.get("base_channels", 18)),
            bottleneck_channels=int(cfg.get("bottleneck_channels", 96)),
            topology_channels=int(cfg.get("topology_channels", 8)),
            segmentation_channels=int(cfg.get("segmentation_channels", 7)),
            graph_hidden_dim=int(cfg.get("graph_hidden_dim", 64)),
            attention_heads=int(cfg.get("attention_heads", 4)),
            dropout=float(cfg.get("dropout", 0.12)),
        )


def _ensure_5d(x: Tensor) -> Tensor:
    if x.ndim != 5:
        raise ValueError(f"Expected tensor with shape [B, C, D, H, W], received {tuple(x.shape)}")
    return x


class LayerNorm3D(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, channels, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x)


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=False)
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.act = nn.SiLU(inplace=True)
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.act(self.norm(self.conv(x))))


class ResidualBlock3D(nn.Module):
    def __init__(self, channels: int, dilation: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False)
        self.norm1 = nn.InstanceNorm3d(channels, affine=True)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(channels, affine=True)
        self.act = nn.SiLU(inplace=True)
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, max(channels // 4, 4), 1),
            nn.SiLU(inplace=True),
            nn.Conv3d(max(channels // 4, 4), channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.act(self.norm1(self.conv1(x)))
        y = self.drop(y)
        y = self.norm2(self.conv2(y))
        y = y * self.gate(y)
        return self.act(x + y)


class AxialWindowAttention3D(nn.Module):
    def __init__(self, channels: int, heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.channels = channels
        self.heads = max(1, heads)
        self.head_dim = channels // self.heads
        if self.head_dim * self.heads != channels:
            self.heads = 1
            self.head_dim = channels
        self.qkv = nn.Conv3d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv3d(channels, channels, 1, bias=False)
        self.norm = LayerNorm3D(channels)
        self.drop = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def _attention_over_depth(self, x: Tensor) -> Tensor:
        b, c, d, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, self.heads, self.head_dim, d, h, w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q = q.permute(0, 3, 1, 4, 5, 2).reshape(b * d, self.heads, h * w, self.head_dim)
        k = k.permute(0, 3, 1, 4, 5, 2).reshape(b * d, self.heads, h * w, self.head_dim)
        v = v.permute(0, 3, 1, 4, 5, 2).reshape(b * d, self.heads, h * w, self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = self.drop(torch.softmax(attn, dim=-1))
        out = torch.matmul(attn, v)
        out = out.reshape(b, d, self.heads, h, w, self.head_dim).permute(0, 2, 5, 1, 3, 4)
        out = out.reshape(b, c, d, h, w)
        return self.proj(out)

    def forward(self, x: Tensor) -> Tensor:
        return x + self._attention_over_depth(x)


class DownStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float, attention_heads: int) -> None:
        super().__init__()
        self.down = ConvNormAct(in_channels, out_channels, kernel_size=3, stride=2, dropout=dropout)
        self.res1 = ResidualBlock3D(out_channels, dilation=1, dropout=dropout)
        self.res2 = ResidualBlock3D(out_channels, dilation=2, dropout=dropout)
        self.attn = AxialWindowAttention3D(out_channels, heads=attention_heads, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.down(x)
        x = self.res1(x)
        x = self.res2(x)
        return self.attn(x)


class UpStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.fuse = ConvNormAct(out_channels + skip_channels, out_channels, dropout=dropout)
        self.res = ResidualBlock3D(out_channels, dropout=dropout)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        dz = skip.shape[2] - x.shape[2]
        dy = skip.shape[3] - x.shape[3]
        dx = skip.shape[4] - x.shape[4]
        if dz or dy or dx:
            x = F.pad(x, [0, max(dx, 0), 0, max(dy, 0), 0, max(dz, 0)])
            x = x[:, :, : skip.shape[2], : skip.shape[3], : skip.shape[4]]
        x = torch.cat([x, skip], dim=1)
        return self.res(self.fuse(x))


class SequenceSpecificEncoder(nn.Module):
    def __init__(self, base_channels: int, dropout: float, attention_heads: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct(1, base_channels, 3, 1, dropout=dropout),
            ResidualBlock3D(base_channels, dropout=dropout),
        )
        self.stage1 = DownStage(base_channels, base_channels * 2, dropout, attention_heads)
        self.stage2 = DownStage(base_channels * 2, base_channels * 4, dropout, attention_heads)
        self.stage3 = DownStage(base_channels * 4, base_channels * 6, dropout, attention_heads)
        self.out_channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 6]

    def forward(self, x: Tensor) -> List[Tensor]:
        x0 = self.stem(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        return [x0, x1, x2, x3]


class ReliabilityEstimator(nn.Module):
    def __init__(self, channels: int, modalities: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(channels * modalities, channels, 1),
            nn.InstanceNorm3d(channels, affine=True),
            nn.SiLU(inplace=True),
            nn.Conv3d(channels, modalities, 1),
        )

    def forward(self, features: List[Tensor]) -> Tensor:
        concat = torch.cat(features, dim=1)
        return torch.softmax(self.net(concat), dim=1)


class ReliabilityGatedFusion(nn.Module):
    def __init__(self, scale_channels: List[int], modalities: int) -> None:
        super().__init__()
        self.estimators = nn.ModuleList([ReliabilityEstimator(ch, modalities) for ch in scale_channels])
        self.post = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(ch, ch, 1, bias=False),
                nn.InstanceNorm3d(ch, affine=True),
                nn.SiLU(inplace=True),
            )
            for ch in scale_channels
        ])

    def forward(self, modality_features: List[List[Tensor]]) -> Tuple[List[Tensor], List[Tensor]]:
        if not modality_features:
            raise ValueError("No modality features supplied")
        levels = len(modality_features[0])
        fused: List[Tensor] = []
        weights: List[Tensor] = []
        for level in range(levels):
            level_features = [mf[level] for mf in modality_features]
            alpha = self.estimators[level](level_features)
            weighted = 0
            for m, feat in enumerate(level_features):
                weighted = weighted + alpha[:, m : m + 1] * feat
            fused.append(self.post[level](weighted))
            weights.append(alpha)
        return fused, weights


class CoordinateFieldEncoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, dropout: float) -> None:
        super().__init__()
        self.extract = nn.Sequential(
            ConvNormAct(in_channels, base_channels, 3, 1, dropout=dropout),
            ResidualBlock3D(base_channels, dilation=1, dropout=dropout),
            ResidualBlock3D(base_channels, dilation=2, dropout=dropout),
        )
        self.anatomy_logits = nn.Conv3d(base_channels, 5, 1)
        self.coord_regression = nn.Conv3d(base_channels, 6, 1)
        self.distance_refiner = nn.Sequential(
            ConvNormAct(base_channels + 6, base_channels, 3, 1, dropout=dropout),
            nn.Conv3d(base_channels, 6, 1),
        )

    def forward(self, fused_level0: Tensor, coordinate_priors: Optional[Tensor] = None) -> Dict[str, Tensor]:
        feat = self.extract(fused_level0)
        coords = torch.tanh(self.coord_regression(feat))
        if coordinate_priors is not None:
            coordinate_priors = F.interpolate(coordinate_priors, size=coords.shape[2:], mode="trilinear", align_corners=False)
            coords = 0.55 * coords + 0.45 * coordinate_priors
        refined = coords + 0.15 * torch.tanh(self.distance_refiner(torch.cat([feat, coords], dim=1)))
        anatomy = self.anatomy_logits(feat)
        return {"anatomy_logits": anatomy, "coordinate_field": refined, "coordinate_features": feat}


class DiseaseDecoder(nn.Module):
    def __init__(self, channels: List[int], out_channels: int, dropout: float) -> None:
        super().__init__()
        c0, c1, c2, c3 = channels
        self.bridge = nn.Sequential(
            ConvNormAct(c3, c3, 3, 1, dropout=dropout),
            ResidualBlock3D(c3, dilation=2, dropout=dropout),
        )
        self.up2 = UpStage(c3, c2, c2, dropout)
        self.up1 = UpStage(c2, c1, c1, dropout)
        self.up0 = UpStage(c1, c0, c0, dropout)
        self.refine = nn.Sequential(
            ConvNormAct(c0 + 6, c0, 3, 1, dropout=dropout),
            ResidualBlock3D(c0, dropout=dropout),
            nn.Conv3d(c0, out_channels, 1),
        )

    def forward(self, fused: List[Tensor], coordinate_field: Tensor) -> Tensor:
        x0, x1, x2, x3 = fused
        x = self.bridge(x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.up0(x, x0)
        coords = F.interpolate(coordinate_field, size=x.shape[2:], mode="trilinear", align_corners=False)
        return self.refine(torch.cat([x, coords], dim=1))


class TopologyDecoder(nn.Module):
    def __init__(self, channels: List[int], out_channels: int, dropout: float) -> None:
        super().__init__()
        c0, c1, c2, c3 = channels
        self.bridge = nn.Sequential(
            ConvNormAct(c3, c3, 3, 1, dropout=dropout),
            ResidualBlock3D(c3, dilation=3, dropout=dropout),
            AxialWindowAttention3D(c3, heads=4, dropout=dropout),
        )
        self.up2 = UpStage(c3, c2, c2, dropout)
        self.up1 = UpStage(c2, c1, c1, dropout)
        self.up0 = UpStage(c1, c0, c0, dropout)
        self.centerline_head = nn.Conv3d(c0 + 6, 1, 1)
        self.opening_head = nn.Conv3d(c0 + 6, 2, 1)
        self.branch_head = nn.Conv3d(c0 + 6, 1, 1)
        self.crossing_head = nn.Conv3d(c0 + 6, 1, 1)
        self.abscess_link_head = nn.Conv3d(c0 + 6, 1, 1)
        self.topology_logits = nn.Conv3d(c0 + 6, out_channels, 1)

    def forward(self, fused: List[Tensor], coordinate_field: Tensor) -> Dict[str, Tensor]:
        x0, x1, x2, x3 = fused
        x = self.bridge(x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.up0(x, x0)
        coords = F.interpolate(coordinate_field, size=x.shape[2:], mode="trilinear", align_corners=False)
        z = torch.cat([x, coords], dim=1)
        return {
            "centerline_logits": self.centerline_head(z),
            "opening_logits": self.opening_head(z),
            "branch_logits": self.branch_head(z),
            "sphincter_crossing_logits": self.crossing_head(z),
            "abscess_link_logits": self.abscess_link_head(z),
            "topology_logits": self.topology_logits(z),
            "topology_features": x,
        }


class GraphFeatureProjector(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv3d(in_channels + 6, hidden_dim, 1),
            nn.InstanceNorm3d(hidden_dim, affine=True),
            nn.SiLU(inplace=True),
            nn.Dropout3d(dropout),
            nn.Conv3d(hidden_dim, hidden_dim, 1),
        )

    def forward(self, topology_features: Tensor, coordinate_field: Tensor) -> Tensor:
        coords = F.interpolate(coordinate_field, size=topology_features.shape[2:], mode="trilinear", align_corners=False)
        return self.project(torch.cat([topology_features, coords], dim=1))


class SphincterAwareGraphAttention(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = max(1, heads)
        if hidden_dim % self.heads != 0:
            self.heads = 1
        self.head_dim = hidden_dim // self.heads
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.edge_bias = nn.Sequential(nn.Linear(8, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, self.heads))
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, node_features: Tensor, edge_features: Optional[Tensor] = None) -> Tensor:
        if node_features.ndim != 3:
            raise ValueError("node_features must have shape [B, N, C]")
        b, n, c = node_features.shape
        q = self.q(self.norm(node_features)).reshape(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(self.norm(node_features)).reshape(b, n, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(self.norm(node_features)).reshape(b, n, self.heads, self.head_dim).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if edge_features is not None:
            if edge_features.shape[:3] != (b, n, n):
                raise ValueError("edge_features must have shape [B, N, N, F]")
            bias = self.edge_bias(edge_features).permute(0, 3, 1, 2)
            logits = logits + bias
        attn = self.drop(torch.softmax(logits, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, n, c)
        return node_features + self.proj(out)


class GraphReasoningHead(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attn1 = SphincterAwareGraphAttention(hidden_dim, heads, dropout)
        self.attn2 = SphincterAwareGraphAttention(hidden_dim, heads, dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.class_head = nn.Linear(hidden_dim, 6)
        self.complexity_head = nn.Linear(hidden_dim, 1)
        self.eas_head = nn.Linear(hidden_dim, 1)
        self.branch_head = nn.Linear(hidden_dim, 1)
        self.abscess_head = nn.Linear(hidden_dim, 1)

    def forward(self, node_features: Tensor, edge_features: Optional[Tensor] = None) -> Dict[str, Tensor]:
        x = self.attn1(node_features, edge_features)
        x = x + self.ffn(x)
        x = self.attn2(x, edge_features)
        pooled = x.mean(dim=1)
        return {
            "graph_class_logits": self.class_head(pooled),
            "graph_complexity": torch.sigmoid(self.complexity_head(pooled)),
            "eas_involvement": torch.sigmoid(self.eas_head(pooled)) * 100.0,
            "branch_burden": F.softplus(self.branch_head(pooled)),
            "abscess_communication_probability": torch.sigmoid(self.abscess_head(pooled)),
        }


class FistulaNet(nn.Module):
    def __init__(self, cfg: FistulaNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoders = nn.ModuleList([
            SequenceSpecificEncoder(cfg.base_channels, cfg.dropout, cfg.attention_heads)
            for _ in range(cfg.input_channels)
        ])
        scale_channels = self.encoders[0].out_channels
        self.fusion = ReliabilityGatedFusion(scale_channels, cfg.input_channels)
        self.coordinate_encoder = CoordinateFieldEncoder(scale_channels[0], cfg.base_channels, cfg.dropout)
        self.disease_decoder = DiseaseDecoder(scale_channels, cfg.segmentation_channels, cfg.dropout)
        self.topology_decoder = TopologyDecoder(scale_channels, cfg.topology_channels, cfg.dropout)
        self.graph_projector = GraphFeatureProjector(scale_channels[0], cfg.graph_hidden_dim, cfg.dropout)
        self.graph_head = GraphReasoningHead(cfg.graph_hidden_dim, cfg.attention_heads, cfg.dropout)

    def encode_modalities(self, x: Tensor) -> List[List[Tensor]]:
        x = _ensure_5d(x)
        if x.shape[1] != len(self.encoders):
            raise ValueError(f"Expected {len(self.encoders)} modalities, received {x.shape[1]}")
        return [encoder(x[:, idx : idx + 1]) for idx, encoder in enumerate(self.encoders)]

    def select_graph_nodes(self, graph_feature_map: Tensor, topology_outputs: Dict[str, Tensor], max_nodes: int = 32) -> Tensor:
        b, c, d, h, w = graph_feature_map.shape
        centerline = torch.sigmoid(topology_outputs["centerline_logits"]).reshape(b, -1)
        branch = torch.sigmoid(topology_outputs["branch_logits"]).reshape(b, -1)
        crossing = torch.sigmoid(topology_outputs["sphincter_crossing_logits"]).reshape(b, -1)
        score = centerline + 0.7 * branch + 0.6 * crossing
        k = min(max_nodes, score.shape[1])
        idx = torch.topk(score, k=k, dim=1).indices
        flat = graph_feature_map.reshape(b, c, -1).transpose(1, 2)
        gathered = []
        for batch in range(b):
            gathered.append(flat[batch, idx[batch]])
        return torch.stack(gathered, dim=0)

    def build_dense_edge_features(self, node_features: Tensor) -> Tensor:
        b, n, c = node_features.shape
        diff = node_features.unsqueeze(2) - node_features.unsqueeze(1)
        dist = torch.norm(diff, dim=-1, keepdim=True)
        sim = F.cosine_similarity(node_features.unsqueeze(2), node_features.unsqueeze(1), dim=-1).unsqueeze(-1)
        deg_hint = torch.linspace(0, 1, n, device=node_features.device, dtype=node_features.dtype)
        row = deg_hint.view(1, n, 1, 1).expand(b, n, n, 1)
        col = deg_hint.view(1, 1, n, 1).expand(b, n, n, 1)
        invdist = 1.0 / (dist + 1.0)
        tri = torch.minimum(row, col)
        edge = torch.cat([dist, invdist, sim, row, col, torch.abs(row - col), tri, torch.ones_like(dist)], dim=-1)
        return edge

    def forward(self, x: Tensor, coordinate_priors: Optional[Tensor] = None) -> Dict[str, Tensor]:
        modality_features = self.encode_modalities(x)
        fused, reliability_weights = self.fusion(modality_features)
        coordinate_outputs = self.coordinate_encoder(fused[0], coordinate_priors=coordinate_priors)
        disease_logits = self.disease_decoder(fused, coordinate_outputs["coordinate_field"])
        topology_outputs = self.topology_decoder(fused, coordinate_outputs["coordinate_field"])
        graph_feature_map = self.graph_projector(topology_outputs["topology_features"], coordinate_outputs["coordinate_field"])
        node_features = self.select_graph_nodes(graph_feature_map, topology_outputs, max_nodes=32)
        edge_features = self.build_dense_edge_features(node_features)
        graph_outputs = self.graph_head(node_features, edge_features)
        return {
            "segmentation_logits": disease_logits,
            "tract_probability": torch.sigmoid(disease_logits[:, 0:1]),
            "secondary_probability": torch.sigmoid(disease_logits[:, 1:2]),
            "abscess_probability": torch.sigmoid(disease_logits[:, 2:3]),
            "ias_probability": torch.sigmoid(disease_logits[:, 3:4]),
            "eas_probability": torch.sigmoid(disease_logits[:, 4:5]),
            "inflammation_probability": torch.sigmoid(disease_logits[:, 5:6]),
            "anatomy_logits": coordinate_outputs["anatomy_logits"],
            "coordinate_field": coordinate_outputs["coordinate_field"],
            "reliability_weights": reliability_weights,
            **topology_outputs,
            **graph_outputs,
        }


def create_model(config_dict: Dict) -> FistulaNet:
    return FistulaNet(FistulaNetConfig.from_dict(config_dict))


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_encoders(model: FistulaNet, modality_indices: Optional[Iterable[int]] = None) -> None:
    indices = set(range(len(model.encoders))) if modality_indices is None else set(modality_indices)
    for idx, encoder in enumerate(model.encoders):
        if idx in indices:
            for parameter in encoder.parameters():
                parameter.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True


def model_signature(model: FistulaNet) -> Dict[str, int]:
    return {
        "trainable_parameters": count_trainable_parameters(model),
        "modalities": len(model.encoders),
        "base_channels": model.cfg.base_channels,
        "segmentation_heads": model.cfg.segmentation_channels,
        "topology_heads": model.cfg.topology_channels,
        "graph_hidden_dim": model.cfg.graph_hidden_dim,
    }
