"""Neural architectures needed for checkpoint-only inference.

This module intentionally contains no optimizer, loss, backward pass, data
generation, or checkpoint-writing routine.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.fc2(self.dropout(self.fc1(values)))
        if hidden.shape[-1] == values.shape[-1]:
            hidden = hidden + values
        return self.norm(hidden)


class TGCMInferenceModel(nn.Module):
    """Exact checkpoint architecture, exposing only assignment inference."""

    def __init__(
        self,
        vocab_size: int,
        k_max: int = 6,
        d_model: int = 64,
        nhead: int = 8,
        nlayers: int = 8,
        timesteps: int = 10,
        dropout: float = 0.15,
        fused_type: str = "adaln",
    ):
        super().__init__()
        self.k_max = int(k_max)
        self.fused_type = str(fused_type)
        self.pre_ln = nn.LayerNorm(d_model)
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.time_emb = nn.Embedding(timesteps + 1, d_model, padding_idx=0)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, 2 * d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.head_noise = nn.Sequential(
            ResidualBlock(d_model, d_model, dropout),
            ResidualBlock(d_model, d_model, dropout),
        )
        self.head_aptid = nn.Sequential(
            ResidualBlock(d_model, d_model, dropout),
            ResidualBlock(d_model, k_max + 1, dropout),
        )

    @staticmethod
    def _position_encoding(length: int, width: int, device: torch.device) -> torch.Tensor:
        position = torch.arange(length, device=device).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, width, 2, device=device) * (-math.log(10000.0) / width)
        )
        result = torch.zeros(length, width, device=device)
        result[:, 0::2] = torch.sin(position * divisor)
        result[:, 1::2] = torch.cos(position * divisor)
        return result

    def forward(
        self, token_ids: torch.Tensor, timestep: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if timestep.ndim > 1:
            timestep = timestep.squeeze(-1)
        embedded = self.tok_emb(token_ids)
        batch, length, width = embedded.shape
        position = self._position_encoding(length, width, embedded.device).unsqueeze(0)
        if self.fused_type == "add":
            time = self.time_emb(timestep).unsqueeze(1).expand(batch, length, width)
            hidden = self.enc(embedded + time + position, src_key_padding_mask=~mask)
        elif self.fused_type == "adaln":
            gamma, beta = self.time_mlp(self.time_emb(timestep)).chunk(2, dim=-1)
            normalized = self.pre_ln(embedded + position)
            conditioned = normalized * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
            hidden = self.pre_ln(
                self.enc(conditioned, src_key_padding_mask=~mask)
            )
        else:
            raise ValueError(f"Unsupported fused_type={self.fused_type!r}")
        return self.head_aptid(hidden)

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str | torch.device = "cpu"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        config = payload["config"]
        model = cls(**config)
        model.load_state_dict(payload["state_dict"], strict=True)
        return model.to(device).eval(), payload


class DANetAPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 64,
        embedding_k: int = 64,
        n_layers: int = 2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            n_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, embedding_k)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(self.embedding(token_ids))
        return torch.tanh(self.fc(output))

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str | torch.device = "cpu"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(**payload["config"])
        model.load_state_dict(payload["state_dict"], strict=True)
        return model.to(device).eval(), payload


class ConvModule(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(channels, 2 * channels, 1)
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=channels,
        )
        self.norm = nn.GroupNorm(1, channels)
        self.act = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = nn.functional.glu(self.pointwise_conv1(values), dim=1)
        values = self.depthwise_conv(values)
        values = self.pointwise_conv2(self.act(self.norm(values)))
        return self.dropout(values) + residual


class AttentionModule(nn.Module):
    def __init__(self, channels: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.attn = nn.MultiheadAttention(channels, n_heads, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        transposed = self.norm(values).transpose(1, 2)
        attended, _ = self.attn(transposed, transposed, transposed, need_weights=False)
        return self.dropout(attended.transpose(1, 2)) + residual


class MossFormerBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 17, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.conv = ConvModule(channels, kernel_size, dropout)
        self.attn = AttentionModule(channels, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Conv1d(channels, 2 * channels, 1),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(2 * channels, channels, 1),
        )
        self.norm_ffn = nn.GroupNorm(1, channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.attn(self.conv(values))
        return self.ffn(self.norm_ffn(values)) + values


class MossFormer2PIT(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, depth: int = 4, num_apts: int = 6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.blocks = nn.ModuleList([MossFormerBlock(embed_dim) for _ in range(depth)])
        self.classifier = nn.Conv1d(embed_dim, num_apts, 1)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(token_ids).transpose(1, 2)
        for block in self.blocks:
            hidden = block(hidden)
        return self.classifier(hidden).transpose(1, 2)

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str | torch.device = "cpu"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(**payload["config"])
        model.load_state_dict(payload["state_dict"], strict=True)
        return model.to(device).eval(), payload
