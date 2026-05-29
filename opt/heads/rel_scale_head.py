# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Project-owned portions of this file are licensed under CC-BY-NC-SA-4.0.
# See LICENSE and NOTICE for details. Third-party notices remain applicable.

# vggt/heads/rel_scale_head.py
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class RelScaleHead(nn.Module):
    """
    Estimate per-frame log scale purely from RGB cues and the object latent embedding.

    This head fuses a global visual descriptor with the z_obj token produced by the
    NOCS pipeline to regress the average scale of each frame. It outputs both the
    log-scale prediction and a confidence score that can be used to mask unreliable
    frames during training or evaluation.

    """

    def __init__(
        self,
        z_obj_dim: int = 256,
        visual_dim: int = 256,
        meta_dim: int = 7,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.meta_dim = meta_dim

        self.visual_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.roi_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.z_proj = nn.Sequential(
            nn.LayerNorm(z_obj_dim),
            nn.Linear(z_obj_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.meta_proj = nn.Sequential(
            nn.LayerNorm(meta_dim),
            nn.Linear(meta_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.anchor_fuse = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.delta_fuse = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.anchor_head = nn.Linear(hidden_dim, 1)
        self.scale_head = nn.Linear(hidden_dim, 1)
        self.confidence_head = nn.Linear(hidden_dim, 1)

        nn.init.zeros_(self.anchor_head.bias)
        nn.init.zeros_(self.scale_head.bias)
        nn.init.zeros_(self.confidence_head.bias)

    def forward(
        self,
        z_obj: torch.Tensor,
        visual_feats: torch.Tensor,
        roi_feats: Optional[torch.Tensor] = None,
        meta_feats: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_obj: Tensor containing object embeddings. Shape [B, S, z_obj_dim] or [BS, z_obj_dim].
            visual_feats: Visual descriptors. Accepted shapes:
                - [B, S, visual_dim]
                - [B, S, visual_dim, H, W] (dense feature map, will be globally pooled)
                - [BS, visual_dim] (flattened batch/sequence, requires either z_obj or valid_mask to
                  provide the [B, S] shape)
            roi_feats: Object-focused descriptors with the same accepted shapes as visual_feats.
                If omitted, the global visual descriptor is reused.
            meta_feats: Optional metadata tensor with shape [B, S, meta_dim] or [BS, meta_dim].
                Expected to encode crop geometry / mask coverage / intrinsics cues.
            valid_mask: Optional boolean tensor of shape [B, S] marking frames to keep.

        Returns:
            log_scale: [B, S] tensor with per-frame log scale estimates (NaN for invalid frames).
            confidence: [B, S] tensor with per-frame confidence scores in [0, 1].
        """
        if visual_feats.dim() == 5:
            B, S, C, H, W = visual_feats.shape
            visual_flat = visual_feats.view(B * S, C, H * W).mean(dim=-1)
        elif visual_feats.dim() == 3:
            B, S, C = visual_feats.shape
            visual_flat = visual_feats.reshape(B * S, C)
        elif visual_feats.dim() == 2:
            if z_obj.dim() == 3:
                B, S = z_obj.shape[:2]
            elif valid_mask is not None and valid_mask.dim() == 2:
                B, S = valid_mask.shape
            else:
                raise ValueError(
                    "When passing flattened visual features you must also provide "
                    "z_obj with shape [B, S, ...] or a valid_mask of shape [B, S]."
                )
            visual_flat = visual_feats
        else:
            raise ValueError(
                f"visual_feats must have 3, 5, or 2 dimensions; received shape {visual_feats.shape}"
            )

        if roi_feats is None:
            roi_flat = visual_flat
        elif roi_feats.dim() == 5:
            roi_flat = roi_feats.view(B * S, roi_feats.shape[2], -1).mean(dim=-1)
        elif roi_feats.dim() == 3:
            roi_flat = roi_feats.reshape(B * S, roi_feats.shape[-1])
        elif roi_feats.dim() == 2:
            roi_flat = roi_feats
        else:
            raise ValueError(
                f"roi_feats must have 3, 5, or 2 dimensions; received shape {roi_feats.shape}"
            )

        if z_obj.dim() == 3:
            if z_obj.shape[0] != B or z_obj.shape[1] != S:
                raise ValueError(
                    f"z_obj shape {z_obj.shape} is incompatible with inferred batch shape {(B, S)}"
                )
            z_flat = z_obj.reshape(B * S, -1)
        elif z_obj.dim() == 2:
            if z_obj.shape[0] != B * S:
                raise ValueError(
                    f"Flattened z_obj has length {z_obj.shape[0]}, expected {B * S}"
                )
            z_flat = z_obj
        else:
            raise ValueError(f"Unsupported z_obj shape {z_obj.shape}")

        if meta_feats is None:
            meta_flat = visual_flat.new_zeros((B * S, self.meta_dim))
        elif meta_feats.dim() == 3:
            if meta_feats.shape[:2] != (B, S):
                raise ValueError(
                    f"meta_feats shape {meta_feats.shape} is incompatible with inferred batch shape {(B, S)}"
                )
            meta_flat = meta_feats.reshape(B * S, meta_feats.shape[-1])
        elif meta_feats.dim() == 2:
            if meta_feats.shape[0] != B * S:
                raise ValueError(
                    f"Flattened meta_feats has length {meta_feats.shape[0]}, expected {B * S}"
                )
            meta_flat = meta_feats
        else:
            raise ValueError(f"Unsupported meta_feats shape {meta_feats.shape}")

        visual_embed = self.visual_proj(visual_flat)
        roi_embed = self.roi_proj(roi_flat)
        z_embed = self.z_proj(z_flat)
        meta_embed = self.meta_proj(meta_flat)

        anchor_hidden = self.anchor_fuse(torch.cat([z_embed, meta_embed], dim=-1))
        delta_hidden = self.delta_fuse(
            torch.cat([visual_embed, roi_embed, z_embed, meta_embed], dim=-1)
        )

        log_scale_flat = (
            self.anchor_head(anchor_hidden).squeeze(-1)
            + self.scale_head(delta_hidden).squeeze(-1)
        )
        confidence_flat = torch.sigmoid(self.confidence_head(delta_hidden).squeeze(-1))

        if valid_mask is None:
            mask_flat = torch.ones(B * S, dtype=torch.bool, device=visual_flat.device)
        else:
            if valid_mask.shape != (B, S):
                raise ValueError(
                    f"valid_mask must have shape {(B, S)}, received {valid_mask.shape}"
                )
            mask_flat = valid_mask.reshape(B * S).to(dtype=torch.bool, device=visual_flat.device)

        log_scale_flat = log_scale_flat.masked_fill(~mask_flat, float("nan"))
        confidence_flat = confidence_flat.masked_fill(~mask_flat, 0.0)

        log_scale = log_scale_flat.view(B, S)
        confidence = confidence_flat.view(B, S)

        return log_scale, confidence