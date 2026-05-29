# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Project-owned portions of this file are licensed under CC-BY-NC-SA-4.0.
# See LICENSE and NOTICE for details. Third-party notices remain applicable.

# heads/nocs_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dpt_head import DPTHead
from .head_act import activate_head  # Existing implementation: contains expp1, etc.
from geo_models.modules import PointNet2MSG

class NOCSPredictor(nn.Module):
    """
    NOCS Predictor module - simplified with FiLM conditioning on z_obj
    Removes hardcoded category indexing for better generalization
    """
    def __init__(self, features: int = 256, bins_num: int = 64, use_self_attn: bool = True, d_obj: int = 256, use_film_conditioning: bool = True):
        super().__init__()
        self.bins_num = bins_num
        self.use_self_attn = use_self_attn
        self.use_film_conditioning = use_film_conditioning
        
        # NOCS prediction parameters (device-agnostic using register_buffer)
        bin_length = 1.0 / bins_num
        half_bin = bin_length / 2
        bins = torch.linspace(-0.5, 0.5 - bin_length, bins_num) + half_bin
        self.register_buffer('bins_center', bins.view(bins_num, 1))
        
        # Feature processing MLP with BatchNorm for better training stability
        self.nocs_mlp = nn.Sequential(
            nn.Conv1d(features, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, features, 1),
            nn.BatchNorm1d(features),
        )

        # FiLM conditioning: project z_obj to affine parameters (gamma, beta)
        if self.use_film_conditioning:
            self.cond_proj = nn.Linear(d_obj, 2 * features)
            self.cond_proj_head = nn.Linear(d_obj, 2 * 512)

        # Shared prediction trunk (no category-specific branches)
        self.head_conv1 = nn.Conv1d(features, 512, 1)
        self.head_conv2 = nn.Conv1d(512, 512, 1)
        self.head_out = nn.Conv1d(512, 3 * bins_num, 1)
        
        # Optional self-attention layer for feature refinement
        if self.use_self_attn:
            from .self_attn import SelfAttnLayer, SelfAttnLayerConfig
            attn_cfg = SelfAttnLayerConfig(
                block_num=2,
                d_model=features,
                num_head=4,
                dim_ffn=features
            )
            self.self_attn_layer = SelfAttnLayer(attn_cfg)
    
    def forward(self, kpt_features: torch.Tensor, z_obj: torch.Tensor = None) -> torch.Tensor:
        """
        Simplified forward pass using only FiLM conditioning on z_obj.
        
        Args:
            kpt_features: [BS, K, features] keypoint features
            z_obj: [BS, d_obj] object latent embedding (L2-normalized), optional if use_film_conditioning=False
        Returns:
            kpt_nocs: [BS, K, 3] predicted NOCS coordinates
        """
        BS, K, _ = kpt_features.shape
        
        # Process features through MLP with BatchNorm
        x = self.nocs_mlp(kpt_features.transpose(1, 2)).transpose(1, 2)  # [BS, K, features]
        
        # Apply self-attention for feature refinement (optional)
        if self.use_self_attn:
            x = self.self_attn_layer(x)  # [BS, K, features]
        
        x = x.transpose(1, 2)  # [BS, features, K]
        
        # FiLM modulation: condition on object embedding
        if self.use_film_conditioning:
            if z_obj is None:
                raise ValueError("z_obj must be provided when use_film_conditioning=True")
            gamma, beta = self.cond_proj(z_obj).chunk(2, dim=-1)  # Each [BS, features]
            x = x * (1 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)
        
        # Predict bin logits and convert to NOCS coordinates
        x = F.relu(self.head_conv1(x))
        x = F.relu(self.head_conv2(x))
        if self.use_film_conditioning:
            gamma_h, beta_h = self.cond_proj_head(z_obj).chunk(2, dim=-1)  # Each [BS, 512]
            x = x * (1 + gamma_h.unsqueeze(-1)) + beta_h.unsqueeze(-1)
        logits = self.head_out(x)  # [BS, 3*bins_num, K]
        attn = logits.view(BS, 3, self.bins_num, K)
        attn = torch.softmax(attn, dim=2).permute(0, 3, 1, 2)  # [BS, K, 3, bins_num]
        kpt_nocs = torch.matmul(attn, self.bins_center).squeeze(-1)  # [BS, K, 3]
        
        return kpt_nocs


class NOCSHead(nn.Module):
    """
    NOCS Head - now simplified to focus on feature extraction and keypoint processing.
    Following AG-pose design pattern where feature processing is moved to main model.
    """

    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        features: int = 256,
        intermediate_layer_idx = [4, 11, 17, 23],
        down_ratio: int = 1,
        # NOCS prediction
        bins_num: int = 64,
        use_self_attn: bool = True,  # Use self-attention in NOCS predictor
        d_obj: int = 256,
        use_film_conditioning: bool = True,  # Enable/disable FiLM conditioning
    ) -> None:
        super().__init__()

        # Visual feature extractor (DPT backbone)
        self.visual_extractor = DPTHead(
            dim_in=dim_in,
            patch_size=patch_size,
            features=features,
            intermediate_layer_idx=intermediate_layer_idx,
            pos_embed=False,
            feature_only=True,
            down_ratio=down_ratio,
        )
        
        # NOCS predictor with FiLM conditioning (no category indexing)
        self.nocs_predictor = NOCSPredictor(
            features=features, 
            bins_num=bins_num,
            use_self_attn=use_self_attn,
            d_obj=d_obj,
            use_film_conditioning=use_film_conditioning
        )
        
    def extract_visual_features(
        self,
        aggregated_tokens_list,
        images,
        patch_start_idx,
        choose_indices,
        visual_projector,
        frames_chunk_size: int = 8,
        return_dense_feats: bool = False,
    ):
        """Extract dense visual features using DPT backbone"""
        # Extract raw visual features using DPT backbone
        visual_feats = self.visual_extractor(
            aggregated_tokens_list, images, patch_start_idx, frames_chunk_size=frames_chunk_size
        )  # [B, S, C, H, W]
        
        # Process visual features for keypoint detection
        B, S, C, H, W = visual_feats.shape
        N = H * W
        BS = B * S
        
        # Flatten choose indices and visual features
        choose_indices_flat = choose_indices.reshape(BS, -1)  # [BS, K_dense]
        visual_feats_flat = visual_feats.reshape(BS, C, N)  # [BS, C, N]
        
        # Gather visual features at chosen indices
        gather_idxC = choose_indices_flat.unsqueeze(1).expand(-1, C, -1)  # [BS, C, K_dense]
        rgb_local = torch.gather(visual_feats_flat, 2, gather_idxC)  # [BS, C, K_dense]
        
        # Apply visual projector
        rgb_local = visual_projector(rgb_local)  # [BS, K_kpt_num, K_dense]
        
        if return_dense_feats:
            return rgb_local, visual_feats
        return rgb_local
    
    def predict_nocs(self, kpt_features: torch.Tensor, z_obj: torch.Tensor = None) -> torch.Tensor:
        """
        Predict NOCS coordinates from keypoint features conditioned on object embedding.
        
        Args:
            kpt_features: [BS, K, features] keypoint features
            z_obj: [BS, d_obj] object latent embedding, optional if use_film_conditioning=False
        Returns:
            kpt_nocs: [BS, K, 3] predicted NOCS coordinates
        """
        return self.nocs_predictor(kpt_features, z_obj)
