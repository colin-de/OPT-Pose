# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Project-owned portions of this file are licensed under CC-BY-NC-SA-4.0.
# See LICENSE and NOTICE for details. Third-party notices remain applicable.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from typing import Tuple


__all__ = ["ReconstructorHead"]


class ReconstructorHead(nn.Module):
    """
    Reconstructor head for point cloud reconstruction from keypoints.
    
    Reconstructs dense point clouds from sparse keypoints by predicting
    per-keypoint offsets that generate multiple points around each keypoint.
    
    Args:
        pts_per_kpt: Number of points to generate per keypoint (default: 8)
        ndim: Feature dimension for encoding (default: 256)
    """
    
    def __init__(self, pts_per_kpt: int = 8, ndim: int = 256):
        super().__init__()
        self.pts_per_kpt = pts_per_kpt
        self.ndim = ndim
        
        # Position encoding: 3D coordinates -> feature space
        self.pos_enc = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, self.ndim, 1),
        )
        
        # Feature processing MLP
        self.mlp = nn.Sequential(
            nn.Conv1d(self.ndim, self.ndim, 1),
            nn.ReLU(),
            nn.Conv1d(self.ndim, self.ndim, 1),
        )
        
        # Shape decoder: generates 3D offsets for reconstruction
        self.shape_decoder = nn.Sequential(
            nn.Conv1d(2 * self.ndim, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 3 * self.pts_per_kpt, 1),
        )

    def forward(self, kpt_3d: torch.Tensor, kpt_feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstruct point cloud from keypoints.
        
        Args:
            kpt_3d: Keypoint 3D coordinates [b, 3, kpt_num]
            kpt_feature: Keypoint features [b, c, kpt_num]
        
        Returns:
            recon_model: Reconstructed point cloud [b, 3, pts_per_kpt*kpt_num]
            recon_delta: Predicted offsets from keypoints [b, pts_per_kpt*kpt_num, 3]
        """
        b = kpt_3d.shape[0]
        kpt_num = kpt_3d.shape[2]
        
        # Encode 3D positions
        pos_enc_3d = self.pos_enc(kpt_3d)  # [b, ndim, kpt_num]
        
        # Process keypoint features
        kpt_feature = self.mlp(kpt_feature)  # [b, ndim, kpt_num]
        
        # Global feature: average of position encoding + keypoint features
        global_feature = torch.mean(pos_enc_3d + kpt_feature, dim=2, keepdim=True)  # [b, ndim, 1]
        
        # Concatenate global and local features
        recon_feature = torch.cat([
            global_feature.repeat(1, 1, kpt_num), 
            kpt_feature
        ], dim=1)  # [b, 2*ndim, kpt_num]
        
        # Decode to 3D offsets: [b, 3*pts_per_kpt, kpt_num]
        recon_delta = self.shape_decoder(recon_feature)
        
        # Reshape offsets: [b, pts_per_kpt*kpt_num, 3]
        recon_delta = recon_delta.transpose(1, 2).reshape(b, kpt_num * self.pts_per_kpt, 3).contiguous()
        
        # Interleave keypoint coordinates to match offset dimensions
        # [b, kpt_num, 3] -> [b, pts_per_kpt*kpt_num, 3]
        kpt_3d_interleave = kpt_3d.transpose(1, 2).repeat_interleave(self.pts_per_kpt, dim=1).contiguous()
        
        # Add offsets to keypoint coordinates
        recon_model = (recon_delta + kpt_3d_interleave).transpose(1, 2).contiguous()  # [b, 3, pts_per_kpt*kpt_num]
        
        return recon_model, recon_delta

