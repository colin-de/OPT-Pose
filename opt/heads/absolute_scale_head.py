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
from geo_models.modules import PointNet2MSG


def generate_augmentation(batchsize, device):
    """
    Generate random augmentation parameters for training robustness.
    
    Args:
        batchsize: batch size
        device: torch device
    
    Returns:
        delta_r: [B, 3, 3] rotation matrices
        delta_t: [B, 1, 3] translation offsets
        delta_s: [B, 1] scale factors
    """
    # Random translation [-0.02m, 0.02m]
    delta_t = torch.rand(batchsize, 1, 3, device=device)
    delta_t = delta_t * 0.04 - 0.02
    
    # Random rotation angles [-20°, 20°]
    angle_r = torch.rand(batchsize, 3, device=device)
    angle_r = angle_r * 40 - 20
    angle_r = angle_r / 180 * torch.pi  # convert to radians
    
    # Construct rotation matrices
    delta_r_x = torch.eye(3, device=device).unsqueeze(0).repeat(batchsize, 1, 1)
    delta_r_y = torch.eye(3, device=device).unsqueeze(0).repeat(batchsize, 1, 1)
    delta_r_z = torch.eye(3, device=device).unsqueeze(0).repeat(batchsize, 1, 1)
    
    # Rotation around X-axis
    delta_r_x[:, 1, 1] = torch.cos(angle_r[:, 0])
    delta_r_x[:, 1, 2] = -torch.sin(angle_r[:, 0])
    delta_r_x[:, 2, 1] = torch.sin(angle_r[:, 0])
    delta_r_x[:, 2, 2] = torch.cos(angle_r[:, 0])
    
    # Rotation around Y-axis
    delta_r_y[:, 0, 0] = torch.cos(angle_r[:, 1])
    delta_r_y[:, 0, 2] = torch.sin(angle_r[:, 1])
    delta_r_y[:, 2, 0] = -torch.sin(angle_r[:, 1])
    delta_r_y[:, 2, 2] = torch.cos(angle_r[:, 1])
    
    # Rotation around Z-axis
    delta_r_z[:, 0, 0] = torch.cos(angle_r[:, 2])
    delta_r_z[:, 0, 1] = -torch.sin(angle_r[:, 2])
    delta_r_z[:, 1, 0] = torch.sin(angle_r[:, 2])
    delta_r_z[:, 1, 1] = torch.cos(angle_r[:, 2])
    
    # Combined rotation: R = Rz @ Ry @ Rx
    delta_r = torch.bmm(torch.bmm(delta_r_z, delta_r_y), delta_r_x)
    
    # Random scale [0.8, 1.2]
    delta_s = torch.rand(batchsize, 1, device=device)
    delta_s = delta_s * 0.4 + 0.8
    
    return delta_r, delta_t, delta_s


class AbsoluteScaleHead(nn.Module):
    """
    Predict absolute translation and size from sensor depth point cloud.
    This allows deriving avg_scale by comparing with normalized predictions.
    
    Improvements:
    1. Apply data augmentation during training for robustness
    2. Work with all K sampled points (choose_indices) for better coverage
    3. Use z_obj conditioning for category-aware predictions
    """
    def __init__(self, z_obj_dim=256, rgb_feature_dim=128, radii_list=None, use_augmentation=True):
        super().__init__()
        self.z_obj_dim = z_obj_dim
        self.rgb_feature_dim = rgb_feature_dim
        self.use_augmentation = use_augmentation
        
        if radii_list is None:
            radii_list = [[0.01, 0.02], [0.02, 0.04], [0.04, 0.08], [0.08, 0.16]]
        
        # PointNet++ for feature extraction from sensor point cloud
        # Input: xyz (3) + rgb_feature (128) = 131 dimensions (use_xyz=True adds 3 more)
        self.pn2msg = PointNet2MSG(radii_list=radii_list, dim_in=3+rgb_feature_dim, use_xyz=True)
        
        # Project z_obj to match point feature dimension for fusion
        self.z_proj = nn.Sequential(
            nn.Linear(z_obj_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        
        # Translation head: predicts absolute translation (object center in camera frame)
        # Input: PointNet++ features (256) + projected z_obj (256) = 512
        self.t_mlp = nn.Sequential(
            nn.Conv1d(256 + z_obj_dim, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(128, 3, 1),
        )
        # Initialize translation bias to zero (residual connection will provide base)
        self.t_mlp[-1].bias.data.zero_()
        
        # Size head: predicts absolute size conditioned on z_obj
        # Input: PointNet++ features (256) + projected z_obj (256) = 512
        self.s_mlp = nn.Sequential(
            nn.Conv1d(256 + z_obj_dim, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(128, 3, 1),  # Directly predict 3D size
        )
    
    def forward(self, pts, rgb_features, z_obj):
        """
        Args:
            pts: [BS, K, 3] point cloud from sensor depth (centered at sensor mean)
            rgb_features: [BS, K, rgb_feature_dim] visual features at sampled points
            z_obj: [BS, z_obj_dim] object latent embedding
        
        Returns:
            translation: [BS, 3] translation offset from sensor center
            size: [BS, 3] absolute size (object dimensions)
        """
        bs, k, _ = pts.shape
        device = pts.device
        
        # Store augmentation parameters for output transformation
        delta_r, delta_t, delta_s = None, None, None
        
        # Apply data augmentation during training
        if self.training and self.use_augmentation:
            delta_r, delta_t, delta_s = generate_augmentation(bs, device)
            # Apply inverse transformation to centered points: (pts - t) / s @ R
            # Note: pts are already centered, so delta_t shifts the relative geometry
            pts_aug = (pts - delta_t) / delta_s.unsqueeze(2) @ delta_r
        else:
            pts_aug = pts
        
        # Concatenate features: [pts, pts, rgb_features] -> [BS, K, 3+3+128]
        # PointNet2MSG will use first 3 as xyz, remaining as features
        x = torch.cat([pts_aug, pts_aug, rgb_features], dim=2)
        
        # Extract features: [BS, 256, K]
        x = self.pn2msg(x)
        
        # Project z_obj and expand to match point dimension
        z_proj = self.z_proj(z_obj)  # [BS, 256]
        z_expanded = z_proj.unsqueeze(-1).expand(-1, -1, k)  # [BS, 256, K]
        
        # Fuse PointNet++ features with z_obj
        x_fused = torch.cat([x, z_expanded], dim=1)  # [BS, 512, K]
        
        # Predict translation offset with residual connection
        # t_mlp predicts offset from mean point position
        t_offset = self.t_mlp(x_fused)  # [BS, 3, K]
        t = t_offset + pts_aug.transpose(1, 2)  # Add residual to augmented points
        t = torch.mean(t, dim=2)  # [BS, 3] - average over points
        
        # Predict size conditioned on z_obj
        s = self.s_mlp(x_fused)  # [BS, 3, K]
        s = torch.mean(s, dim=2)  # [BS, 3] - average over points
        
        # pred_output = delta_r^T @ (pred * delta_s) + delta_t
        if self.training and self.use_augmentation:
            # t_original = delta_r^T @ (t_aug * delta_s) + delta_t
            t = (t.unsqueeze(1) * delta_s.unsqueeze(2)) @ delta_r.transpose(1, 2)
            t = t.squeeze(1) + delta_t.squeeze(1)
            s = s * delta_s
        
        return t, s
    
    def compute_scale_from_absolute(self, abs_translation, abs_size, 
                                   norm_translation, norm_size, 
                                   reduction='geometric_mean',
                                   valid_mask=None):
        """
        Derive avg_scale by comparing absolute and normalized predictions.
        
        Scale relationship:
            abs_translation = norm_translation * scale
            abs_size = norm_size * scale
        
        Args:
            abs_translation: [B, S, 3] absolute translation
            abs_size: [B, S, 3] absolute size  
            norm_translation: [B, S, 3] normalized translation
            norm_size: [B, S, 3] normalized size
            reduction: How to combine scale estimates ('geometric_mean', 'median', 'mean')
            valid_mask: [B, S] optional mask indicating which frames have valid sensor depth (inference only)
        
        Returns:
            scale: [B, S] derived scale factor
        """
        eps = 1e-6
        
        # Compute scale from translation: scale = ||abs_t|| / ||norm_t||
        scale_t = torch.norm(abs_translation, dim=-1) / (torch.norm(norm_translation, dim=-1) + eps)
        
        # Compute scale from size: scale = ||abs_s|| / ||norm_s||
        scale_s = torch.norm(abs_size, dim=-1) / (torch.norm(norm_size, dim=-1) + eps)
        
        # Combine scale estimates
        if reduction == 'geometric_mean':
            # Geometric mean is more robust to outliers in log space
            log_scale_t = torch.log(scale_t.clamp_min(eps))
            log_scale_s = torch.log(scale_s.clamp_min(eps))
            log_scale = (log_scale_t + log_scale_s) / 2.0
            scale_per_frame = torch.exp(log_scale)
        elif reduction == 'median':
            # Stack and take median
            scales = torch.stack([scale_t, scale_s], dim=-1)  # [B, S, 2]
            scale_per_frame = torch.median(scales, dim=-1)[0]
        else:  # 'mean'
            scale_per_frame = (scale_t + scale_s) / 2.0
        
        # Inference mode with valid_mask: use average of valid frames for entire sequence
        if valid_mask is not None:
            B, S = scale_per_frame.shape
            derived_scale_list = []
            for b in range(B):
                valid_frames = valid_mask[b]  # [S]
                if valid_frames.any():
                    valid_scales = scale_per_frame[b, valid_frames]
                    valid_scales = valid_scales[torch.isfinite(valid_scales)]
                    if len(valid_scales) > 0:
                        avg_scale = valid_scales.mean()
                    else:
                        avg_scale = torch.tensor(1.0, device=scale_per_frame.device)
                    derived_scale_list.append(avg_scale.expand(S))
                else:
                    derived_scale_list.append(torch.ones(S, device=scale_per_frame.device))
            
            scale = torch.stack(derived_scale_list, dim=0)  # [B, S]
        else:
            # Training mode: all frames valid, use per-frame scale
            scale = scale_per_frame
        
        return scale

