# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Project-owned portions of this file are licensed under CC-BY-NC-SA-4.0.
# See LICENSE and NOTICE for details. Third-party notices remain applicable.

from typing import Optional, Dict, Tuple, List
import torch
import torch.nn as nn
import pdb
from .dpt_head import DPTHead


__all__ = ["PoseNetHead"]


def _flatten_hw(x: torch.Tensor) -> torch.Tensor:
    """[B, S, H, W, C] -> [B, S, N, C] with N=H*W."""
    B, S, H, W, C = x.shape
    return x.reshape(B, S, H * W, C)


def _normalize_weights_seq(w: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize weights per sequence to sum to 1 (over frames & pixels).
    Falls back to uniform when the sum is ~0 to keep gradients well-defined.
    Input/Return: [B, K] (K = S*N or top-k size)
    """
    s = w.sum(dim=-1, keepdim=True)
    w_norm = w / (s + eps)
    uniform = torch.full_like(w_norm, 1.0 / w.shape[-1])
    use_uniform = (s <= eps).float()
    return w_norm * (1 - use_uniform) + uniform * use_uniform


def rotation_6d_to_matrix(r6: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to 3x3 rotation matrix using
    Gram-Schmidt orthogonalization (Zhou et al. CVPR'19).

    Args:
        r6: [B, 6]
    Returns:
        R: [B, 3, 3]
    """
    a1 = r6[:, 0:3]
    a2 = r6[:, 3:6]
    b1 = nn.functional.normalize(a1, dim=-1)
    proj = (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = nn.functional.normalize(a2 - proj, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    R = torch.stack([b1, b2, b3], dim=-1)
    return R

def normalize_vector(v, dim=1, return_mag =False):
    """Device/dtype-safe vector normalization with epsilon clamp.

    Args:
        v: Tensor to normalize.
        dim: Dimension along which to compute the norm.
        return_mag: If True, also return the magnitudes.
    Returns:
        v_norm (and optionally magnitudes) with safe epsilon.
    """
    v_mag = v.norm(dim=dim, keepdim=True).clamp_min(1e-8)
    v_norm = v / v_mag
    if return_mag:
        return v_norm, v_mag
    return v_norm

def cross_product(u, v):
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
    out = torch.cat((i.unsqueeze(1), j.unsqueeze(1), k.unsqueeze(1)),1)#batch*3
    return out
    
def Ortho6d2Mat(x_raw, y_raw):
    y = normalize_vector(y_raw)
    z = cross_product(x_raw, y)
    z = normalize_vector(z)#batch*3
    x = cross_product(y,z)#batch*3

    x = x.unsqueeze(2)
    y = y.unsqueeze(2)
    z = z.unsqueeze(2)
    matrix = torch.cat((x,y,z), 2) #batch*3*3
    return matrix


class PoseNetHead(nn.Module):
    """
    Sim(3) pose head (MLP-based regressor).

    Supports two modes controlled by ``multi_view_aggregation``:

    * **Per-frame** (``multi_view_aggregation=False``): estimates one pose per
      frame independently from K correspondences.
    * **Multi-view** (``multi_view_aggregation=True``): aggregates S*K
      correspondences (all in frame-0 space via pointmap) to estimate a single
      canonical -> frame-0 pose.  The output is broadcast back to [B, S, ...]
      for downstream compatibility.  When S=1 this is identical to per-frame.

    Inputs:
        kpt_nocs:    [B, S, K, 3]  (keypoints in NOCS canonical coords [-0.5,0.5])
        kpt_3d:      [B, S, K, 3]  (keypoints in frame-0 / camera coordinates)
        kpt_feat:    [B, S, K, C]  (keypoint features)

    Returns:
        pose: dict with keys
            - "scale":        [B, S, 1]
            - "size":         [B, S, 3]
            - "rotation":     [B, S, 3, 3]
            - "translation":  [B, S, 3]
            - "transform_4x4":[B, S, 4, 4]
    """

    def __init__(
        self,
        sqrt_weight: bool = True,
        rotation_only: bool = False,
        eps: float = 1e-8,
        hidden_dim_pts: int = 128,
        hidden_dim_pose: int = 512,
        features: int = 256,
        d_obj: int = 256,
        multi_view_aggregation: bool = False,
    ):
        super().__init__()
        self.sqrt_weight = sqrt_weight
        self.rotation_only = rotation_only
        self.eps = eps
        self.features = features
        self.d_obj = d_obj
        self.multi_view_aggregation = multi_view_aggregation

        # Inline pose-size estimator: point encoders + fusion heads
        # pts mlps for two point sets (world and nocs)
        self.pts_mlp1 = nn.Sequential(
            nn.Conv1d(3, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 1),
            nn.ReLU(inplace=True),
        )
        self.pts_mlp2 = nn.Sequential(
            nn.Conv1d(3, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 1),
            nn.ReLU(inplace=True),
        )
        self.pose_mlp1 = nn.Sequential(
            nn.Conv1d(64 + self.features + 64, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(inplace=True),
        )
        self.pose_mlp2 = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

        # ---- FiLM conditioning on z_obj (category/object-aware modulation) ----
        # Modulates per-keypoint features (256-dim) after pose_mlp1
        self.film_kpt = nn.Sequential(
            nn.Linear(d_obj, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2 * 256),  # gamma and beta
        )
        # Modulates global pose vector (512-dim) after pose_mlp2
        self.film_global = nn.Sequential(
            nn.Linear(d_obj, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2 * 512),  # gamma and beta
        )
        # Initialize FiLM to identity transform (gamma=1, beta=0) so
        # untrained z_obj doesn't disturb existing pose features
        with torch.no_grad():
            # film_kpt: last linear bias -> [1,1,...,1, 0,0,...,0]
            nn.init.zeros_(self.film_kpt[-1].weight)
            self.film_kpt[-1].bias[:256].fill_(1.0)   # gamma = 1
            self.film_kpt[-1].bias[256:].fill_(0.0)   # beta = 0
            # film_global: last linear bias -> [1,1,...,1, 0,0,...,0]
            nn.init.zeros_(self.film_global[-1].weight)
            self.film_global[-1].bias[:512].fill_(1.0)  # gamma = 1
            self.film_global[-1].bias[512:].fill_(0.0)  # beta = 0

        self.rotation_estimator = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 6),
        )
        self.translation_estimator = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3),
        )
        self.size_estimator = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3),
        )

        # Initialize rotation estimator to predict near-identity at start
        # This helps avoid collapsing to a random fixed rotation early in training.
        last = self.rotation_estimator[-1]
        if isinstance(last, nn.Linear):
            with torch.no_grad():
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
                # Bias to 6D approx: [1,0,0, 0,1,0] -> identity after Ortho6d2Mat/Gram-Schmidt
                init_bias = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=last.bias.dtype)
                last.bias.copy_(init_bias)

    @torch.amp.autocast("cuda", enabled=False)
    def forward(
        self,
        kpt_nocs: torch.Tensor,    # [BS,K,3] or [B,S,K,3]
        kpt_3d: torch.Tensor,      # [BS,K,3] or [B,S,K,3]
        kpt_feat: torch.Tensor,    # [BS,K,C] or [B,S,K,C]
        batch_size: Optional[int] = None,  # B (required if inputs are 3D)
        seq_len: Optional[int] = None,     # S (required if inputs are 3D)
        center: Optional[torch.Tensor] = None,  # [B,S,1,3] - center offset to add to translation
        z_obj: Optional[torch.Tensor] = None,   # [B,S,d_obj] or [BS,d_obj] - object/category embedding
    ) -> Dict[str, torch.Tensor]:
        # ---- Resolve B, S, K from either 3D or 4D inputs ----
        if kpt_3d.dim() == 4:
            B, S, K, _ = kpt_3d.shape
            C = self.features
            kpt_nocs = kpt_nocs.view(B * S, K, 3)
            kpt_3d = kpt_3d.view(B * S, K, 3)
            kpt_feat = kpt_feat.view(B * S, K, C)
            if z_obj is not None and z_obj.dim() == 3:
                z_obj = z_obj.view(B * S, -1)
        else:
            assert batch_size is not None and seq_len is not None, \
                "batch_size and seq_len must be provided when using 3D input format"
            B, S = batch_size, seq_len
            BS, K, _ = kpt_3d.shape
            C = self.features
            assert BS == B * S, f"Expected BS={B*S}, got {BS}"

        # ---- Multi-view aggregation: merge S frames into one ----
        # All kpt_3d are already in frame-0 coordinates (from pointmap),
        # so S*K correspondences constrain the same canonical -> frame-0 transform.
        use_mv = self.multi_view_aggregation and S > 1
        if use_mv:
            kpt_nocs = kpt_nocs.view(B, S * K, 3)   # [B, S*K, 3]
            kpt_3d = kpt_3d.view(B, S * K, 3)       # [B, S*K, 3]
            kpt_feat = kpt_feat.view(B, S * K, C)    # [B, S*K, C]
            if z_obj is not None:
                z_obj = z_obj.view(B, S, -1)[:, 0]   # [B, d_obj] (shared across frames)
            N_eff = B  # effective batch dim
        else:
            N_eff = B * S  # per-frame processing

        # ---- Estimate pose and size ----
        pts1 = self.pts_mlp1(kpt_3d.transpose(1, 2))                              # [N_eff,64,K']
        pts2 = self.pts_mlp2(kpt_nocs.transpose(1, 2))                            # [N_eff,64,K']
        pose_feat = torch.cat([pts1, kpt_feat.transpose(1, 2), pts2], dim=1)      # [N_eff,64+C+64,K']

        pose_feat = self.pose_mlp1(pose_feat)                                      # [N_eff,256,K']

        # ---- FiLM conditioning on z_obj (per-keypoint level) ----
        if z_obj is not None:
            film_params_kpt = self.film_kpt(z_obj)                                 # [N_eff, 2*256]
            gamma_kpt = film_params_kpt[:, :256].unsqueeze(2)                      # [N_eff, 256, 1]
            beta_kpt = film_params_kpt[:, 256:].unsqueeze(2)                       # [N_eff, 256, 1]
            pose_feat = gamma_kpt * pose_feat + beta_kpt                           # [N_eff, 256, K']

        pose_global = torch.mean(pose_feat, dim=2, keepdim=True)                   # [N_eff,256,1]
        pose_feat = torch.cat([pose_feat, pose_global.expand_as(pose_feat)], dim=1)  # [N_eff,512,K']
        pose_vec = self.pose_mlp2(pose_feat).squeeze(2)                            # [N_eff,512]

        # ---- FiLM conditioning on z_obj (global level) ----
        if z_obj is not None:
            film_params_global = self.film_global(z_obj)                           # [N_eff, 2*512]
            gamma_global = film_params_global[:, :512]
            beta_global = film_params_global[:, 512:]
            pose_vec = gamma_global * pose_vec + beta_global                       # [N_eff, 512]

        r6 = self.rotation_estimator(pose_vec)                                     # [N_eff,6]
        R = Ortho6d2Mat(r6[:, :3].contiguous(), r6[:, 3:].contiguous()).view(-1, 3, 3)
        t_raw = self.translation_estimator(pose_vec)                               # [N_eff,3]
        s_vec = self.size_estimator(pose_vec)                                      # [N_eff,3]

        # ---- Reshape to [B, S, ...] ----
        if use_mv:
            # Single pose per batch element -> broadcast to all S frames
            size = s_vec.view(B, 1, 3).expand(B, S, 3)
            scale = size.abs().mean(dim=-1, keepdim=True).clamp_min(self.eps)
            rotation = R.view(B, 1, 3, 3).expand(B, S, 3, 3)
            translation = t_raw.view(B, 1, 3).expand(B, S, 3)
        else:
            size = s_vec.reshape(B, S, 3)
            scale = size.abs().mean(dim=-1, keepdim=True).clamp_min(self.eps)
            rotation = R.reshape(B, S, 3, 3)
            translation = t_raw.reshape(B, S, 3)

        if center is not None:
            if use_mv:
                # Use frame-0 center only (all kpt_3d are in frame-0 space)
                center_flat = center[:, 0:1].squeeze(2).expand(B, S, 3)  # [B,S,3]
            else:
                center_flat = center.squeeze(2)  # [B,S,1,3] -> [B,S,3]
            translation = translation + center_flat

        if self.rotation_only:
            scale = torch.ones_like(scale)
            translation = torch.zeros_like(translation)
            size = torch.ones_like(size)

        # Build 4x4 homogeneous transform per frame
        transform = torch.zeros(B, S, 4, 4, device=kpt_3d.device, dtype=kpt_3d.dtype)
        transform[:, :, :3, :3] = scale[..., None] * rotation
        transform[:, :, :3, 3] = translation
        transform[:, :, 3, 3] = 1.0

        pose: Dict[str, torch.Tensor] = {
            "scale": scale,
            "size": size,
            "rotation": rotation,
            "translation": translation,
            "transform_4x4": transform,
        }
        return pose
