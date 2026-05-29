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
from huggingface_hub import PyTorchModelHubMixin  # used for model hub
import sonata
# import flash_attn
import warnings
from typing import Optional, Dict, List

from opt.models.aggregator import Aggregator
from opt.heads.camera_head import CameraHead
from opt.heads.dpt_head import DPTHead
from opt.heads.nocs_head import NOCSHead
from opt.heads.pose_head import PoseHead
from opt.heads.posenet_head import PoseNetHead
from opt.heads.absolute_scale_head import AbsoluteScaleHead, generate_augmentation
from opt.heads.rel_scale_head import RelScaleHead
from opt.heads.reconstructor_head import ReconstructorHead
from geo_models.modules import PointNet2MSG
import torch.nn.functional as F

__all__ = ["OPT", "VGGT"]


def batched_transform_default(
    coords_bsk3: torch.Tensor,           # [B,S,K,3], float32
    colors_bsk3: Optional[torch.Tensor], # [B,S,K,3], float32 in [0,1] or [0,255]
    normals_bsk3: Optional[torch.Tensor],# [B,S,K,3], float32 (unit)
    grid_size: float = 0.004,
    center_shift: bool = True,
    apply_z: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Vectorized equivalent of Sonata's default() + collate_fn, for the whole [B,S] batch.
    Returns a single 'point' dict compatible with the Sonata model forward().
    Keys: coord [Nv,3] float, grid_coord [Nv,3] int32, inverse [Ntot] int64, offset [BS] int32, feat [Nv,F] float, (color [Nv,3] float)
    where Nv is total voxel count across frames and Ntot = B*S*K.
    """
    assert coords_bsk3.dim() == 4 and coords_bsk3.size(-1) == 3
    B, S, K, _ = coords_bsk3.shape
    BS = B * S
    device = coords_bsk3.device
    dtype = coords_bsk3.dtype

    # ---- CenterShift (per-frame) ----
    # Compute per-frame mean [BS,1,1,3] then subtract
    if center_shift:
        coords_bk3 = coords_bsk3.reshape(BS, K, 3)
        means = coords_bk3.mean(dim=1, keepdim=True)  # [BS,1,3]
        if not apply_z:
            means = torch.stack([means[..., 0], means[..., 1], torch.zeros_like(means[..., 2])], dim=-1)
        coords_cs = coords_bk3 - means                     # [BS,K,3]
    else:
        coords_cs = coords_bsk3.reshape(BS, K, 3)

    Ntot = BS * K
    coords_cs_flat = coords_cs.reshape(Ntot, 3)            # [Ntot,3]

    colors_flat  = None
    normals_flat = None
    if colors_bsk3 is not None:
        colors_flat = colors_bsk3.reshape(Ntot, 3).to(device=device, dtype=dtype)
        colors_flat = colors_flat - 0.5  # NormalizeColor equivalent for [0,1] inputs
    if normals_bsk3 is not None:
        normals_flat = normals_bsk3.reshape(Ntot, 3).to(device=device, dtype=dtype)

    # ---- GridSample: quantize + per-frame unique ----
    # Per-point integer voxel coords
    grid_coord_flat = torch.floor(coords_cs_flat / grid_size).to(torch.int32)  # [Ntot,3]

    # Keep frames independent by including frame_id in uniqueness
    frame_ids = torch.arange(BS, device=device, dtype=torch.int64).repeat_interleave(K)  # [Ntot]
    # Stack [frame_id, gx, gy, gz] so unique() is per frame
    grid_with_f = torch.stack(
        [frame_ids,
         grid_coord_flat[:, 0].to(torch.int64),
         grid_coord_flat[:, 1].to(torch.int64),
         grid_coord_flat[:, 2].to(torch.int64)], dim=1
    )  # [Ntot,4] int64

    # Unique voxels across all frames at once
    unique_rows, inverse_global, counts = torch.unique(
        grid_with_f, dim=0, return_inverse=True, return_counts=True
    )
    Nv = unique_rows.shape[0]  # total voxels across BS frames

    # Split back out
    voxel_frame = unique_rows[:, 0].to(torch.int64)        # [Nv]
    grid_coord  = unique_rows[:, 1:].to(torch.int32)       # [Nv,3]

    # ---- Reduce to per-voxel means (coord/color/normal) ----
    # sums via index_add_
    ones = torch.ones(Ntot, 1, device=device, dtype=dtype)
    denom = torch.zeros(Nv, 1, device=device, dtype=dtype)
    denom.index_add_(0, inverse_global, ones)              # [Nv,1]

    coord_sum = torch.zeros(Nv, 3, device=device, dtype=dtype)
    coord_sum.index_add_(0, inverse_global, coords_cs_flat)
    coord_vox = coord_sum / denom.clamp_min(1e-8)          # [Nv,3]

    color_vox = None
    if colors_flat is not None:
        color_sum = torch.zeros(Nv, 3, device=device, dtype=dtype)
        color_sum.index_add_(0, inverse_global, colors_flat)
        color_vox = color_sum / denom.clamp_min(1e-8)      # [Nv,3]

    normal_vox = None
    if normals_flat is not None:
        normal_sum = torch.zeros(Nv, 3, device=device, dtype=dtype)
        normal_sum.index_add_(0, inverse_global, normals_flat)
        normal_vox = normal_sum / denom.clamp_min(1e-8)    # [Nv,3]

    # ---- Build feat = [coord,(color),(normal)] ----
    feat_parts = [coord_vox]
    if color_vox is not None:  feat_parts.append(color_vox)
    if normal_vox is not None: feat_parts.append(normal_vox)
    feat = torch.cat(feat_parts, dim=1).contiguous()       # [Nv, F]

    # ---- Build offset (voxel counts per frame) like collate_fn ----
    # Count voxels per frame id
    voxels_per_frame = torch.bincount(voxel_frame, minlength=BS).to(torch.int64)  # [BS]
    offset = torch.cumsum(voxels_per_frame, dim=0).to(torch.int32)                # [BS]

    # inverse: point→voxel (global indexing over Nv)
    point = {
        "coord": coord_vox,          # [Nv,3] float32
        "grid_coord": grid_coord,    # [Nv,3] int32
        "inverse": inverse_global,   # [Ntot] int64
        "offset": offset,            # [BS] int32  (frame voxel offsets)
        "feat": feat,                # [Nv,F] float32
    }
    if color_vox is not None:
        point["color"] = color_vox   # [Nv,3] float32 (normalized)
    return point


class SonataBackbone(nn.Module):
    def __init__(self, device="cuda", enable_flash=True, trainable=True):
        super().__init__()
        self.device = torch.device(device)
        self.trainable = trainable
        self.model = sonata.load("sonata", repo_id="facebook/sonata",
                                 custom_config=dict(enable_flash=enable_flash)).to(self.device)
        if not trainable:
            self.model.eval()
        self.transform = sonata.transform.default()

    def _upcast_point_features(self, point):
        for _ in range(2):
            if "pooling_parent" in point and "pooling_inverse" in point:
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                try:
                    parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                except Exception:
                    parent.feat = point.feat[inverse]
                point = parent
        while "pooling_parent" in point:
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = point.feat[inverse]
            point = parent
        # Map voxels → original points (Ntot)
        if "inverse" in point:
            inv = point.pop("inverse")
            point.feat = point.feat[inv]
        return point

    def extract(self, coords_bsk3, colors_bsk3=None, normals_bsk3=None):
        B, S, K, _ = coords_bsk3.shape
        BS = B * S

        # Keep tensors on CPU for transform (it expects numpy/CPU tensors inside).
        coords_cpu  = coords_bsk3.reshape(BS, K, 3).detach().contiguous().cpu()
        colors_cpu  = colors_bsk3.reshape(BS, K, 3).detach().contiguous().cpu() if colors_bsk3 is not None else None
        normals_cpu = normals_bsk3.reshape(BS, K, 3).detach().contiguous().cpu() if normals_bsk3 is not None else None

        frames = []
        for i in range(BS):
            d = {"coord": coords_cpu[i].numpy()}
            if colors_cpu is not None:  d["color"]  = colors_cpu[i].numpy()
            if normals_cpu is not None: d["normal"] = normals_cpu[i].numpy()
            frames.append(self.transform(d))

        point = sonata.data.collate_fn(frames)

        # # Build a single collated point dict (no per-frame loop)
        # point = batched_transform_default(
        #     coords_bsk3=coords_bsk3, colors_bsk3=colors_bsk3, normals_bsk3=normals_bsk3,
        #     grid_size=0.004, center_shift=True, apply_z=True
        # )

        for k, v in list(point.items()):
            if isinstance(v, torch.Tensor):
                point[k] = v.to(self.device, non_blocking=True)

        # Forward Sonata (with or without gradients based on trainable flag)
        if self.trainable:
            point = self.model(point)
        else:
            with torch.no_grad():
                point = self.model(point)

        # Upcast back to original per-point resolution (Ntot = BS*K)
        point = self._upcast_point_features(point)  # point.feat: [Ntot, 1088]
        feat = point.feat.view(B, S, K, -1).contiguous()
        return feat  # [B,S,K,1088]


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


class KeyPointDetector(nn.Module):
    """
    KeyPoint Detector
    """
    def __init__(self, kpt_num=128, query_dim=256, input_dim=384):
        super().__init__()
        self.kpt_num = kpt_num
        self.query_dim = query_dim
        
        self.kpt_query = nn.Parameter(torch.empty(self.kpt_num, self.query_dim))
        nn.init.xavier_normal_(self.kpt_query)
        
        self.multihead_attn = nn.MultiheadAttention(query_dim, num_heads=4, batch_first=True)
        self.self_attn = nn.MultiheadAttention(query_dim, num_heads=4, batch_first=True)
        
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.norm3 = nn.LayerNorm(query_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, query_dim * 2),
            nn.ReLU(),
            nn.Linear(query_dim * 2, query_dim)
        )
        
    def forward(self, rgb_local, pts_local):
        """
        Args:
            rgb_local: (b, c, n) RGB features
            pts_local: (b, c, n) Point features
        Returns:
            batch_kpt_query: (b, kpt_num, query_dim) keypoint features
            heatmap: (b, kpt_num, n) attention heatmap
        """
        b, c, n = rgb_local.shape
        
        # Concatenate RGB and point features
        input_feature = torch.cat((pts_local, rgb_local), dim=1)  # (b, 2c, n)
        input_feature = input_feature.transpose(1, 2)  # (b, n, 2c)
        
        # Expand keypoint queries
        batch_kpt_query = self.kpt_query.unsqueeze(0).repeat(b, 1, 1)  # (b, kpt_num, query_dim)
        
        # Cross attention
        kpt_query2 = self.norm1(batch_kpt_query)
        kpt_query2, attn = self.multihead_attn(
            query=kpt_query2,
            key=input_feature,
            value=input_feature
        )
        batch_kpt_query = batch_kpt_query + kpt_query2
        
        # Self attention
        kpt_query2 = self.norm2(batch_kpt_query)
        kpt_query2, _ = self.self_attn(kpt_query2, kpt_query2, kpt_query2)
        batch_kpt_query = batch_kpt_query + kpt_query2
        
        # FFN (Feed-Forward Network)
        kpt_query2 = self.norm3(batch_kpt_query)
        kpt_query2 = self.ffn(kpt_query2)
        batch_kpt_query = batch_kpt_query + kpt_query2
        
        # Heatmap
        norm1 = torch.norm(batch_kpt_query, p=2, dim=2, keepdim=True)
        norm2 = torch.norm(input_feature, p=2, dim=2, keepdim=True).transpose(1, 2)
        heatmap = torch.bmm(batch_kpt_query, input_feature.transpose(1, 2)) / (norm1 * norm2 + 1e-7)
        heatmap = F.softmax(heatmap / 0.1, dim=2)  # (b, kpt_num, n)
        
        return batch_kpt_query, heatmap


class VisualGeoFeatureFusion(nn.Module):
    """
    Visual Geometric Feature Fusion
    """
    def __init__(self, K_list=[8, 16, 32], d_model=256):
        super().__init__()
        self.block_num = len(K_list)
        self.K_list = K_list
        self.d_model = d_model
        
        self.fusion_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.fusion_blocks.append(VisualGeoFusionBlock(self.K_list[i], self.d_model))
    
    def forward(self, kpt_feature, kpt_3d, pts_feature, pts):
        """
        Args:
            kpt_feature: (b, kpt_num, dim) keypoint features
            kpt_3d: (b, kpt_num, 3) keypoint 3D coordinates
            pts_feature: (b, n, dim) point features
            pts: (b, n, 3) point coordinates
        Returns:
            kpt_feature: (b, kpt_num, dim) enhanced keypoint features
        """
        for i in range(self.block_num):
            kpt_feature = self.fusion_blocks[i](kpt_feature, kpt_3d, pts_feature, pts)
        return kpt_feature


class ObjectEmbeddingHead(nn.Module):
    """Build object embedding from keypoint features."""

    def __init__(self, dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.obj_init_token = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.xavier_normal_(self.obj_init_token)
        self.obj_cross_attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.obj_query_norm = nn.LayerNorm(dim)
        self.obj_kv_norm = nn.LayerNorm(dim)
        self.obj_attn_proj = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.z_obj_mix_logit = nn.Parameter(torch.tensor(0.0))
        self.z_mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, kpt_features: torch.Tensor) -> torch.Tensor:
        bs = kpt_features.shape[0]
        z_mean = kpt_features.mean(dim=1)  # [BS, dim]
        obj_query = self.obj_init_token.expand(bs, -1, -1)  # [BS, 1, dim]
        q = self.obj_query_norm(obj_query)
        kv = self.obj_kv_norm(kpt_features)
        z_attn, _ = self.obj_cross_attn(q, kv, kv, need_weights=False)  # [BS, 1, dim]
        z_attn = self.obj_attn_proj(z_attn.squeeze(1))  # [BS, dim]
        mix = torch.sigmoid(self.z_obj_mix_logit)
        z_seed = mix * z_attn + (1.0 - mix) * z_mean
        return F.normalize(self.z_mlp(z_seed), dim=-1)  # [BS, dim]


class VisualGeoFusionBlock(nn.Module):
    """Single fusion block for VisualGeoFeatureFusion"""
    def __init__(self, k, d_model):
        super().__init__()
        self.k = k
        self.d_model = d_model
        
        self.fc_in = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, 1),
        )
        
        self.fc_delta = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        
        self.fc_delta_1 = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        self.fc_delta_l = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        
        self.fc_delta_abs = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        
        self.fuse_mlp = nn.Sequential(
            nn.Conv1d(3 * d_model, d_model, 1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, 1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
        )
        
        self.out_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        self.relu = nn.ReLU()
        self.tau = 5.0
    
    def forward(self, kpt_feature, kpt_3d, pts_feature, pts):
        """
        Args:
            kpt_feature: (b, kpt_num, d_model)
            kpt_3d: (b, kpt_num, 3)
            pts_feature: (b, n, d_model)
            pts: (b, n, 3)
        Returns:
            kpt_feature: (b, kpt_num, d_model)
        """
        # Find K nearest neighbors for each keypoint
        dis_mat = torch.norm(kpt_3d.unsqueeze(2) - pts.unsqueeze(1), dim=3)  # (b, kpt_num, n)
        knn_idx = dis_mat.argsort()[:, :, :self.k]  # (b, kpt_num, k)
        knn_xyz = index_points(pts, knn_idx)  # (b, kpt_num, k, 3)
        knn_feature = index_points(pts_feature, knn_idx)  # (b, kpt_num, k, d_model)
        
        # Process features
        pre = kpt_feature
        kpt_feature = self.fc_in(kpt_feature.transpose(1, 2))  # (b, d_model, kpt_num)
        
        # Position encoding
        pos_enc = kpt_3d.unsqueeze(2) - knn_xyz  # (b, kpt_num, k, 3)
        pos_enc = self.fc_delta(pos_enc)  # (b, kpt_num, k, d_model)
        
        # Feature fusion
        knn_feature = self.fc_delta_1(torch.cat([knn_feature, pos_enc], dim=-1))
        
        # Attention mechanism
        sim = F.cosine_similarity(
            kpt_feature.transpose(1, 2).unsqueeze(2).repeat(1, 1, self.k, 1),
            knn_feature, dim=-1
        )
        sim = F.softmax(sim / self.tau, dim=2)
        kpt_feature = torch.matmul(sim.unsqueeze(2), knn_feature).squeeze(2)
        kpt_feature = F.relu(kpt_feature + pre)
        
        # Global context
        pre = kpt_feature
        pos_enc_abs = self.fc_delta_abs(kpt_3d)
        
        # Inter-keypoint relationships
        dis_mat_l = kpt_3d.unsqueeze(2) - kpt_3d.unsqueeze(1)
        pos_enc_abs_l = self.fc_delta_l(dis_mat_l)
        pos_enc_abs_l = torch.mean(pos_enc_abs_l, dim=2)
        
        pos_enc_l = pos_enc_abs + pos_enc_abs_l
        kpt_num = kpt_3d.shape[1]
        kpt_global = torch.mean(kpt_feature, dim=1, keepdim=True)
        
        # Final fusion
        kpt_feature = self.fuse_mlp(torch.cat([
            kpt_feature.transpose(1, 2),
            kpt_global.transpose(1, 2).repeat(1, 1, kpt_num),
            pos_enc_l.transpose(1, 2)
        ], dim=1))
        
        kpt_feature = F.relu(kpt_feature.transpose(1, 2) + pre)
        pre = kpt_feature
        kpt_feature = self.out_mlp(kpt_feature)
        
        return self.relu(pre + kpt_feature)


class CrossFrameAttention(nn.Module):
    def __init__(self, dim: int = 256, num_heads: int = 4, num_layers: int = 1):
        super().__init__()
        self.embed_scale = nn.Parameter(torch.ones(1))  # learnable scale
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=num_heads,
                dim_feedforward=dim * 2, batch_first=True,
            )
            for _ in range(num_layers)
        ])

    def _frame_embed(self, S: int, dim: int, device) -> torch.Tensor:
        pos = torch.arange(S, device=device).float()
        i = torch.arange(dim // 2, device=device).float()
        theta = pos.unsqueeze(1) / (10000 ** (2 * i / dim)).unsqueeze(0)
        return torch.cat([theta.sin(), theta.cos()], dim=-1)  # [S, dim]

    def forward(self, x: torch.Tensor, B: int, S: int, K: int) -> torch.Tensor:
        x = x.reshape(B, S * K, -1)
        frame_emb = self._frame_embed(S, x.shape[-1], x.device)  # [S, dim]
        frame_emb = frame_emb.repeat_interleave(K, dim=0).unsqueeze(0)  # [1, S*K, dim]
        x = x + self.embed_scale * frame_emb
        for layer in self.layers:
            x = layer(x)
        return x.reshape(B * S, K, -1)


class OPT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True,
                 enable_nocs=True, enable_pose=True, pose_head_type: str = "mlp",
                 kpt_num=128, enable_metric_scale=True, enable_rel_scale=True,
                 use_film_conditioning=True, trainable_sonata=True,
                 cross_frame_z_obj=True, cross_frame_nocs_attn=True,
                 multi_view_pose_aggregation=True):
        super().__init__()

        self.cross_frame_z_obj = cross_frame_z_obj
        self.cross_frame_nocs_attn = cross_frame_nocs_attn
        self.multi_view_pose_aggregation = multi_view_pose_aggregation

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.nocs_head = NOCSHead(dim_in=2 * embed_dim, patch_size=patch_size, use_film_conditioning=use_film_conditioning) if enable_nocs else None
        self.absolute_scale_head = AbsoluteScaleHead(z_obj_dim=256, rgb_feature_dim=128) if enable_metric_scale else None
        self.rel_scale_head = RelScaleHead(z_obj_dim=256, visual_dim=256) if enable_rel_scale else None
        if enable_nocs:
            self.kpt_num = kpt_num
            self.visual_projector = nn.Sequential(nn.Conv1d(256, 128, 1),)
            self.sonata_backbone = SonataBackbone(device="cuda", enable_flash=False, trainable=trainable_sonata)
            self.pts_projector   = nn.Conv1d(1088, 128, 1)  # Linear probe for predicted depth
            self.keypoint_detector = KeyPointDetector(kpt_num=kpt_num, query_dim=256, input_dim=256)  # 128+128=256
            self.visual_geo_fusion = VisualGeoFeatureFusion(K_list=[8, 16, 32], d_model=256)
            self.obj_encoder = ObjectEmbeddingHead(dim=256, num_heads=4)
            if cross_frame_nocs_attn:
                self.cross_frame_attn = CrossFrameAttention(dim=256, num_heads=4, num_layers=1)
        if enable_pose:
            head_type = (pose_head_type or "mlp").strip().lower()
            if head_type == "mlp":
                self.pose_head = PoseNetHead(
                    features=256, sqrt_weight=True, rotation_only=False,
                    multi_view_aggregation=multi_view_pose_aggregation,
                )
            elif head_type == "analytical":
                self.pose_head = PoseHead(sqrt_weight=True, topk=8192, rotation_only=False, min_points=64)
            else:
                raise ValueError(f"Unsupported pose_head_type '{pose_head_type}'. Use 'mlp' or 'analytical'.")
            self.reconstructor = ReconstructorHead(pts_per_kpt=8, ndim=256)
        else:
            self.pose_head = None
            self.reconstructor = None

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None, cat_labels: torch.Tensor = None,
                choose_indices: torch.Tensor = None, nocs_gt: torch.Tensor = None,
                depth_sensor: torch.Tensor = None, category_names: list = None,
                intrinsics: torch.Tensor = None, crop_boxes: torch.Tensor = None,
                masks: torch.Tensor = None, use_gt_intrinsics: bool = False):
        """
        Forward pass of the OPT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            choose_indices (torch.Tensor, optional): Pre-computed choose indices from dataset sampling.
                Shape: [B, S, K] where K is the number of sampled points per frame.
                Default: None
            cat_labels (torch.Tensor, optional): Category labels for each sequence in the batch.
                Shape: [B, S] where each value is a category ID (0-12 for NOCS/HouseCat).
                Default: None
            nocs_gt (torch.Tensor, optional): Ground truth dense NOCS maps for supervision.
                Shape: [B, S, H, W, 3] where values are in [-0.5, 0.5] range.
                Default: None
            depth_sensor (torch.Tensor, optional): Sensor depth maps for each pixel.
                Shape: [B, S, H, W, 1]. Used to estimate the relative scale when metric
                supervision is enabled. Default: None
            category_names (list, optional): Per-frame or per-sequence category names.
                Default: None
            intrinsics (torch.Tensor, optional): Camera intrinsics matrix.
                Shape: [B, S, 3, 3] or [3, 3] (broadcast to all frames).
                **Camera-agnostic design**: Intrinsics are always predicted by camera_head.
                - If provided: Used for loss computation only (stored as intrinsics_enc_gt)
                - If use_gt_intrinsics=True at eval: FoV in pose_enc is replaced with GT, so depth-to-cam uses GT intrinsics.
                - If not provided: No GT available, but prediction still works (camera-agnostic)
                Default: None
            use_gt_intrinsics (bool): If True and intrinsics is provided, replace predicted FoV in pose_enc with GT
                (converted to pose_enc format). Used at evaluation/test to isolate pose vs intrinsic errors. Default: False.
            crop_boxes (torch.Tensor, optional): Relative crop position/size features.
                Expected shape [B, S, 4] with values such as [cx, cy, log_w, log_h] in normalized coordinates.
                Default: None
            masks (torch.Tensor, optional): Foreground masks aligned with images.
                Expected shape [B, S, H, W] (or [B, S, 1, H, W]).
                If provided, masked pixels are replaced by a learned mask token before patch embedding.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - intrinsics_enc (torch.Tensor): Camera intrinsics encoding (FoV) with shape [B, S, 2] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

        If metric scale prediction is enabled (enable_metric_scale=True), returns in addition:
        - abs_translation (torch.Tensor): Absolute translation from sensor depth [B, S, 3]
        - abs_size (torch.Tensor): Absolute size from sensor depth [B, S, 3]
        - derived_scale (torch.Tensor): Scale derived from abs vs norm comparison [B, S]
        - log_derived_scale (torch.Tensor): Log of derived scale [B, S]
        
        The derived_scale is computed by comparing:
          - abs_translation & abs_size (from AbsoluteScaleHead using sensor depth)
          - pose["translation"] & pose["size"] (from PoseNetHead, normalized predictions)
        This implicit scale derivation is more reliable than direct regression.
        
        Note: Requires depth_sensor input when enable_metric_scale=True.
        """        
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
            
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images, masks=masks)

        predictions = {}

        with torch.amp.autocast("cuda", enabled=False):
            pose_enc_for_projection = None
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                pose_enc_for_projection = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list
                if intrinsics is not None:
                    B, S = images.shape[:2]
                    image_hw = images.shape[-2:]
                    K = intrinsics.to(device=images.device, dtype=images.dtype)
                    if K.dim() == 2:
                        K = K.unsqueeze(0).unsqueeze(0).expand(B, S, 3, 3)
                    elif K.dim() == 3:
                        K = K.unsqueeze(1).expand(B, S, 3, 3)
                    elif K.shape[1] == 1 and S > 1:
                        K = K.expand(B, S, 3, 3)
                    gt_fov = self._intrinsics_matrix_to_enc(K, image_hw)  # [B, S, 2]
                    predictions["intrinsics_enc_gt"] = gt_fov
                    # Replace FoV for any geometry projection path at eval/test.
                    if use_gt_intrinsics:
                        pose_enc_for_projection = pose_enc_for_projection.clone()
                        pose_enc_for_projection[..., 7:9] = gt_fov.to(pose_enc_for_projection.device)
                predictions["pose_enc"] = pose_enc_for_projection
            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf
                if choose_indices is not None:
                    cam_pts_dense = self.depth_to_cam_points_dense(depth, pose_enc_for_projection)     # [B,S,H,W,3]
                    normals_dense = self.normals_from_cam_points_dense(cam_pts_dense, True)  # [B,S,H,W,3]
                    predictions["normal"]  = self.gather_normals_at_indices(normals_dense, choose_indices)  # [B,S,K,3]
                    predictions["color"] = self.gather_rgb_at_indices(images, choose_indices)  # [B,S,K,3]
                    predictions["pts_3d_cam"] = self._depth_to_cam_coords_points_torch(depth, pose_enc_for_projection, choose_indices)
                    predictions['center'] = torch.mean(predictions["pts_3d_cam"], dim=2, keepdim=True)
                    predictions["pts_3d_cam_c"] = predictions["pts_3d_cam"] - predictions['center']

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf
                
                # Extract pts_3d_pm from point map at chosen indices for relative pose estimation
                if choose_indices is not None:
                    pts_3d_pm = self.gather_world_points_at_indices(pts3d, choose_indices)  # [B,S,K,3]
                    predictions["pts_3d_pm"] = pts_3d_pm
                    # Unified center across all S frames (pointmap is in frame-0 space)
                    B_pm, S_pm = pts_3d_pm.shape[:2]
                    center_pm = pts_3d_pm.mean(dim=(1, 2), keepdim=True)  # [B, 1, 1, 3]
                    predictions['center_pm'] = center_pm.expand(-1, S_pm, -1, -1).contiguous()  # [B, S, 1, 3]
                    predictions["pts_3d_pm_c"] = predictions["pts_3d_pm"] - predictions['center_pm']

            if self.nocs_head is not None:
                B, S = images.shape[:2]
                if cat_labels is not None and len(cat_labels.shape) == 1:
                    cat_labels = cat_labels.unsqueeze(1).expand(B, S)
                rgb_local, visual_feats = self.nocs_head.extract_visual_features(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                    choose_indices=choose_indices,
                    visual_projector=self.visual_projector,
                    return_dense_feats=True,
                )  # [BS, 128, K], [B, S, C, H, W]
                predictions["visual_feats"] = visual_feats

                # pts_3d_input = predictions["pts_3d_cam_c"].detach()
                pts_3d_input = predictions["pts_3d_pm_c"].detach()
                B, S, K, C = pts_3d_input.shape
                BS = B * S
                
                # Clip coordinates to valid range for sonata octree encoding
                # With grid_size=0.004m and depth<=16: max_range = 2^16 * 0.004 = 262m
                # Clip to ±5m for safety (52x margin) while handling outliers from depth errors
                pts_3d_input = torch.clamp(pts_3d_input, min=-5.0, max=5.0)
                
                colors_input = predictions.get("color", None)                 # [B,S,K,3] or None
                normals_input = predictions.get("normal", None)               # [B,S,K,3] or None
                sonata_feat = self.sonata_backbone.extract(pts_3d_input, colors_input, normals_input
                )  # [B,S,K,1088]

                pts_local = sonata_feat.view(BS, K, 1088)       # [BS,K,1088]
                pts_local = pts_local.transpose(1, 2)           # [BS,1088,K]
                pts_local = self.pts_projector(pts_local)       # [BS,128,K]
                
                pts_3d_reshaped = pts_3d_input.view(BS, K, C)  # [BS, K, 3]
                # pts_local = self.pts_extractor(pts_3d_reshaped)  # [BS, 128, K]
                fused_local = torch.cat((pts_local, rgb_local), dim=1).transpose(1, 2)  # [BS, K, 256]

                _, heat_map = self.keypoint_detector(rgb_local, pts_local)  # [BS, kpt_num, K]
                kpt_3d_flat = torch.bmm(heat_map, pts_3d_reshaped)  # [BS, kpt_num, 3]
                kpt_feature_flat = torch.bmm(heat_map, fused_local)  # [BS, kpt_num, 256]

                kpt_feature_flat = self.visual_geo_fusion(
                    kpt_feature_flat,
                    kpt_3d_flat.detach(),
                    fused_local,
                    pts_3d_reshaped,
                )  # [BS, kpt_num, 256]

                # Cross-frame attention: let keypoints from different frames
                # exchange information before NOCS prediction (no-op when S=1).
                if self.cross_frame_nocs_attn:
                    kpt_feature_flat = self.cross_frame_attn(
                        kpt_feature_flat, B, S, self.kpt_num,
                    )  # [BS, kpt_num, 256]

                # Object embedding from keypoint features.
                z_obj_flat = self.obj_encoder(kpt_feature_flat)  # [BS, 256]

                # Cross-frame z_obj pooling: share a single object embedding
                # across all S frames (identity when S=1).
                if self.cross_frame_z_obj:
                    z_obj_seq = z_obj_flat.view(B, S, -1)
                    z_obj_mean = z_obj_seq.mean(dim=1, keepdim=True)          # [B, 1, 256]
                    z_obj_flat = z_obj_mean.expand(B, S, -1).reshape(BS, -1)  # [BS, 256]

                # Predict NOCS using FiLM conditioning on the fused representation
                kpt_nocs_flat = self.nocs_head.predict_nocs(kpt_feature_flat, z_obj_flat)  # [BS, kpt_num, 3]

                # Restore explicit batch/sequence dimensions before pushing to downstream consumers
                kpt_nocs = kpt_nocs_flat.view(B, S, self.kpt_num, 3)
                kpt_3d = kpt_3d_flat.view(B, S, self.kpt_num, 3)
                kpt_feature = kpt_feature_flat.view(B, S, self.kpt_num, kpt_feature_flat.shape[-1])
                heat_map = heat_map.view(B, S, self.kpt_num, heat_map.shape[-1])

                predictions["nocs"] = kpt_nocs
                predictions["pred_kpt_3d"] = kpt_3d + predictions["center_pm"]
                predictions["heat_map"] = heat_map  # Store heatmap for loss computation
                predictions["z_obj"] = z_obj_flat.view(B, S, 256)

                if self.rel_scale_head is not None:
                    # The input image is already an instance crop; use global crop context
                    # plus mask-focused ROI pooling and crop/camera metadata for scale cues.
                    z_obj_seq = predictions["z_obj"].detach()  # [B, S, 256]
                    visual_feats_detached = visual_feats.detach()
                    visual_global = visual_feats_detached.mean(dim=(-1, -2))  # [B, S, C]
                    visual_roi = self._masked_average_pool(visual_feats_detached, masks)
                    rel_scale_meta = self._build_rel_scale_metadata(
                        crop_boxes=crop_boxes,
                        masks=masks,
                        pose_enc=predictions.get("pose_enc", None),
                        ref_tensor=visual_global,
                    )
                    log_scale_rgb, scale_conf = self.rel_scale_head(
                        z_obj_seq,
                        visual_global,
                        roi_feats=visual_roi,
                        meta_feats=rel_scale_meta,
                    )

                    predictions["log_avg_scale_per_frame"] = log_scale_rgb  # [B, S]

                    valid_mask = torch.isfinite(log_scale_rgb)
                    exp_input = torch.where(valid_mask, log_scale_rgb, torch.zeros_like(log_scale_rgb))
                    avg_scale_rgb = torch.exp(exp_input)
                    avg_scale_rgb = avg_scale_rgb.masked_fill(~valid_mask, float("nan"))

                    predictions["avg_scale_rgb"] = avg_scale_rgb
                    predictions["avg_scale_rgb_confidence"] = scale_conf

                # Construct keypoint NOCS ground truth if provided
                if nocs_gt is not None and choose_indices is not None:
                    kpt_nocs_gt = self._construct_keypoint_nocs_gt(nocs_gt, choose_indices, heat_map.detach())
                    predictions["kpt_nocs_gt"] = kpt_nocs_gt

                # Absolute scale prediction from sensor depth
                if self.absolute_scale_head is not None and choose_indices is not None and depth_sensor is not None:
                    BS = B * S
                    depth_sensor_device = depth_sensor.to(
                        device=rgb_local.device, dtype=rgb_local.dtype, non_blocking=True
                    )
                    # Use the same pose encoding as other geometry projections, including GT FoV when requested.
                    pts_3d_sensor = self._depth_to_cam_coords_points_torch(
                        depth_sensor_device, pose_enc_for_projection, choose_indices
                    )  # [B,S,K,3]
                    
                    # Check which frames have valid sensor depth (not all inf/nan)
                    valid_mask = torch.isfinite(pts_3d_sensor).all(dim=-1).any(dim=-1)  # [B,S]
                    
                    if valid_mask.any():
                        # Compute sensor depth center from all K points (not from pred depth center)
                        sensor_center = torch.mean(pts_3d_sensor, dim=2, keepdim=True)  # [B,S,1,3]
                        pts_3d_sensor_c = pts_3d_sensor - sensor_center  # [B,S,K,3] centered
                        
                        # Prepare inputs for absolute scale head (use all K sampled points)
                        pts_3d_sensor_c_flat = pts_3d_sensor_c.view(BS, -1, 3)  # [BS, K, 3] centered
                        # Detach RGB features: serve as auxiliary visual cues without affecting NOCS learning
                        rgb_features_flat = rgb_local.detach().transpose(1, 2)  # [BS, K, 128]
                        
                        # Predict absolute translation (offset from sensor center) and size
                        # Main signal: GT pointcloud geometry, Auxiliary: RGB appearance (detached)
                        abs_translation_offset, abs_size = self.absolute_scale_head(
                            pts_3d_sensor_c_flat, rgb_features_flat, z_obj_flat.detach()
                        )  # [BS, 3], [BS, 3]
                        
                        abs_translation = abs_translation_offset + sensor_center.view(BS, 3)
                        abs_translation_view = abs_translation.view(B, S, 3)
                        abs_size_view = abs_size.view(B, S, 3)
                        
                        if not self.training:
                            valid_mask_expanded = valid_mask.unsqueeze(-1)  # [B, S, 1]
                            abs_translation_view = torch.where(
                                valid_mask_expanded, 
                                abs_translation_view, 
                                torch.tensor(float('nan'), device=abs_translation_view.device)
                            )
                            abs_size_view = torch.where(
                                valid_mask_expanded, 
                                abs_size_view, 
                                torch.tensor(float('nan'), device=abs_size_view.device)
                            )
                            predictions["abs_valid_mask"] = valid_mask  # [B,S]
                        
                        predictions["abs_translation"] = abs_translation_view
                        predictions["abs_size"] = abs_size_view

            if self.pose_head is not None:
                delta_r, delta_t, delta_s = None, None, None
                kpt_3d_aug = kpt_3d
                center_aug = predictions["center_pm"]
                
                if self.training:
                    device = kpt_3d.device
                    mv = (isinstance(self.pose_head, PoseNetHead)
                          and self.pose_head.multi_view_aggregation)
                    if mv:
                        # Shared augmentation across S frames (all kpt_3d in frame-0 space)
                        delta_r, delta_t, delta_s = generate_augmentation(B, device)
                        delta_r = delta_r.view(B, 1, 3, 3).expand(B, S, 3, 3).contiguous()
                        delta_t = delta_t.view(B, 1, 1, 3).expand(B, S, 1, 3).contiguous()
                        delta_s = delta_s.view(B, 1, 1).expand(B, S, 1).contiguous()
                    else:
                        delta_r, delta_t, delta_s = generate_augmentation(B * S, device)
                        delta_r = delta_r.view(B, S, 3, 3)
                        delta_t = delta_t.view(B, S, 1, 3)
                        delta_s = delta_s.view(B, S, 1)
                    
                    # Apply augmentation: (pts - delta_t) / delta_s @ delta_r
                    kpt_3d_aug = (kpt_3d - delta_t) / delta_s.unsqueeze(-1) @ delta_r
                    center_aug = (predictions["center_pm"] - delta_t) / delta_s.unsqueeze(-1) @ delta_r
                
                if isinstance(self.pose_head, PoseNetHead):
                    z_obj_for_pose = predictions.get("z_obj", None)
                    pose = self.pose_head(kpt_nocs, kpt_3d_aug, kpt_feature, center=center_aug, z_obj=z_obj_for_pose)
                else:
                    pose = self.pose_head(
                        kpt_nocs.view(B * S, self.kpt_num, 3),
                        (kpt_3d_aug + center_aug.detach()).view(B * S, self.kpt_num, 3),
                        kpt_feature.view(B * S, self.kpt_num, -1),
                    )
                
                # Transform predictions back if augmentation was applied
                if self.training and delta_r is not None:
                    pose["rotation"] = delta_r @ pose["rotation"]
                    t_reshaped = pose["translation"].unsqueeze(-1)  # [B, S, 3, 1]
                    t_transformed = delta_s * (delta_r @ t_reshaped).squeeze(-1)  # [B, S, 3]
                    pose["translation"] = delta_t.squeeze(2) + t_transformed
                    pose["size"] = delta_s * pose["size"]
                    if "scale" in pose:
                        pose["scale"] = delta_s * pose["scale"]
                
                predictions["abs_pose"] = pose
                
                # Run Reconstructor if available
                if self.reconstructor is not None:
                    kpt_3d_for_recon = kpt_3d_aug.view(B * S, self.kpt_num, 3).transpose(1, 2)  # [BS, 3, kpt_num]
                    kpt_feature_for_recon = kpt_feature.view(B * S, self.kpt_num, -1).transpose(1, 2)  # [BS, C, kpt_num]
                    
                    recon_model, recon_delta = self.reconstructor(kpt_3d_for_recon, kpt_feature_for_recon)
                    # recon_model: [BS, 3, pts_per_kpt*kpt_num]
                    if self.training and delta_r is not None:
                        recon_model_reshaped = recon_model.transpose(1, 2)  # [BS, pts_per_kpt*kpt_num, 3]
                        # Apply inverse transformation: (recon @ delta_r^T) * delta_s + delta_t
                        # This undoes: (pts - delta_t) / delta_s @ delta_r
                        # Inverse: (recon @ delta_r^T) * delta_s + delta_t
                        delta_r_bs = delta_r.view(B * S, 3, 3).transpose(1, 2)  # [BS, 3, 3]
                        delta_s_bs = delta_s.view(B * S, 1, 1)  # [BS, 1, 1] for proper broadcasting
                        delta_t_bs = delta_t.view(B * S, 1, 3)  # [BS, 1, 3]
                        recon_model_reshaped = (recon_model_reshaped @ delta_r_bs) * delta_s_bs + delta_t_bs
                        recon_model = recon_model_reshaped.transpose(1, 2)  # [BS, 3, pts_per_kpt*kpt_num]
                    
                    # Add center back to reconstructed model (use original center, not augmented)
                    recon_model_reshaped = recon_model.transpose(1, 2)  # [BS, pts_per_kpt*kpt_num, 3]
                    recon_model_reshaped = recon_model_reshaped + predictions["center_pm"].view(B * S, 1, 3)
                    recon_model = recon_model_reshaped.transpose(1, 2)  # [BS, 3, pts_per_kpt*kpt_num]
                    
                    predictions["recon_model"] = recon_model.view(B, S, 3, -1)  # [B, S, 3, pts_per_kpt*kpt_num]
                    predictions["recon_delta"] = recon_delta.view(B, S, -1, 3)  # [B, S, pts_per_kpt*kpt_num, 3]
                
                # Derive scale from absolute vs normalized predictions.
                # abs_translation is per-frame (camera_s space from sensor depth),
                # while pose["translation"] under multi-view is canonical -> frame-0.
                # Only frame 0 has matching coordinate systems, so restrict there.
                if self.absolute_scale_head is not None and "abs_translation" in predictions and "abs_size" in predictions:
                    abs_t = predictions["abs_translation"]       # [B, S, 3]
                    abs_s = predictions["abs_size"]              # [B, S, 3]
                    norm_t = pose["translation"]                 # [B, S, 3]
                    norm_s = pose["size"]                        # [B, S, 3]
                    abs_valid_mask = predictions.get("abs_valid_mask", None)

                    mv = (isinstance(self.pose_head, PoseNetHead)
                          and self.pose_head.multi_view_aggregation and S > 1)
                    if mv:
                        abs_t = abs_t[:, 0:1]
                        abs_s = abs_s[:, 0:1]
                        norm_t = norm_t[:, 0:1]
                        norm_s = norm_s[:, 0:1]
                        if abs_valid_mask is not None:
                            abs_valid_mask = abs_valid_mask[:, 0:1]

                    derived_scale = self.absolute_scale_head.compute_scale_from_absolute(
                        abs_translation=abs_t,
                        abs_size=abs_s,
                        norm_translation=norm_t,
                        norm_size=norm_s,
                        reduction='geometric_mean',
                        valid_mask=abs_valid_mask,
                    )  # [B, S] or [B, 1]

                    if mv:
                        derived_scale = derived_scale.expand(B, S)

                    predictions["derived_scale"] = derived_scale

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        return predictions
    
    def _intrinsics_enc_to_matrix(self, intrinsics_enc, image_size_hw):
        """
        Convert intrinsics encoding (FoV) to intrinsics matrix.
        
        Args:
            intrinsics_enc: [B, S, 2] field of view predictions [fov_h, fov_w]
            image_size_hw: (H, W) image dimensions
            
        Returns:
            intrinsics: [B, S, 3, 3] camera intrinsics matrix
        """
        H, W = image_size_hw
        fov_h = intrinsics_enc[..., 0]  # [B, S]
        fov_w = intrinsics_enc[..., 1]  # [B, S]
        
        # Convert FoV to focal lengths
        fy = (H / 2.0) / torch.tan(fov_h / 2.0)  # [B, S]
        fx = (W / 2.0) / torch.tan(fov_w / 2.0)  # [B, S]
        
        # Build intrinsics matrix
        intrinsics = torch.zeros(intrinsics_enc.shape[:2] + (3, 3), device=intrinsics_enc.device, dtype=intrinsics_enc.dtype)
        intrinsics[..., 0, 0] = fx
        intrinsics[..., 1, 1] = fy
        intrinsics[..., 0, 2] = W / 2
        intrinsics[..., 1, 2] = H / 2
        intrinsics[..., 2, 2] = 1.0
        
        return intrinsics
    
    def _intrinsics_matrix_to_enc(self, intrinsics_matrix, image_size_hw):
        """
        Convert intrinsics matrix to FoV encoding (inverse of _intrinsics_enc_to_matrix).
        
        Args:
            intrinsics_matrix: [B, S, 3, 3] camera intrinsics matrix
            image_size_hw: (H, W) image dimensions
            
        Returns:
            intrinsics_enc: [B, S, 2] field of view encoding [fov_h, fov_w]
        """
        H, W = image_size_hw
        
        fx = intrinsics_matrix[..., 0, 0]  # [B, S]
        fy = intrinsics_matrix[..., 1, 1]  # [B, S]
        
        # Convert focal lengths to FoV (inverse of _intrinsics_enc_to_matrix)
        fov_w = 2.0 * torch.atan(W / (2.0 * fx))  # [B, S]
        fov_h = 2.0 * torch.atan(H / (2.0 * fy))  # [B, S]
        
        intrinsics_enc = torch.stack([fov_h, fov_w], dim=-1)  # [B, S, 2]
        return intrinsics_enc
    
    def _depth_to_cam_coords_points_torch(self, depth_pred, pose_enc, choose_indices):
        """
        Convert depth predictions to camera coordinates using choose_indices sampling.
        This is the torch implementation similar to depth_to_cam_coords_points from geometry.py.
        
        Args:
            depth_pred: [B, S, H, W, 1] depth predictions from depth head
            pose_enc: [B, S, 9] pose encodings containing intrinsics
            choose_indices: [B, S, K] indices for sampling points
            
        Returns:
            pts_3d_cam: [B, S, K, 3] sampled points in camera coordinates
        """
        from opt.utils.pose_enc import pose_encoding_to_extri_intri
        
        if depth_pred.dim() == 5 and depth_pred.size(-1) == 1:
            depth_4d = depth_pred[..., 0]
        elif depth_pred.dim() == 4:
            depth_4d = depth_pred
        else:
            raise ValueError(
                "depth_pred must have shape [B,S,H,W] or [B,S,H,W,1], "
                f"got {tuple(depth_pred.shape)}"
            )

        B, S, H, W = depth_4d.shape
        K = choose_indices.shape[-1]
        
        # Extract intrinsics from pose encoding
        _, intrinsics = pose_encoding_to_extri_intri(pose_enc, (H, W))  # [B, S, 3, 3]
        
        # Create pixel coordinate grids [H*W]
        u, v = torch.meshgrid(torch.arange(W, device=depth_pred.device), 
                             torch.arange(H, device=depth_pred.device), indexing='xy')
        u = u.flatten().float()  # [H*W]
        v = v.flatten().float()  # [H*W]
        
        # Flatten depth maps [B, S, H*W]
        depth_flat = depth_4d.view(B, S, H * W)
        
        # Sample depth at chosen locations [B, S, K]
        choose_indices_clamped = choose_indices.clamp(0, H*W-1)
        depth_sampled = torch.gather(depth_flat, 2, choose_indices_clamped)
        
        # Sample pixel coordinates at chosen locations [B, S, K]
        u_sampled = u[choose_indices_clamped]  # [B, S, K]
        v_sampled = v[choose_indices_clamped]  # [B, S, K]
        
        # Extract intrinsic parameters [B, S]
        fu = intrinsics[:, :, 0, 0]  # [B, S]
        fv = intrinsics[:, :, 1, 1]  # [B, S]
        cu = intrinsics[:, :, 0, 2]  # [B, S]
        cv = intrinsics[:, :, 1, 2]  # [B, S]
        
        # Convert to camera coordinates following depth_to_cam_coords_points formula
        # Expand intrinsics to match sampled point dimensions [B, S, K]
        fu = fu.unsqueeze(-1).expand(-1, -1, K)
        fv = fv.unsqueeze(-1).expand(-1, -1, K)
        cu = cu.unsqueeze(-1).expand(-1, -1, K)
        cv = cv.unsqueeze(-1).expand(-1, -1, K)
        
        # Apply the same formula as depth_to_cam_coords_points
        x_cam = (u_sampled - cu) * depth_sampled / fu  # [B, S, K]
        y_cam = (v_sampled - cv) * depth_sampled / fv  # [B, S, K]
        z_cam = depth_sampled                          # [B, S, K]
        
        # Stack to form camera coordinates [B, S, K, 3]
        pts_3d_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)
        
        return pts_3d_cam
    
    def _construct_keypoint_nocs_gt(self, nocs_gt, choose_indices, heat_map):
        """
        Construct keypoint-level NOCS ground truth using heatmap-weighted sampling.
        This follows the same approach as compute_nocs_loss in training/loss.py.

        Args:
            nocs_gt: [B, S, H, W, 3] Dense NOCS ground truth maps
            choose_indices: [B, S, K] Indices for sampling points from dense maps
            heat_map: [B, S, kpt_num, K] Attention heatmap from keypoint detector
            
        Returns:
            kpt_nocs_gt: [B, S, kpt_num, 3] Keypoint-level NOCS ground truth
        """
        B, S, H, W, _ = nocs_gt.shape
        kpt_num = heat_map.shape[2]
        K = choose_indices.shape[-1]
        BS = B * S

        heat_map_flat = heat_map.view(BS, kpt_num, K)  # [BS, kpt_num, K]
        choose_indices_flat = choose_indices.view(BS, K)  # [BS, K]
        nocs_gt_flat = nocs_gt.view(BS, H * W, 3)  # [BS, H*W, 3]
        
        max_spatial_idx = H * W - 1
        choose_indices_flat = torch.clamp(choose_indices_flat, 0, max_spatial_idx)
        
        gather_idx = choose_indices_flat.unsqueeze(-1).expand(-1, -1, 3)  # [BS, K, 3]
        nocs_dense_sampled = torch.gather(nocs_gt_flat, 1, gather_idx)  # [BS, K, 3]
        
        # Apply heatmap weighting to get keypoint-level ground truth
        # Use detached heatmap to prevent gradients flowing back to keypoint detector
        kpt_nocs_gt_flat = torch.bmm(heat_map_flat.detach(), nocs_dense_sampled)  # [BS, kpt_num, 3]
        
        # Reshape back to [B, S, kpt_num, 3] format
        kpt_nocs_gt = kpt_nocs_gt_flat.view(B, S, kpt_num, 3)
        
        return kpt_nocs_gt

    def _masked_average_pool(self, features: torch.Tensor, masks: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Pool dense features inside the foreground mask.

        Args:
            features: [B, S, C, Hf, Wf]
            masks: [B, S, H, W] or [B, S, 1, H, W]

        Returns:
            Tensor of shape [B, S, C].
        """
        if masks is None:
            return features.mean(dim=(-1, -2))

        if masks.dim() == 5 and masks.shape[2] == 1:
            masks = masks[:, :, 0]
        elif masks.dim() != 4:
            raise ValueError(f"masks must have shape [B,S,H,W] or [B,S,1,H,W], got {tuple(masks.shape)}")

        B, S, C, Hf, Wf = features.shape
        mask = masks.to(device=features.device, dtype=features.dtype)
        mask = mask.view(B * S, 1, mask.shape[-2], mask.shape[-1])
        mask = F.interpolate(mask, size=(Hf, Wf), mode="bilinear", align_corners=False)
        mask = mask.clamp_(0.0, 1.0).view(B, S, 1, Hf, Wf)

        masked_sum = (features * mask).sum(dim=(-1, -2))
        mask_sum = mask.sum(dim=(-1, -2)).clamp_min(1e-6)
        pooled = masked_sum / mask_sum

        global_pool = features.mean(dim=(-1, -2))
        has_fg = (mask_sum.squeeze(2) > 1e-6).unsqueeze(-1)
        return torch.where(has_fg, pooled, global_pool)

    def _build_rel_scale_metadata(
        self,
        crop_boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
        pose_enc: Optional[torch.Tensor],
        ref_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build compact metadata for RGB-only scale estimation.

        Features: [cx, cy, log_w, log_h, mask_ratio, fov_h, fov_w].
        """
        B, S = ref_tensor.shape[:2]
        meta_parts = []

        if crop_boxes is not None:
            crop_meta = crop_boxes.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
            if crop_meta.dim() == 2:
                crop_meta = crop_meta.unsqueeze(1).expand(B, S, crop_meta.shape[-1])
            elif crop_meta.dim() == 3 and crop_meta.shape[1] == 1 and S > 1:
                crop_meta = crop_meta.expand(B, S, crop_meta.shape[-1])
            meta_parts.append(crop_meta)
        else:
            meta_parts.append(ref_tensor.new_zeros((B, S, 4)))

        if masks is not None:
            if masks.dim() == 5 and masks.shape[2] == 1:
                masks_2d = masks[:, :, 0]
            elif masks.dim() == 4:
                masks_2d = masks
            else:
                raise ValueError(f"masks must have shape [B,S,H,W] or [B,S,1,H,W], got {tuple(masks.shape)}")
            mask_ratio = masks_2d.to(device=ref_tensor.device, dtype=ref_tensor.dtype).mean(dim=(-1, -2), keepdim=True)
            meta_parts.append(mask_ratio)
        else:
            meta_parts.append(ref_tensor.new_zeros((B, S, 1)))

        if pose_enc is not None:
            fov = pose_enc[..., 7:9].to(device=ref_tensor.device, dtype=ref_tensor.dtype)
            meta_parts.append(fov)
        else:
            meta_parts.append(ref_tensor.new_zeros((B, S, 2)))

        return torch.cat(meta_parts, dim=-1)
    
    def depth_to_cam_points_dense(self, depth_pred, pose_enc):
        """
        depth_pred: [B, S, H, W, 1]
        pose_enc:   [B, S, 9]  -> use pose_encoding_to_extri_intri to get intrinsics
        return:     cam_pts: [B, S, H, W, 3]
        """
        from opt.utils.pose_enc import pose_encoding_to_extri_intri
        B, S, H, W, _ = depth_pred.shape
        _, Kmat = pose_encoding_to_extri_intri(pose_enc, (H, W))  # [B,S,3,3]

        # pixel grid (u,v) shape [H,W]
        u, v = torch.meshgrid(
            torch.arange(W, device=depth_pred.device),
            torch.arange(H, device=depth_pred.device),
            indexing='xy'
        )
        u = u[None, None].float()  # [1,1,W,H]
        v = v[None, None].float()  # [1,1,W,H]

        Z = depth_pred[..., 0]  # [B,S,H,W]
        fx = Kmat[:, :, 0, 0].unsqueeze(-1).unsqueeze(-1)  # [B,S,1,1]
        fy = Kmat[:, :, 1, 1].unsqueeze(-1).unsqueeze(-1)
        cx = Kmat[:, :, 0, 2].unsqueeze(-1).unsqueeze(-1)
        cy = Kmat[:, :, 1, 2].unsqueeze(-1).unsqueeze(-1)

        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        cam_pts = torch.stack([X, Y, Z], dim=-1)  # [B,S,H,W,3]
        return cam_pts

    def normals_from_cam_points_dense(self, cam_pts, flip_to_camera=True, eps=1e-8):
        """
        cam_pts: [B,S,H,W,3]
        return:  normals: [B,S,H,W,3]  unit normals (camera coordinate)
        """
        B, S, H, W, _ = cam_pts.shape
        pts = cam_pts.permute(0,1,4,2,3).contiguous()  # [B,S,3,H,W]

        # right/down pad 1 (reflect)
        pts_pad = F.pad(pts.view(B*S, 3, H, W), (0,1,0,1), mode='reflect')  # [B*S,3,H+1,W+1]

        # forward difference, then clip back to (H,W)
        d_ver = (pts_pad[:, :, :, :-1] - pts_pad[:, :, :, 1:])[:, :, :-1, :]  # ∂P/∂x -> [B*S,3,H,W]
        d_hor = (pts_pad[:, :, :-1, :] - pts_pad[:, :, 1:, :])[:, :, :, :-1]  # ∂P/∂y -> [B*S,3,H,W]

        # cross product (hor × ver) -> [B*S,3,H,W]
        n = torch.cross(d_hor, d_ver, dim=1)

        # unitize
        mag = torch.linalg.norm(n, dim=1, keepdim=True).clamp_min(eps)
        n = n / mag

        # optional: flip normals to face camera (same direction as view direction: view direction ~ -P)
        if flip_to_camera:
            P = pts.view(B*S, 3, H, W)
            view_dir = -P / (torch.linalg.norm(P, dim=1, keepdim=True).clamp_min(eps))
            # dot(n, view_dir) < 0 then flip
            flip = (torch.sum(n * view_dir, dim=1, keepdim=True) < 0).float()
            n = n * (1.0 - 2.0 * flip)

        normals = n.view(B, S, 3, H, W).permute(0,1,3,4,2).contiguous()  # [B,S,H,W,3]
        return normals

    def gather_depth_at_indices(self, depth_dense: torch.Tensor, choose_indices: torch.Tensor) -> torch.Tensor:
        """Gather depth values at flattened indices.

        Args:
            depth_dense: [B, S, H, W] or [B, S, H, W, 1]
            choose_indices: [B, S, K]

        Returns:
            torch.Tensor: [B, S, K] sampled depth values.
        """
        if depth_dense.dim() == 5 and depth_dense.size(-1) == 1:
            depth_dense = depth_dense[..., 0]

        B, S, H, W = depth_dense.shape
        N = H * W

        depth_flat = depth_dense.reshape(B, S, N)
        idx = choose_indices.clamp(0, N - 1)
        sampled = torch.gather(depth_flat, 2, idx)
        return sampled

    def gather_normals_at_indices(self, normals_dense, choose_indices):
        """
        normals_dense: [B,S,H,W,3]
        choose_indices: [B,S,K]  (same as depth/point sampling: indices are linear indices flattened from H*W)
        return: normals_sampled: [B,S,K,3]
        """
        B,S,H,W,_ = normals_dense.shape
        N = H * W
        K = choose_indices.shape[-1]

        n_flat = normals_dense.view(B, S, N, 3)
        idx = choose_indices.clamp(0, N-1).unsqueeze(-1).expand(-1,-1,-1,3)  # [B,S,K,3]
        normals_sampled = torch.gather(n_flat, 2, idx)  # [B,S,K,3]
        return normals_sampled

    def gather_rgb_at_indices(self, images: torch.Tensor, choose_indices: torch.Tensor) -> torch.Tensor:
        """
        images:         [B, S, 3, H, W]  (channels-first)
        choose_indices: [B, S, K]        (linear indices over H*W, same as used for depth)
        returns:        [B, S, K, 3]     (RGB aligned with sampled points/normals)
        """
        B, S, C, H, W = images.shape
        N = H * W
        K = choose_indices.shape[-1]

        # Flatten to [B, S, C, N]
        img_flat = images.reshape(B, S, C, N)

        # Expand indices to gather over the last dim (N)
        idx = choose_indices.clamp(0, N - 1).unsqueeze(2).expand(-1, -1, C, K)  # [B,S,C,K]

        rgb_cs = torch.gather(img_flat, dim=3, index=idx)  # [B,S,C,K]
        rgb = rgb_cs.permute(0, 1, 3, 2).contiguous()      # [B,S,K,3]
        return rgb

    def gather_world_points_at_indices(self, world_points: torch.Tensor, choose_indices: torch.Tensor) -> torch.Tensor:
        """
        world_points:   [B, S, H, W, 3]  (point map from point_head)
        choose_indices: [B, S, K]        (linear indices over H*W)
        returns:        [B, S, K, 3]     (3D points from point map at sampled locations)
        """
        B, S, H, W, _ = world_points.shape
        N = H * W
        K = choose_indices.shape[-1]

        # Flatten to [B, S, N, 3]
        pts_flat = world_points.reshape(B, S, N, 3)

        # Expand indices to gather over spatial dim
        idx = choose_indices.clamp(0, N - 1).unsqueeze(-1).expand(-1, -1, -1, 3)  # [B,S,K,3]

        pts_sampled = torch.gather(pts_flat, 2, idx)  # [B,S,K,3]
        return pts_sampled


class VGGT(OPT):
    """Backward-compatible alias for the original VGGT class name."""

    pass
