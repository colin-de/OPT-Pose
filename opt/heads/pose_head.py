# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Project-owned portions of this file are licensed under CC-BY-NC-SA-4.0.
# See LICENSE and NOTICE for details. Third-party notices remain applicable.

# vggt/heads/pose_head.py
# Per-frame Sim(3) pose head (class name: PoseHead).
# Returns a pose for each frame by aligning NOCS to pointmap
# independently per frame (weighted Umeyama). Optionally uses an
# object mask to zero-out background pixels before weighting.
#
# Forward signature matches your VGGT.forward call:
#     pose = self.pose_head(nocs, pts3d, nocs_conf, pts3d_conf)
#
# Outputs:
#   - pose: dict with per-frame outputs
#       "s_seq":    [B, S, 1]
#       "R_seq":    [B, S, 3, 3]
#       "t_seq":    [B, S, 3]
#       "T_seq_4x4":[B, S, 4, 4]
#   - (no pose_conf)
#
# Notes:
#   - Fully differentiable (torch.linalg.svd); no losses are computed here.
#   - If your NOCS is in [0,1], centerize by subtracting 0.5 upstream to stabilize t.
#   - Optional masked random sampling to limit compute on high-res inputs (per-frame).

from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn


__all__ = ["PoseHead"]


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


@torch.amp.autocast("cuda", enabled=False)
def _umeyama_sim3_seq(
    X: torch.Tensor, Y: torch.Tensor, W: torch.Tensor, eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Weighted, differentiable Umeyama Sim(3) on sequence-concatenated points:
        Y ≈ s * R @ X + t
    Args:
        X: [B, K, 3]
        Y: [B, K, 3]
        W: [B, K]  (assumed already normalized per sequence: sum=1)
    Returns:
        s: [B, 1], R: [B, 3, 3], t: [B, 3]
    """
    B, K, _ = X.shape
    W1 = W.unsqueeze(-1)  # [B, K, 1]

    # Weighted means
    muX = (W1 * X).sum(dim=1, keepdim=True)  # [B,1,3]
    muY = (W1 * Y).sum(dim=1, keepdim=True)  # [B,1,3]

    Xc = X - muX
    Yc = Y - muY

    # Weighted covariance
    Sigma = Xc.transpose(1, 2) @ (W1 * Yc)  # [B, 3, 3]

    # SVD
    U, Svals, Vh = torch.linalg.svd(Sigma)  # U:[B,3,3], Svals:[B,3], Vh:[B,3,3]
    V = Vh.transpose(-2, -1)
    R = V @ U.transpose(-2, -1)

    # Enforce det(R)=+1 (reflection fix)
    detR = torch.det(R)  # [B]
    D = torch.eye(3, device=R.device, dtype=R.dtype).unsqueeze(0).repeat(B, 1, 1)
    D[:, -1, -1] = torch.sign(detR).clamp(min=-1.0, max=1.0)
    R = V @ D @ U.transpose(-2, -1)

    # Isotropic scale: s = trace(R * Sigma) / sum(W * ||Xc||^2)
    trace_RS = torch.einsum("bij,bij->b", R, Sigma).unsqueeze(-1)          # [B,1]
    varX = (W1.squeeze(-1) * (Xc ** 2).sum(dim=-1)).sum(dim=1, keepdim=True).clamp_min(eps)  # [B,1]
    s = (trace_RS / varX).clamp_min(eps)                                   # [B,1]

    # Translation: t = muY - s * R * muX
    t = muY.squeeze(1) - s * (R @ muX.squeeze(1).unsqueeze(-1)).squeeze(-1)  # [B,3]

    return s, R, t

 
class PoseHead(nn.Module):
    """
    Per-frame Sim(3) pose head (one pose per frame).

    __init__(..., rotation_only=False, sqrt_weight=True, min_points=64, topk=1024):
        - rotation_only: if True, only compute R (s=1, t=0 returned for consistency)
        - sqrt_weight:   apply sqrt to fused weights to reduce dynamic range
        - min_points:    minimum effective points per frame; fallback to identity when not met
        - topk:          when point_masks are provided, randomly sample up to K masked points per frame

    forward(nocs, world_pts, nocs_conf=None, world_conf=None, point_masks=None) -> pose
        Inputs:
            nocs:        [B, S, H, W, 3]  (canonical coords)
            world_pts:   [B, S, H, W, 3]  (camera-frame pointmap)
            nocs_conf:   [B, S, H, W]     (optional)
            world_conf:  [B, S, H, W]     (optional)
            point_masks: [B, S, H, W]     (optional object mask; 1=valid object pixels, 0=background)
        Returns:
            pose: dict with per-frame keys "s_seq", "R_seq", "t_seq", "T_seq_4x4"
    """

    def __init__(
        self,
        sqrt_weight: bool = True,
        rotation_only: bool = False,
        min_points: int = 64,
        eps: float = 1e-8,
        topk: int = 1024,
        sampling: str = "fps",  # one of: [fps, weighted, random, topk]
    ):
        super().__init__()
        self.sqrt_weight = sqrt_weight
        self.rotation_only = rotation_only
        self.min_points = min_points
        self.eps = eps
        self.topk = topk
        self.sampling = sampling

    @torch.amp.autocast("cuda", enabled=False)
    def forward(
        self,
        nocs: torch.Tensor,
        world_pts: torch.Tensor,
        nocs_conf: Optional[torch.Tensor] = None,
        world_conf: Optional[torch.Tensor] = None,
        point_masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        assert nocs.ndim == 5 and world_pts.ndim == 5, "Expect [B, S, H, W, 3] inputs."
        B, S, H, W, _ = nocs.shape
        N = H * W

        # NOCS should already be in [-0.5, 0.5] range from NOCS head
        X = nocs                                          # [B,S,H,W,3]
        Y = world_pts                                     # [B,S,H,W,3]

        # Flatten spatial dims
        X = _flatten_hw(X)          # [B,S,N,3]
        Y = _flatten_hw(Y)          # [B,S,N,3]

        # Build fused weights from confidences and optional point masks
        if nocs_conf is None:
            nocs_conf = torch.ones((B, S, H, W), dtype=nocs.dtype, device=nocs.device)
        if world_conf is None:
            world_conf = torch.ones((B, S, H, W), dtype=world_pts.dtype, device=world_pts.device)
        
        # Apply point mask to filter out invalid/background pixels
        if point_masks is not None:
            # Ensure point_masks is float and matches the tensor type
            point_masks = point_masks.float().to(dtype=nocs_conf.dtype, device=nocs_conf.device)
            # Apply mask to confidences - zero out invalid pixels
            nocs_conf = nocs_conf * point_masks
            world_conf = world_conf * point_masks

        W = (nocs_conf * world_conf).reshape(B, S, N).clamp_min(0.0)  # [B,S,N]
        if self.sqrt_weight:
            W = torch.sqrt(W)

        # Per-frame solve: merge batch and frame dims for Umeyama
        X_bs = X.reshape(B * S, N, 3)            # [B*S, N, 3]
        Y_bs = Y.reshape(B * S, N, 3)            # [B*S, N, 3]
        W_bs = W.reshape(B * S, N)               # [B*S, N]

        # If sampling enabled (topk>0), sample K correspondences per frame (FPS/weighted/random/topk)
        if self.topk is not None and self.topk > 0:
            BS = B * S
            K = min(self.topk, N)
            mask_bs: Optional[torch.Tensor] = None
            if point_masks is not None:
                mask_bs = (point_masks.reshape(BS, N) > 0)

            # Per-frame FPS utility (uses NOCS for stability)
            def _fps_indices(points_row: torch.Tensor, mask_row: Optional[torch.Tensor], K: int) -> torch.Tensor:
                Nrow = points_row.shape[0]
                if mask_row is not None and mask_row.any():
                    valid_idx = torch.nonzero(mask_row, as_tuple=False).squeeze(1)
                else:
                    valid_idx = torch.arange(Nrow, device=points_row.device, dtype=torch.long)
                pts = points_row[valid_idx]
                M = pts.shape[0]
                if M == 0:
                    return torch.randint(low=0, high=Nrow, size=(K,), device=points_row.device)
                cap = max(K, 16384)
                if M > cap:
                    sel = torch.randperm(M, device=points_row.device)[:cap]
                    pts = pts[sel]
                    base_idx = valid_idx[sel]
                    M = pts.shape[0]
                else:
                    base_idx = valid_idx
                sel_local = torch.empty(K, dtype=torch.long, device=points_row.device)
                dists = torch.full((M,), float('inf'), device=points_row.device, dtype=pts.dtype)
                init = torch.randint(0, M, (1,), device=points_row.device)
                last = pts[init]
                sel_local[0] = base_idx[init]
                dists = torch.minimum(dists, ((pts - last) ** 2).sum(dim=-1))
                steps = min(K, M)
                for i in range(1, steps):
                    nxt = torch.argmax(dists)
                    sel_local[i] = base_idx[nxt]
                    last = pts[nxt]
                    dists = torch.minimum(dists, ((pts - last) ** 2).sum(dim=-1))
                if K > M:
                    pad = sel_local[:1].repeat(K - M)
                    sel_local = torch.cat([sel_local[:M], pad], dim=0)
                else:
                    sel_local = sel_local[:K]
                return sel_local

            def _sample_indices_per_frame(W_row: torch.Tensor, mask_row: Optional[torch.Tensor], K: int, points_row: torch.Tensor) -> torch.Tensor:
                Nrow = W_row.shape[0]
                # fps
                if self.sampling == "fps":
                    return _fps_indices(points_row, mask_row, K)
                # Determine valid set
                if mask_row is not None:
                    valid = mask_row
                    valid_count = int(valid.sum().item())
                else:
                    valid = torch.ones_like(W_row, dtype=torch.bool)
                    valid_count = Nrow
                if valid_count <= 0:
                    return torch.randint(low=0, high=Nrow, size=(K,), device=W_row.device)
                if self.sampling == "random":
                    candidates = torch.nonzero(valid, as_tuple=False).squeeze(1)
                    if candidates.numel() >= K:
                        perm = torch.randperm(candidates.numel(), device=W_row.device)[:K]
                        return candidates[perm]
                    idx_rep = torch.randint(0, candidates.numel(), (K,), device=W_row.device)
                    return candidates[idx_rep]
                if self.sampling == "weighted":
                    w = torch.where(valid, W_row, torch.zeros_like(W_row))
                    ssum = w.sum()
                    if ssum <= self.eps:
                        candidates = torch.nonzero(valid, as_tuple=False).squeeze(1)
                        if candidates.numel() >= K:
                            perm = torch.randperm(candidates.numel(), device=W_row.device)[:K]
                            return candidates[perm]
                        idx_rep = torch.randint(0, candidates.numel(), (K,), device=W_row.device)
                        return candidates[idx_rep]
                    p = w / ssum
                    replace = (valid_count < K)
                    return torch.multinomial(p, num_samples=K, replacement=replace)
                # topk by weights
                scores = torch.where(valid, W_row, torch.full_like(W_row, -1e9))
                # Use local Nrow (number of candidates in this frame) instead of outer-scope N
                k_eff = min(K, Nrow)
                idx_topk = torch.topk(scores, k=k_eff, dim=0, largest=True).indices
                if k_eff == K:
                    return idx_topk
                pad = idx_topk[:1].repeat(K - k_eff)
                return torch.cat([idx_topk, pad], dim=0)

            idx_list = [
                _sample_indices_per_frame(
                    W_bs[i], None if mask_bs is None else mask_bs[i], K, X_bs[i]
                )
                for i in range(B * S)
            ]
            idx = torch.stack(idx_list, dim=0)  # [BS,K]

            # Gather sampled correspondences and weights
            gather_idx = idx.unsqueeze(-1).expand(-1, -1, 3)               # [BS,K,3]
            X_bs = torch.gather(X_bs, 1, gather_idx)                       # [BS,K,3]
            Y_bs = torch.gather(Y_bs, 1, gather_idx)                       # [BS,K,3]
            W_bs = torch.gather(W_bs, 1, idx)                              # [BS,K]

            # Normalize weights per frame (selected K)
            W_bs = _normalize_weights_seq(W_bs, eps=self.eps)              # [BS,K]

            # Degeneracy guard and solve (batched)
            eff_pts = (W_bs > 0).float().sum(dim=-1)                       # [BS]
            fallback = (eff_pts < self.min_points).float()                 # [BS]

            s_bs, R_bs, t_bs = _umeyama_sim3_seq(X_bs, Y_bs, W_bs, eps=self.eps)  # [BS,1],[BS,3,3],[BS,3]

            # Fallback to identity
            if fallback.any():
                I3 = torch.eye(3, device=nocs.device, dtype=nocs.dtype).unsqueeze(0).repeat(BS, 1, 1)
                ones = torch.ones(BS, 1, device=nocs.device, dtype=nocs.dtype)
                zeros = torch.zeros(BS, 3, device=nocs.device, dtype=nocs.dtype)
                R_bs = torch.where(fallback[:, None, None] > 0, I3, R_bs)
                s_bs = torch.where(fallback[:, None] > 0, ones, s_bs)
                t_bs = torch.where(fallback[:, None] > 0, zeros, t_bs)

            if self.rotation_only:
                s_bs = torch.ones_like(s_bs)
                t_bs = torch.zeros_like(t_bs)

            pass
        else:
            # Use all points per frame (batched)
            # Normalize weights per frame
            W_bs = _normalize_weights_seq(W_bs, eps=self.eps)              # [B*S, N]

            # Degeneracy guard: minimum effective points (by weight support)
            eff_pts = (W_bs > 0).float().sum(dim=-1)                       # [B*S]
            fallback = (eff_pts < self.min_points).float()                 # [B*S]

            # Solve per-frame Sim(3)
            s_bs, R_bs, t_bs = _umeyama_sim3_seq(X_bs, Y_bs, W_bs, eps=self.eps)  # [B*S,1], [B*S,3,3], [B*S,3]

            # Fallback to identity if too few effective points
            if fallback.any():
                I3 = torch.eye(3, device=nocs.device, dtype=nocs.dtype).unsqueeze(0).repeat(B * S, 1, 1)
                ones = torch.ones(B * S, 1, device=nocs.device, dtype=nocs.dtype)
                zeros = torch.zeros(B * S, 3, device=nocs.device, dtype=nocs.dtype)
                R_bs = torch.where(fallback[:, None, None] > 0, I3, R_bs)
                s_bs = torch.where(fallback[:, None] > 0, ones, s_bs)
                t_bs = torch.where(fallback[:, None] > 0, zeros, t_bs)

            if self.rotation_only:
                s_bs = torch.ones_like(s_bs)
                t_bs = torch.zeros_like(t_bs)

            pass

        # Done batched path

        # Reshape back to [B,S,...]
        s_seq = s_bs.reshape(B, S, 1)
        R_seq = R_bs.reshape(B, S, 3, 3)
        t_seq = t_bs.reshape(B, S, 3)

        # Build 4x4 homogeneous transform per frame
        T_seq = torch.zeros(B, S, 4, 4, device=nocs.device, dtype=nocs.dtype)
        T_seq[:, :, :3, :3] = s_seq[..., None] * R_seq                     # [B,S,3,3]
        T_seq[:, :, :3, 3] = t_seq                                         # [B,S,3]
        T_seq[:, :, 3, 3] = 1.0

        pose: Dict[str, torch.Tensor] = {
            "s_seq": s_seq,             # [B,S,1]
            "R_seq": R_seq,             # [B,S,3,3]
            "t_seq": t_seq,             # [B,S,3]
            "T_seq_4x4": T_seq,         # [B,S,4,4]
        }

        return pose
