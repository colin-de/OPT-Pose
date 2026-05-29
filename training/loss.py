# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Project-owned portions of this file are licensed under CC-BY-NC-SA-4.0.
# See LICENSE and NOTICE for details. Third-party notices remain applicable.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
import torch.distributed as dist
import pdb
import torch.nn as nn
import math

from dataclasses import dataclass
from typing import Optional, Tuple, List
from opt.utils import rotation
from opt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri
from train_utils.general import check_and_fix_inf_nan
from math import ceil, floor


# ============================================================================
# Symmetry canonicalization utilities
# ============================================================================

def _axis_rot_from_cos_sin_torch(axis: str, c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Build rotation matrix about an axis given cosine & sine (torch version)."""
    device = c.device
    dtype = c.dtype
    
    if axis == "x":
        R = torch.zeros(3, 3, device=device, dtype=dtype)
        R[0, 0] = 1.0
        R[1, 1] = c
        R[1, 2] = -s
        R[2, 1] = s
        R[2, 2] = c
    elif axis == "y":
        R = torch.zeros(3, 3, device=device, dtype=dtype)
        R[0, 0] = c
        R[0, 2] = s
        R[1, 1] = 1.0
        R[2, 0] = -s
        R[2, 2] = c
    elif axis == "z":
        R = torch.zeros(3, 3, device=device, dtype=dtype)
        R[0, 0] = c
        R[0, 1] = -s
        R[1, 0] = s
        R[1, 1] = c
        R[2, 2] = 1.0
    else:
        raise ValueError("axis must be one of {'x','y','z'}")
    return R


def _extract_cs_for_right_multiply_torch(R: torch.Tensor, axis: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract cos/sin so that right-multiplying cancels axis rotation (torch version)."""
    if axis == "x":
        c = R[..., 1, 1] + R[..., 2, 2]
        s = R[..., 2, 1] - R[..., 1, 2]
    elif axis == "y":
        c = R[..., 0, 0] + R[..., 2, 2]
        s = R[..., 0, 2] - R[..., 2, 0]
    elif axis == "z":
        c = R[..., 0, 0] + R[..., 1, 1]
        s = R[..., 1, 0] - R[..., 0, 1]
    else:
        raise ValueError("axis must be one of {'x','y','z'}")
    
    norm = torch.sqrt(c**2 + s**2).clamp(min=1e-8)
    return c / norm, s / norm


def _get_symmetry_transform_matrix(R: torch.Tensor, axis: str, code: int) -> torch.Tensor:
    """Get symmetry transform matrix S for given axis and code (torch version)."""
    device = R.device
    dtype = R.dtype
    
    if code == 0:
        return torch.eye(3, device=device, dtype=dtype)
    
    c, s = _extract_cs_for_right_multiply_torch(R, axis)
    phi = torch.atan2(s, c)
    
    if code == 1:
        # Reflection symmetry
        S = _axis_rot_from_cos_sin_torch(axis, c, -s)
    elif code == 2:
        # 4-fold rotational symmetry (90 degrees)
        sector = math.pi / 2.0
        phi_q = torch.round(phi / sector) * sector
        c_new = torch.cos(-phi_q)
        s_new = torch.sin(-phi_q)
        S = _axis_rot_from_cos_sin_torch(axis, c_new, s_new)
    elif code == 3:
        # 2-fold rotational symmetry (180 degrees)
        sector = math.pi
        phi_q = torch.round(phi / sector) * sector
        c_new = torch.cos(-phi_q)
        s_new = torch.sin(-phi_q)
        S = _axis_rot_from_cos_sin_torch(axis, c_new, s_new)
    else:
        return torch.eye(3, device=device, dtype=dtype)
    
    return S


def apply_symmetry_to_rotation_batch(pred_rotation: torch.Tensor, sym_codes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply symmetry canonicalization to predicted rotations.
    
    Args:
        pred_rotation: [B, S, 3, 3] predicted rotation matrices
        sym_codes: [B, S, 3] or [B, 3] symmetry codes for [x, y, z] axes (int codes 0-3)
        
    Returns:
        R_canonicalized: [B, S, 3, 3] canonicalized rotations
        S_total: [B, S, 3, 3] total symmetry transform matrices
    """
    B, S = pred_rotation.shape[:2]
    device = pred_rotation.device
    dtype = pred_rotation.dtype
    
    # Handle [B, 3] -> [B, S, 3]
    if sym_codes.dim() == 2 and sym_codes.shape[0] == B and sym_codes.shape[1] == 3:
        sym_codes = sym_codes.unsqueeze(1).expand(B, S, 3)
    
    R_canon = pred_rotation.clone()
    S_total = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(B, S, 3, 3).clone()
    
    # Process each batch and sequence element
    for b in range(B):
        for s_idx in range(S):
            codes = sym_codes[b, s_idx].long()
            R = R_canon[b, s_idx]
            S_accum = torch.eye(3, device=device, dtype=dtype)
            
            # Apply symmetry for each axis
            for axis, code in zip(["x", "y", "z"], codes):
                code_val = int(code.item())
                if code_val > 0:
                    S_axis = _get_symmetry_transform_matrix(R.unsqueeze(0).unsqueeze(0), axis, code_val)
                    S_axis = S_axis.squeeze()
                    R = R @ S_axis
                    S_accum = S_accum @ S_axis
            
            R_canon[b, s_idx] = R
            S_total[b, s_idx] = S_accum
    
    return R_canon, S_total


# ============================================================================
# End of symmetry canonicalization utilities (DEPRECATED for rotation loss)
# The above apply_symmetry_to_rotation_batch uses torch.round which creates
# non-differentiable, discontinuous canonicalization. Use the symmetry-aware
# min-over-equivalents loss below instead.
# ============================================================================


# ============================================================================
# Symmetry-aware rotation loss (min-over-equivalents, differentiable)
# ============================================================================

def _build_axis_rotation_matrices_batch(axis: str, n: int, device, dtype) -> torch.Tensor:
    """
    Build n rotation matrices for n-fold discrete symmetry about a principal axis.

    Args:
        axis: 'x', 'y', or 'z'
        n: number of discrete rotations (e.g. 4 for 90° symmetry, 2 for 180°)
    Returns:
        [n, 3, 3] tensor of rotation matrices (standard convention)
    """
    angles = torch.arange(n, device=device, dtype=dtype) * (2.0 * math.pi / n)
    c = torch.cos(angles)
    s = torch.sin(angles)

    R = torch.zeros(n, 3, 3, device=device, dtype=dtype)
    if axis == 'x':
        R[:, 0, 0] = 1.0
        R[:, 1, 1] = c
        R[:, 1, 2] = -s
        R[:, 2, 1] = s
        R[:, 2, 2] = c
    elif axis == 'y':
        R[:, 0, 0] = c
        R[:, 0, 2] = s
        R[:, 1, 1] = 1.0
        R[:, 2, 0] = -s
        R[:, 2, 2] = c
    elif axis == 'z':
        R[:, 0, 0] = c
        R[:, 0, 1] = -s
        R[:, 1, 0] = s
        R[:, 1, 1] = c
        R[:, 2, 2] = 1.0
    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis}")
    return R


def _generate_discrete_symmetry_group(sym_codes_single: torch.Tensor, device, dtype) -> torch.Tensor:
    """
    Generate all discrete symmetry group elements from sym_codes for one object.

    Discrete codes: 2 -> 4-fold (90 deg), 3 -> 2-fold (180 deg).
    Codes 0 (none) and 1 (continuous) produce identity for the discrete part.
    Uses Cartesian product over axes for multi-axis symmetry.

    Args:
        sym_codes_single: [3] tensor of symmetry codes for x, y, z
    Returns:
        [K, 3, 3] tensor of discrete symmetry rotation matrices
    """
    per_axis = []
    for axis, code_val in zip(['x', 'y', 'z'], sym_codes_single):
        c = int(code_val.item())
        if c == 2:
            per_axis.append(_build_axis_rotation_matrices_batch(axis, 4, device, dtype))
        elif c == 3:
            per_axis.append(_build_axis_rotation_matrices_batch(axis, 2, device, dtype))
        else:
            # code 0 (none) or 1 (continuous) -> just identity for discrete part
            per_axis.append(torch.eye(3, device=device, dtype=dtype).unsqueeze(0))

    # Cartesian product via sequential matmul
    result = per_axis[0]  # [n_x, 3, 3]
    for group in per_axis[1:]:
        n1, n2 = result.shape[0], group.shape[0]
        result = (result.unsqueeze(1) @ group.unsqueeze(0)).reshape(n1 * n2, 3, 3)

    return result


def _project_nocs_to_invariant_continuous(points: torch.Tensor, axis: str) -> torch.Tensor:
    """
    Project NOCS points to invariant subspace for continuous rotational symmetry.

    For continuous symmetry about `axis`, the invariants are:
    - axis 'y': (r_xz, y) where r_xz = sqrt(x² + z²) — distance from y-axis, height
    - axis 'x': (r_yz, x) where r_yz = sqrt(y² + z²)
    - axis 'z': (r_xy, z) where r_xy = sqrt(x² + y²)

    The angle around the symmetry axis is free and not supervised.

    Args:
        points: [..., 3] NOCS points (x, y, z)
        axis: 'x', 'y', or 'z' — the continuous symmetry axis
    Returns:
        [..., 2] invariant coordinates
    """
    eps = 1e-8
    if axis == 'x':
        r = torch.sqrt(points[..., 1] ** 2 + points[..., 2] ** 2 + eps)
        h = points[..., 0]
    elif axis == 'y':
        r = torch.sqrt(points[..., 0] ** 2 + points[..., 2] ** 2 + eps)
        h = points[..., 1]
    elif axis == 'z':
        r = torch.sqrt(points[..., 0] ** 2 + points[..., 1] ** 2 + eps)
        h = points[..., 2]
    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis}")
    return torch.stack([r, h], dim=-1)


def _project_nocs_to_radial(points: torch.Tensor) -> torch.Tensor:
    """
    Project NOCS points to radial distance for spherical / multi-axis continuous symmetry.

    When 2+ axes have continuous symmetry (e.g. sphere), the only invariant is
    r = sqrt(x² + y² + z²) — distance from origin.

    Args:
        points: [..., 3] NOCS points (x, y, z)
    Returns:
        [..., 1] radial distance per point
    """
    eps = 1e-8
    r = torch.sqrt((points ** 2).sum(dim=-1) + eps)
    return r.unsqueeze(-1)


def _generate_symmetry_group_for_nocs(
    sym_codes_single: torch.Tensor,
    device,
    dtype,
    n_continuous_samples: int = 16,  # unused; kept for API compatibility
) -> torch.Tensor:
    """
    Generate symmetry group elements for NOCS min-over-equivalents loss.

    Discrete (code 2, 3): enumerate all rotations as before.
    Continuous (code 1): use identity only — continuous axis is handled via
    invariant projection (_project_nocs_to_invariant_continuous) in the loss,
    not via enumeration.

    Args:
        sym_codes_single: [3] tensor of symmetry codes for x, y, z
        n_continuous_samples: deprecated, unused
    Returns:
        [K, 3, 3] tensor of symmetry rotation matrices
    """
    per_axis = []
    for axis, code_val in zip(['x', 'y', 'z'], sym_codes_single):
        c = int(code_val.item())
        if c == 2:
            per_axis.append(_build_axis_rotation_matrices_batch(axis, 4, device, dtype))
        elif c == 3:
            per_axis.append(_build_axis_rotation_matrices_batch(axis, 2, device, dtype))
        elif c == 1:
            # Continuous: identity only; canonicalization via invariant projection in loss
            per_axis.append(torch.eye(3, device=device, dtype=dtype).unsqueeze(0))
        else:
            per_axis.append(torch.eye(3, device=device, dtype=dtype).unsqueeze(0))

    result = per_axis[0]
    for group in per_axis[1:]:
        n1, n2 = result.shape[0], group.shape[0]
        result = (result.unsqueeze(1) @ group.unsqueeze(0)).reshape(n1 * n2, 3, 3)
    return result


def _continuous_axis_max_trace(R_diff: torch.Tensor, axis: str) -> torch.Tensor:
    """
    For continuous rotational symmetry about `axis`, analytically compute:
        max_theta  tr(R_diff @ R_axis(theta))

    Derivation:
        tr(R_diff @ R_axis(theta)) = C*cos(theta) + S*sin(theta) + fixed
        where C, S depend on R_diff entries and `axis`.
        Maximum = sqrt(C^2 + S^2) + fixed.

    This is fully differentiable w.r.t. R_diff (no torch.round, no argmin).

    Args:
        R_diff: [..., 3, 3] relative rotation matrices (R_pred^T @ R_gt_k)
        axis: 'x', 'y', or 'z'
    Returns:
        [...] maximum trace values
    """
    if axis == 'x':
        C = R_diff[..., 1, 1] + R_diff[..., 2, 2]
        S_val = R_diff[..., 2, 1] - R_diff[..., 1, 2]
        fixed = R_diff[..., 0, 0]
    elif axis == 'y':
        C = R_diff[..., 0, 0] + R_diff[..., 2, 2]
        S_val = R_diff[..., 2, 0] - R_diff[..., 0, 2]
        fixed = R_diff[..., 1, 1]
    elif axis == 'z':
        C = R_diff[..., 0, 0] + R_diff[..., 1, 1]
        S_val = R_diff[..., 1, 0] - R_diff[..., 0, 1]
        fixed = R_diff[..., 2, 2]
    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis}")

    # sqrt with eps for numerical safety (gradient of sqrt at 0 is inf)
    return torch.sqrt(C ** 2 + S_val ** 2 + 1e-12) + fixed


def _trace_to_distance(tr: torch.Tensor, loss_type: str) -> torch.Tensor:
    """
    Convert trace of R_pred^T @ R_gt (or its symmetry-optimized variant) to a
    rotation distance metric.

    Both geodesic and Frobenius ultimately depend on the same trace quantity,
    only the final mapping differs:

        geodesic:   d = arccos(clamp((tr - 1) / 2))           [radians]
        frobenius:  d = sqrt(clamp(6 - 2*tr, min=0))          [unitless]

    The Frobenius identity holds because for rotation matrices A, B:
        ||A - B||_F^2 = tr((A-B)^T (A-B)) = tr(I) - 2 tr(A^T B) + tr(I) = 6 - 2 tr(A^T B)

    Args:
        tr: [...] trace values  (tr(R_pred^T @ R_gt @ S_k)  or  max_theta version)
        loss_type: "geodesic" or "frobenius"
    Returns:
        [...] distance values
    """
    if loss_type == "geodesic":
        cos_a = ((tr - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        return torch.acos(cos_a)
    elif loss_type == "frobenius":
        # ||A - B||_F = sqrt(6 - 2*tr(A^T B)),  clamp for numerical safety
        return torch.sqrt((6.0 - 2.0 * tr).clamp(min=1e-12))
    else:
        raise ValueError(f"Unknown loss_type '{loss_type}', expected 'geodesic' or 'frobenius'")


def symmetry_aware_rotation_loss(
    R_pred: torch.Tensor,
    R_gt: torch.Tensor,
    sym_codes: torch.Tensor,
    loss_type: str = "geodesic",
    soft_min_temperature: float = 0.0,
) -> torch.Tensor:
    """
    Symmetry-aware rotation loss via min-over-equivalents distance.

    Instead of canonicalizing predictions (non-differentiable due to torch.round),
    this enumerates GT equivalent rotations under the symmetry group and computes:

        loss = min_{S_k in Sym} d(R_pred, R_gt @ S_k)

    where d() is either geodesic or Frobenius distance.  Both metrics reduce to
    the same trace optimization (max tr(R_pred^T @ R_gt @ S_k)), only the final
    trace-to-distance mapping differs.

    For continuous symmetry axes, the minimum over the continuous parameter theta is
    computed analytically (fully differentiable).

    Advantages over canonicalization-based approach:
      - No torch.round -> no zero-gradient, no sector-boundary jumps
      - No independent canonicalization of pred/GT -> no "different sector" artifacts
      - Gradient flows cleanly through R_pred via the winning candidate
      - Consistent with ShapeMatch / DenseFusion symmetric loss conventions

    Args:
        R_pred: [B, S, 3, 3] predicted rotation matrices
        R_gt: [B, S, 3, 3] ground truth rotation matrices (may be pre-canonicalized)
        sym_codes: [B, S, 3] or [B, 3] symmetry codes per axis
            0 = no symmetry, 1 = continuous, 2 = 4-fold (90 deg), 3 = 2-fold (180 deg)
        loss_type: "geodesic" (angle in radians) or "frobenius" (||R_pred - R_gt||_F)
        soft_min_temperature: if > 0, use soft-min (log-sum-exp) instead of hard min.
            Provides smoother gradients near candidate boundaries at the cost of a
            slight upward bias.  Recommended range: 0.01 ~ 0.1 (in the distance unit).

    Returns:
        Scalar mean loss (radians for geodesic, unitless for Frobenius)
    """
    B, S = R_pred.shape[:2]
    device = R_pred.device
    dtype = R_pred.dtype

    # Normalize sym_codes shape to [B, S, 3]
    if sym_codes.dim() == 2 and sym_codes.shape == (B, 3):
        sym_codes = sym_codes.unsqueeze(1).expand(B, S, 3)

    # Flatten batch and sequence dims
    N_total = B * S
    R_pred_flat = R_pred.reshape(N_total, 3, 3)
    R_gt_flat = R_gt.reshape(N_total, 3, 3)
    sym_flat = sym_codes.reshape(N_total, 3)

    # Group by unique sym_codes for efficiency
    unique_codes, inverse_idx = torch.unique(sym_flat, dim=0, return_inverse=True)

    distances = torch.zeros(N_total, device=device, dtype=dtype)

    for ui in range(unique_codes.shape[0]):
        code = unique_codes[ui]
        mask = (inverse_idx == ui)

        if not mask.any():
            continue

        R_p = R_pred_flat[mask]  # [M, 3, 3]
        R_g = R_gt_flat[mask]    # [M, 3, 3]

        # Identify symmetry structure
        continuous_axes = [ax for ax, c in zip('xyz', code) if int(c.item()) == 1]
        has_discrete = any(int(c.item()) in (2, 3) for c in code)

        # --- Special case: no symmetry at all ---
        if not continuous_axes and not has_discrete:
            R_diff = R_p.transpose(-1, -2) @ R_g
            tr = R_diff.diagonal(dim1=-2, dim2=-1).sum(-1)
            distances[mask] = _trace_to_distance(tr, loss_type)
            continue

        # --- Special case: >=2 continuous axes -> near-spherical symmetry ---
        if len(continuous_axes) >= 2:
            distances[mask] = 0.0
            continue

        # --- General case ---
        # 1) Build discrete candidates [K, 3, 3]
        S_k = _generate_discrete_symmetry_group(code, device, dtype)  # [K, 3, 3]
        K = S_k.shape[0]

        # 2) R_gt_k = R_gt @ S_k  for all K candidates
        #    [M, 1, 3, 3] @ [1, K, 3, 3] -> [M, K, 3, 3]
        R_g_k = R_g.unsqueeze(1) @ S_k.unsqueeze(0)

        # 3) R_diff_k = R_pred^T @ R_gt_k
        #    [M, 1, 3, 3]^T @ [M, K, 3, 3] -> [M, K, 3, 3]
        R_diff_k = R_p.unsqueeze(1).transpose(-1, -2) @ R_g_k

        # 4) Compute trace for each candidate
        #    Both geodesic and Frobenius reduce to: max trace -> min distance
        if len(continuous_axes) == 0:
            # Pure discrete: standard trace
            tr_k = R_diff_k.diagonal(dim1=-2, dim2=-1).sum(-1)  # [M, K]
        else:
            # One continuous axis: analytical max-trace over theta
            tr_k = _continuous_axis_max_trace(R_diff_k, continuous_axes[0])  # [M, K]

        # 5) Convert trace to distance for each candidate
        dist_k = _trace_to_distance(tr_k, loss_type)  # [M, K]

        # 6) Aggregate over candidates
        if soft_min_temperature > 0:
            tau = soft_min_temperature
            min_dist = -tau * torch.logsumexp(-dist_k / tau, dim=-1).clamp(min=1e-6)
        else:
            # Hard min: passes gradient through the closest candidate only.
            # Unlike torch.round (zero gradient), min has well-defined subgradients.
            min_dist = dist_k.min(dim=-1).values  # [M]

        distances[mask] = min_dist

    return distances.mean()


# ============================================================================
# End of symmetry-aware rotation loss
# ============================================================================


def instance_variance_loss(z_flat: torch.Tensor, seq_id):
    """
    Encourage embeddings from the same instance (same seq_id) to be consistent across frames.
    This helps maintain instance-level coherence in the embedding space.
    
    Args:
        z_flat: [BS, D] object embeddings (L2-normalized)
        seq_id: [BS] or [B, S] sequence IDs identifying same instances
    Returns:
        loss: scalar variance loss (mean squared distance from instance centroid)
    """
    if seq_id is None:
        return z_flat.new_tensor(0.0)
    
    # Flatten seq_id if needed
    seq = seq_id.reshape(-1) if seq_id.dim() == 2 else seq_id
    
    loss = z_flat.new_tensor(0.0)
    n = 0
    
    # For each unique instance, compute variance from mean
    for sid in torch.unique(seq):
        mask = (seq == sid)
        if mask.sum() <= 1:
            continue  # Skip single-frame instances
        
        zi = z_flat[mask]  # Embeddings for this instance
        mu = zi.mean(dim=0, keepdim=True)  # Instance centroid
        loss = loss + ((zi - mu) ** 2).sum(dim=1).mean()
        n += 1
    
    return loss / max(n, 1)


def batch_names_and_index(batch, expected_length: int | None = None):
    """
    Extract category names from batch and build index mapping for text prototype losses.
    
    Args:
        batch: Dict containing 'category_name' key with shape [B], [B,S], or [B,S,1]
    Returns:
        uniq_names: List of unique category names in sorted order
        y_idx: [N] tensor mapping each sample to its category index in uniq_names
    """
    if "category_name" not in batch:
        raise KeyError("batch must contain 'category_name' key for text prototype loss")
    
    names = batch["category_name"]
    
    # Handle different input shapes
    if isinstance(names, torch.Tensor):
        if names.dim() == 1:  # [B]
            # Expand to [B, S] if images are available
            if "images" in batch:
                B, S = batch["images"].shape[:2]
                names = names.unsqueeze(1).expand(B, S)
        elif names.dim() == 3:  # [B, S, 1]
            names = names.squeeze(-1)
        # names should now be [B, S]
        flat_names = names.reshape(-1).tolist()
    elif isinstance(names, (list, tuple)):
        # Flatten arbitrarily nested sequences while preserving order
        flat_names = []
        stack = [names]
        while stack:
            current = stack.pop()
            if isinstance(current, (list, tuple)):
                stack.extend(reversed(current))
            else:
                flat_names.append(current)
    else:
        raise TypeError(f"Unsupported category_name type: {type(names)}")

    # Convert to string list
    flat_names = [str(n) for n in flat_names]

    if expected_length is None:
        if "images" in batch:
            img_shape = batch["images"].shape
            if len(img_shape) >= 2:
                expected_length = img_shape[0] * img_shape[1]
            else:
                expected_length = img_shape[0]
        else:
            expected_length = len(flat_names)

    if len(flat_names) != expected_length:
        if len(flat_names) == 0:
            raise ValueError("No category names found in batch")
        if expected_length % len(flat_names) == 0:
            repeat = expected_length // len(flat_names)
            flat_names = [name for name in flat_names for _ in range(repeat)]
        else:
            raise ValueError(
                f"category_name length {len(flat_names)} incompatible with expected {expected_length}"
            )

    # Preserve order of first appearance when building unique name list
    uniq_names = []
    idx_map = {}
    for name in flat_names:
        if name not in idx_map:
            idx_map[name] = len(uniq_names)
            uniq_names.append(name)

    # Build index mapping for each sample
    y_idx = torch.tensor(
        [idx_map[name] for name in flat_names],
        dtype=torch.long,
        device=batch["images"].device if "images" in batch else None
    )
    
    return uniq_names, y_idx


def supcon_infonce(
    z: torch.Tensor,
    pos_mask: torch.Tensor,
    temperature: float = 0.1,
    eps: float = 1e-8,
    pos_weights: Optional[torch.Tensor] = None,
    positive_mode: str = "logsum",
) -> torch.Tensor:
    """Supervised InfoNCE supporting log-average or log-sum positive reductions."""

    N = z.size(0)
    z = F.normalize(z, dim=-1)
    logits = (z @ z.T) / temperature

    # Improve numerical stability by subtracting per-row maxima.
    logits = logits - logits.max(dim=1, keepdim=True).values

    self_mask = torch.eye(N, dtype=torch.bool, device=z.device)
    denom = torch.logsumexp(logits.masked_fill(self_mask, float("-inf")), dim=1)

    reduction = positive_mode.lower()
    if reduction not in {"logsum", "logavg"}:
        raise ValueError(f"Unsupported positive_mode '{positive_mode}'. Use 'logsum' or 'logavg'.")

    if reduction == "logsum":
        if pos_weights is None:
            pos_logits = logits.masked_fill(~pos_mask, float("-inf"))
        else:
            w = pos_weights.clamp_min(0.0)
            w = w.masked_fill(~pos_mask, 0.0)
            pos_logits = logits + w.clamp_min(eps).log()
            pos_logits = pos_logits.masked_fill(~pos_mask, float("-inf"))
    else:  # logavg
        if pos_weights is None:
            w = pos_mask.float()
        else:
            w = pos_weights.clamp(0.0, 1.0) * pos_mask.float()
        pos_logits = logits + (w + eps).log()
        pos_logits = pos_logits.masked_fill(~pos_mask, float("-inf"))

    num = torch.logsumexp(pos_logits, dim=1)

    valid = pos_mask.any(dim=1)
    if valid.any():
        loss = -(num[valid] - denom[valid]).mean()
    else:
        loss = logits.new_tensor(0.0, requires_grad=True)

    return loss


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _gather_primary_tensor(tensor: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor, int, int]:
    """All-gather the leading dimension of a tensor across ranks (with padding).

    Returns the list of unpadded chunks per rank, the lengths tensor, max length, and total count.
    """

    world_size = dist.get_world_size()
    device = tensor.device
    local_count = tensor.shape[0]
    length = torch.tensor([local_count], device=device, dtype=torch.long)
    length_list = [torch.zeros_like(length) for _ in range(world_size)]
    dist.all_gather(length_list, length)
    counts = torch.stack(length_list, dim=0).squeeze(-1)
    max_count = int(counts.max().item()) if world_size > 0 else local_count

    if max_count == 0:
        chunks = [tensor.new_empty((0,) + tensor.shape[1:]) for _ in range(world_size)]
        total = 0
        return chunks, counts, max_count, total

    if local_count < max_count:
        pad_shape = (max_count - local_count,) + tensor.shape[1:]
        padded = torch.cat([tensor, tensor.new_zeros(pad_shape)], dim=0)
    else:
        padded = tensor

    gather_list = [padded.new_zeros(padded.shape) for _ in range(world_size)]
    to_gather = padded.detach().contiguous()
    dist.all_gather(gather_list, to_gather)

    chunks = [gather_list[i][: int(counts[i].item())].clone() for i in range(world_size)]
    total = int(counts.sum().item())
    return chunks, counts, max_count, total


def _gather_optional_tensor(
    tensor: Optional[torch.Tensor], counts: torch.Tensor, max_count: int
) -> Optional[torch.Tensor]:
    if tensor is None:
        return None

    world_size = dist.get_world_size()
    device = tensor.device
    local_count = tensor.shape[0]
    expected = int(counts[dist.get_rank()].item())
    if local_count != expected:
        return None

    if max_count == 0:
        return tensor.new_empty((0,) + tensor.shape[1:])

    if local_count < max_count:
        pad_shape = (max_count - local_count,) + tensor.shape[1:]
        padded = torch.cat([tensor, tensor.new_zeros(pad_shape)], dim=0)
    else:
        padded = tensor

    gather_list = [padded.new_zeros(padded.shape) for _ in range(world_size)]
    to_gather = padded.detach().contiguous()
    dist.all_gather(gather_list, to_gather)
    chunks = [gather_list[i][: int(counts[i].item())] for i in range(world_size)]
    return torch.cat(chunks, dim=0)


def _flatten_category_names(batch, expected_length: int) -> Optional[List[str]]:
    if "category_name" not in batch:
        return None

    names = batch["category_name"]

    if isinstance(names, torch.Tensor):
        if names.dim() == 1:
            flat_names = names.reshape(-1).tolist()
        elif names.dim() == 3 and names.size(-1) == 1:
            flat_names = names.squeeze(-1).reshape(-1).tolist()
        else:
            flat_names = names.reshape(-1).tolist()
    elif isinstance(names, (list, tuple)):
        flat_names = []
        stack = [names]
        while stack:
            current = stack.pop()
            if isinstance(current, (list, tuple)):
                stack.extend(reversed(current))
            else:
                flat_names.append(current)
    else:
        flat_names = [names]

    if len(flat_names) != expected_length:
        if len(flat_names) == 0:
            return None
        if expected_length % len(flat_names) == 0:
            repeat = expected_length // len(flat_names)
            flat_names = [name for name in flat_names for _ in range(repeat)]
        else:
            return None

    return [str(n) for n in flat_names]


def _gather_category_names(local_names: Optional[List[str]], total: int) -> Optional[List[str]]:
    if local_names is None:
        local_names = []
    world_size = dist.get_world_size()
    gathered: List[List[str]] = [None for _ in range(world_size)]  # type: ignore
    dist.all_gather_object(gathered, local_names)
    combined: List[str] = []
    for names in gathered:
        if names:
            combined.extend(names)
    if len(combined) != total:
        return None
    return combined


def _gather_infonce_context(
    z_flat: torch.Tensor,
    predictions: dict,
    batch: dict,
) -> Optional[dict]:
    if not _dist_ready() or dist.get_world_size() <= 1:
        return None

    with torch.no_grad():
        chunks, counts, max_count, total = _gather_primary_tensor(z_flat.detach())

    if total == 0:
        return None

    rank = dist.get_rank()
    device = z_flat.device

    z_for_loss_chunks = []
    for i, chunk in enumerate(chunks):
        if i == rank:
            z_for_loss_chunks.append(z_flat)
        else:
            z_for_loss_chunks.append(chunk)
    z_for_loss = torch.cat(z_for_loss_chunks, dim=0)
    z_global = torch.cat(chunks, dim=0)

    anchor_mask = torch.zeros(total, dtype=torch.bool, device=device)
    if counts.numel() > 0:
        start = int(counts[:rank].sum().item()) if rank > 0 else 0
        length = int(counts[rank].item())
        anchor_mask[start:start + length] = True

    z_obj = predictions.get("z_obj")
    if z_obj is not None:
        z_obj = z_obj.reshape(-1, z_obj.shape[-1])
        if z_obj.shape[0] == z_flat.shape[0]:
            z_obj_global = _gather_optional_tensor(z_obj.detach(), counts, max_count)
        else:
            z_obj_global = None
    else:
        z_obj_global = None

    seq_id = batch.get("seq_id")
    seq_tensor = None
    if seq_id is not None:
        if isinstance(seq_id, torch.Tensor):
            seq_tensor = seq_id.to(device=device, dtype=torch.long).reshape(-1)
        else:
            seq_tensor = torch.as_tensor(seq_id, dtype=torch.long, device=device).reshape(-1)
        if seq_tensor.numel() != z_flat.shape[0]:
            seq_tensor = None

    if seq_tensor is not None:
        seq_global = _gather_optional_tensor(seq_tensor, counts, max_count)
    else:
        seq_global = None

    local_names = _flatten_category_names(batch, z_flat.shape[0])
    if local_names is not None and len(local_names) != z_flat.shape[0]:
        local_names = None
    category_names = _gather_category_names(local_names, total)
    if category_names is None:
        category_names = [f"sample_{i}" for i in range(total)]

    batch_global = {"category_name": category_names}
    batch_global["images"] = torch.empty(total, device=device)
    if seq_global is not None:
        batch_global["seq_id"] = seq_global

    predictions_global = {"z_obj": z_global}
    if z_obj_global is not None:
        predictions_global["z_obj"] = z_obj_global

    return {
        "z_for_loss": z_for_loss,
        "predictions": predictions_global,
        "batch": batch_global,
        "anchor_mask": anchor_mask,
    }


def build_pos_mask_from_batch(batch, predictions) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Build supervised positive mask (instance or category) for InfoNCE."""

    z = predictions.get("z_obj")
    if z is None:
        raise ValueError("predictions must contain 'z_obj' to build positive mask")

    if z.dim() == 3:
        B, S, D = z.shape
        N = B * S
    else:
        N = z.shape[0]

    device = z.device
    
    # Build instance-based positive mask
    pos_inst = torch.zeros(N, N, dtype=torch.bool, device=device)
    seq = batch.get("seq_id")
    if isinstance(seq, torch.Tensor):
        seq = seq.reshape(-1).to(device).long()
        if seq.numel() == N:
            pos_inst = seq[:, None].eq(seq[None, :])
            pos_inst.fill_diagonal_(False)
    elif seq is not None:
        seq_tensor = torch.as_tensor(seq, dtype=torch.long).reshape(-1)
        if seq_tensor.numel() == N:
            seq_tensor = seq_tensor.to(device)
            pos_inst = seq_tensor[:, None].eq(seq_tensor[None, :])
            pos_inst.fill_diagonal_(False)
    
    # Build category-based positive mask
    try:
        _, y_idx = batch_names_and_index(batch, expected_length=N)
        y_idx = y_idx.to(device)
        pos_cls = y_idx[:, None].eq(y_idx[None, :])
        pos_cls.fill_diagonal_(False)
    except Exception:
        pos_cls = torch.zeros(N, N, dtype=torch.bool, device=device)
    
    # Combine instance and category masks
    pos_mask = pos_inst | pos_cls
    soft_weights = None

    return pos_mask, soft_weights


@dataclass(eq=False)
class MultitaskLoss(torch.nn.Module):
    """
    Multi-task loss module that combines different loss types for VGGT.
    
    Supports:
    - Camera loss
    - Depth loss 
    - Point loss
    - NOCS loss
    - Pose loss
    - Representation regularizers (InfoNCE)
    - Tracking loss (not cleaned yet, dirty code is at the bottom of this file)
    """
    def __init__(self, camera=None, depth=None, point=None, nocs=None, pose=None,
                 scale=None, rel_scale=None, recon=None, track=None, infonce=None,
                 cycle_consistency=None, **kwargs):
        super().__init__()
        # Loss configuration dictionaries for each task
        self.camera = camera
        self.depth = depth
        self.point = point
        self.nocs = nocs
        self.pose = pose
        self.scale = scale
        self.rel_scale = rel_scale
        self.recon = recon
        self.track = track
        self.cycle_consistency = cycle_consistency
        self.infonce_cfg = infonce or None
        self.infonce_weight = 0.0
        self.infonce_tau = 0.1
        self.instance_var_weight = 0.0
        self.infonce_pos_mode = "logsum"
        if self.infonce_cfg is not None:
            self.infonce_weight = self.infonce_cfg.get("weight", 0.0)
            self.infonce_tau = self.infonce_cfg.get("temperature", 0.1)
            self.instance_var_weight = self.infonce_cfg.get("instance_weight", 0.0)
            self.infonce_pos_mode = self.infonce_cfg.get("positive_mode", "logsum")

    def forward(self, predictions, batch) -> torch.Tensor:
        """
        Compute the total multi-task loss.
        
        Args:
            predictions: Dict containing model predictions for different tasks
            batch: Dict containing ground truth data and masks
            
        Returns:
            Dict containing individual losses and total objective
        """
        total_loss = 0
        loss_dict = {}
        
        # Camera pose loss - if pose encodings are predicted
        if "pose_enc_list" in predictions:
            camera_loss_dict = compute_camera_loss(predictions, batch, **self.camera)   
            camera_loss = camera_loss_dict["loss_camera"] * self.camera["weight"]   
            total_loss = total_loss + camera_loss
            loss_dict.update(camera_loss_dict)
        
        # Depth estimation loss - if depth maps are predicted
        if "depth" in predictions:
            depth_loss_dict = compute_depth_loss(predictions, batch, **self.depth)
            depth_loss = depth_loss_dict["loss_conf_depth"] + depth_loss_dict["loss_reg_depth"] + depth_loss_dict["loss_grad_depth"]
            depth_loss = depth_loss * self.depth["weight"]
            total_loss = total_loss + depth_loss
            loss_dict.update(depth_loss_dict)

        # 3D point reconstruction loss - if world points are predicted
        if "world_points" in predictions:
            point_loss_dict = compute_point_loss(predictions, batch, **self.point)
            point_loss = point_loss_dict["loss_conf_point"] + point_loss_dict["loss_reg_point"] + point_loss_dict["loss_grad_point"]
            point_loss = point_loss * self.point["weight"]
            total_loss = total_loss + point_loss
            loss_dict.update(point_loss_dict)

        # NOCS loss - if NOCS maps are predicted
        if "nocs" in predictions and self.nocs:
            nocs_loss_dict = compute_nocs_loss(predictions, batch, **self.nocs)
            nocs_component_weights = self.nocs.get("component_weights", {})
            nocs_loss = (
                nocs_component_weights.get("conf", 1.0) * nocs_loss_dict["loss_conf_nocs"]
                + nocs_component_weights.get("reg", 1.0) * nocs_loss_dict["loss_reg_nocs"]
                + nocs_component_weights.get("grad", 1.0) * nocs_loss_dict["loss_grad_nocs"]
            )
            nocs_loss = nocs_loss * self.nocs.get("weight", 1.0)
            total_loss = total_loss + nocs_loss
            loss_dict.update(nocs_loss_dict)
            loss_dict["loss_nocs_total"] = nocs_loss

            if "pred_kpt_3d" in predictions and self.nocs.get("enable_keypoint_losses", False):
                keypoint_loss_dict = compute_keypoint_losses(predictions, batch, **self.nocs)
                keypoint_component_weights = self.nocs.get("keypoint_loss_weights", {})
                keypoint_loss = (
                    keypoint_component_weights.get("diversity", 1.0) * keypoint_loss_dict["loss_diversity"]
                    + keypoint_component_weights.get("cd", 1.0) * keypoint_loss_dict["loss_cd"]
                )
                keypoint_loss = keypoint_loss * self.nocs.get("keypoint_weight", 0.1)
                total_loss = total_loss + keypoint_loss
                loss_dict.update(keypoint_loss_dict)
                loss_dict["loss_keypoint_total"] = keypoint_loss

        # Pose loss - if pose is predicted
        if "abs_pose" in predictions:
            pose_config = self.pose or {}
            pose_weight = pose_config.get("weight", 1.0)
            pose_loss_dict = compute_pose_loss(predictions, batch, **pose_config)
            pose_loss = pose_loss_dict["loss_abs_pose"] * pose_weight
            total_loss = total_loss + pose_loss
            loss_dict.update(pose_loss_dict)

        # Cycle consistency loss - absolute pose vs camera head relative (normalized space)
        if (self.cycle_consistency is not None) and ("abs_pose" in predictions) and ("pose_enc" in predictions):
            cc_config = self.cycle_consistency if isinstance(self.cycle_consistency, dict) else {}
            cc_weight = cc_config.get("weight", 1.0)
            cc_loss_dict = compute_cycle_consistency_loss(predictions, batch, **cc_config)
            cc_loss = cc_loss_dict["loss_cycle_consistency"] * cc_weight
            total_loss = total_loss + cc_loss
            loss_dict.update(cc_loss_dict)

        # Reconstruction loss - if reconstructed model is available
        if "recon_model" in predictions:
            recon_config = getattr(self, "recon", None) or {}
            recon_weight = recon_config.get("weight", 1.0)
            delta_weight = recon_config.get("delta_weight", 0.1)
            recon_loss_dict = compute_reconstruction_loss(predictions, batch, **recon_config)
            recon_loss = recon_loss_dict["loss_recon"] * recon_weight
            delta_loss = recon_loss_dict["loss_delta"] * delta_weight
            total_loss = total_loss + recon_loss + delta_loss
            loss_dict.update(recon_loss_dict)

        # Absolute scale loss - if absolute translation and size are predicted
        if (self.scale is not None) and ("abs_translation" in predictions) and ("abs_size" in predictions):
            scale_config = self.scale if isinstance(self.scale, dict) else {}
            t_weight = scale_config.get("t_weight", 1.0)
            s_weight = scale_config.get("s_weight", 1.0)
            loss_type = scale_config.get("loss_type", "l1")
            scale_weight = scale_config.get("weight", 1.0)
            
            scale_loss_dict = compute_absolute_scale_loss(
                predictions, batch,
                t_weight=t_weight,
                s_weight=s_weight,
                loss_type=loss_type
            )
            scale_loss = scale_loss_dict["loss_absolute_scale"] * scale_weight
            total_loss = total_loss + scale_loss
            loss_dict.update(scale_loss_dict)

        # Relative scale loss - if per-frame RGB-based scale is predicted
        if (self.rel_scale is not None) and ("log_avg_scale_per_frame" in predictions):
            rel_scale_config = self.rel_scale if isinstance(self.rel_scale, dict) else {}
            rel_scale_weight = rel_scale_config.get("weight", 1.0)
            metric_consistency_weight = rel_scale_config.get("metric_consistency_weight", 5.0)
            
            rel_scale_loss_dict = compute_rel_scale_loss(
                predictions, batch,
                metric_consistency_weight=metric_consistency_weight
            )
            rel_scale_loss = rel_scale_loss_dict["loss_rel_scale"] * rel_scale_weight
            total_loss = total_loss + rel_scale_loss
            loss_dict.update(rel_scale_loss_dict)

        # Tracking loss - not cleaned yet, dirty code is at the bottom of this file
        if "track" in predictions:
            raise NotImplementedError("Track loss is not cleaned up yet")
        
        # InfoNCE regularization for object embeddings
        if (self.infonce_cfg is not None) and ("z_obj" in predictions) and (self.infonce_weight > 0):
            z = predictions["z_obj"]
            B = None
            S = None
            if z.dim() == 3:
                B, S, D = z.shape
                z_flat = z.reshape(B * S, D)
            else:
                B = z.shape[0]
                S = 1
                z_flat = z.reshape(B, -1)
            z_norm_local = F.normalize(z_flat, dim=-1)

            gather_ctx = _gather_infonce_context(z_flat, predictions, batch)
            if gather_ctx is not None:
                z_for_loss = gather_ctx["z_for_loss"]
                mask_batch = gather_ctx["batch"]
                mask_predictions = gather_ctx["predictions"]
                anchor_mask = gather_ctx["anchor_mask"]
                pos_mask, softW = build_pos_mask_from_batch(mask_batch, mask_predictions)
                pos_mask = pos_mask.to(z_for_loss.device)
                if softW is not None:
                    softW = softW.to(z_for_loss.device)
                if anchor_mask is not None:
                    anchor_mask = anchor_mask.to(pos_mask.device)
                    pos_mask = pos_mask & anchor_mask[:, None]
                    if softW is not None:
                        softW = softW * anchor_mask[:, None].float()
            else:
                z_for_loss = z_flat
                pos_mask, softW = build_pos_mask_from_batch(batch, predictions)
                anchor_mask = None

            L_infonce = supcon_infonce(
                z=z_for_loss,
                pos_mask=pos_mask,
                temperature=self.infonce_tau,
                pos_weights=softW,
                positive_mode=self.infonce_pos_mode,
            )
            L_infonce = check_and_fix_inf_nan(L_infonce, "loss_infonce")
            total_loss = total_loss + self.infonce_weight * L_infonce
            loss_dict["loss_infonce"] = L_infonce

            with torch.no_grad():
                z_stats = z_for_loss
                z_stats_norm = F.normalize(z_stats, dim=-1)
                sim = z_stats_norm.detach() @ z_stats_norm.detach().t()
                N = sim.size(0)
                eye = torch.eye(N, device=sim.device, dtype=torch.bool)
                if anchor_mask is None:
                    anchor_rows = torch.ones(N, dtype=torch.bool, device=sim.device)
                else:
                    anchor_rows = anchor_mask.to(sim.device)
                pos_mask_rows = pos_mask & anchor_rows[:, None]
                if anchor_rows.any():
                    pos_counts = pos_mask_rows.float().sum(1)[anchor_rows]
                    pos_count = pos_counts.mean()
                else:
                    pos_count = sim.new_tensor(float("nan"))
                if pos_mask_rows.any():
                    pos_sim = sim[pos_mask_rows].mean()
                else:
                    pos_sim = sim.new_tensor(float("nan"))
                neg_mask = anchor_rows[:, None] & (~eye) & (~pos_mask)
                if neg_mask.any():
                    neg_sim = sim[neg_mask].mean()
                else:
                    neg_sim = sim.new_tensor(float("nan"))
                loss_dict["stat_infonce_pos_count"] = pos_count.detach()
                loss_dict["stat_infonce_pos_sim"] = pos_sim.detach()
                loss_dict["stat_infonce_neg_sim"] = neg_sim.detach()
                loss_dict["stat_infonce_gap"] = (pos_sim - neg_sim).detach()

            if self.instance_var_weight > 0:
                seq_id = batch.get("seq_id")
                seq_tensor = None
                if seq_id is not None:
                    if isinstance(seq_id, torch.Tensor):
                        seq_tensor = seq_id.to(z_flat.device).long()
                    else:
                        seq_tensor = torch.as_tensor(seq_id, dtype=torch.long, device=z_flat.device)
                    if seq_tensor.dim() == 0:
                        seq_tensor = seq_tensor.reshape(1)
                    if (
                        seq_tensor.dim() == 1
                        and (B is not None)
                        and (S is not None)
                        and seq_tensor.numel() == B
                        and S > 1
                    ):
                        seq_tensor = seq_tensor[:, None].expand(-1, S)
                    seq_tensor = seq_tensor.reshape(-1)
                    if seq_tensor.numel() != z_flat.size(0):
                        seq_tensor = None
                if seq_tensor is not None:
                    L_instance = instance_variance_loss(z_norm_local, seq_tensor)
                    L_instance = check_and_fix_inf_nan(L_instance, "loss_instance_consistency")
                    total_loss = total_loss + self.instance_var_weight * L_instance
                    loss_dict["loss_instance_consistency"] = L_instance

        loss_dict["objective"] = total_loss

        return loss_dict


def compute_camera_loss(
    pred_dict,              # predictions dict, contains pose encodings
    batch_data,             # ground truth and mask batch dict
    loss_type="l1",         # "l1" or "l2" loss
    gamma=0.6,              # temporal decay weight for multi-stage training
    pose_encoding_type="absT_quaR_FoV",
    weight_trans=1.0,       # weight for translation loss
    weight_rot=1.0,         # weight for rotation loss
    weight_focal=0.5,       # weight for focal length loss
    **kwargs
):
    # List of predicted pose encodings per stage
    pred_pose_encodings = pred_dict['pose_enc_list']
    # Binary mask for valid points per frame (B, N, H, W)
    point_masks = batch_data['point_masks']
    # Only consider frames with enough valid points (>100)
    valid_frame_mask = point_masks[:, 0].sum(dim=[-1, -2]) > 100
    # Number of prediction stages
    n_stages = len(pred_pose_encodings)

    # Get ground truth camera extrinsics and intrinsics
    gt_extrinsics = batch_data['extrinsics']
    gt_intrinsics = batch_data['intrinsics']
    image_hw = batch_data['images'].shape[-2:]

    # Encode ground truth pose to match predicted encoding format
    gt_pose_encoding = extri_intri_to_pose_encoding(
        gt_extrinsics, gt_intrinsics, image_hw, pose_encoding_type=pose_encoding_type
    )

    # Initialize loss accumulators for translation, rotation, focal length
    total_loss_T = total_loss_R = total_loss_FL = 0

    # Compute loss for each prediction stage with temporal weighting
    for stage_idx in range(n_stages):
        # Later stages get higher weight (gamma^0 = 1.0 for final stage)
        stage_weight = gamma ** (n_stages - stage_idx - 1)
        pred_pose_stage = pred_pose_encodings[stage_idx]

        if valid_frame_mask.sum() == 0:
            # If no valid frames, set losses to zero to avoid gradient issues
            loss_T_stage = (pred_pose_stage * 0).mean()
            loss_R_stage = (pred_pose_stage * 0).mean()
            loss_FL_stage = (pred_pose_stage * 0).mean()
        else:
            # Only consider valid frames for loss computation
            loss_T_stage, loss_R_stage, loss_FL_stage = camera_loss_single(
                pred_pose_stage[valid_frame_mask].clone(),
                gt_pose_encoding[valid_frame_mask].clone(),
                loss_type=loss_type
            )
        # Accumulate weighted losses across stages
        total_loss_T += loss_T_stage * stage_weight
        total_loss_R += loss_R_stage * stage_weight
        total_loss_FL += loss_FL_stage * stage_weight

    # Average over all stages
    avg_loss_T = total_loss_T / n_stages
    avg_loss_R = total_loss_R / n_stages
    avg_loss_FL = total_loss_FL / n_stages

    # Compute total weighted camera loss
    total_camera_loss = (
        avg_loss_T * weight_trans +
        avg_loss_R * weight_rot +
        avg_loss_FL * weight_focal
    )

    # Return loss dictionary with individual components
    return {
        "loss_camera": total_camera_loss,
        "loss_T": avg_loss_T,
        "loss_R": avg_loss_R,
        "loss_FL": avg_loss_FL
    }

def camera_loss_single(pred_pose_enc, gt_pose_enc, loss_type="l1"):
    """
    Computes translation, rotation, and focal loss for a batch of pose encodings.
    
    Args:
        pred_pose_enc: (N, D) predicted pose encoding
        gt_pose_enc: (N, D) ground truth pose encoding
        loss_type: "l1" (abs error) or "l2" (euclidean error)
    Returns:
        loss_T: translation loss (mean)
        loss_R: rotation loss (mean)
        loss_FL: focal length/intrinsics loss (mean)
    
    NOTE: The paper uses smooth l1 loss, but we found l1 loss is more stable than smooth l1 and l2 loss.
        So here we use l1 loss.
    """
    if loss_type == "l1":
        # Translation: first 3 dims; Rotation: next 4 (quaternion); Focal/Intrinsics: last dims
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).abs()
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).abs()
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).abs()
    elif loss_type == "l2":
        # L2 norm for each component
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).norm(dim=-1, keepdim=True)
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).norm(dim=-1)
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).norm(dim=-1)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Check/fix numerical issues (nan/inf) for each loss component
    loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
    loss_R = check_and_fix_inf_nan(loss_R, "loss_R")
    loss_FL = check_and_fix_inf_nan(loss_FL, "loss_FL")

    # Clamp outlier translation loss to prevent instability, then average
    loss_T = loss_T.clamp(max=100).mean()
    loss_R = loss_R.mean()
    loss_FL = loss_FL.mean()

    return loss_T, loss_R, loss_FL



def compute_point_loss(predictions, batch, gamma=1.0, alpha=0.2, gradient_loss_fn = None, valid_range=-1, **kwargs):
    """
    Compute point loss.
    
    Args:
        predictions: Dict containing 'world_points' and 'world_points_conf'
        batch: Dict containing ground truth 'world_points' and 'point_masks'
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        gradient_loss_fn: Type of gradient loss to apply
        valid_range: Quantile range for outlier filtering
    """
    pred_points = predictions['world_points']  # [B, S, H, W, 3]
    pred_points_conf = predictions['world_points_conf']
    gt_points = batch['world_points']
    gt_points_mask = batch['point_masks']
    
    gt_points = check_and_fix_inf_nan(gt_points, "gt_points")
    
    if gt_points_mask.sum() < 100:
        # If there are less than 100 valid points, skip this batch
        dummy_loss = (0.0 * pred_points).mean()
        loss_dict = {f"loss_conf_point": dummy_loss,
                    f"loss_reg_point": dummy_loss,
                    f"loss_grad_point": dummy_loss,}
        return loss_dict
    
    # ===== Apply symmetry canonicalization to predicted world_points =====
    # if "sym_codes" in batch:
    #     sym_codes = batch["sym_codes"]
    #     # Convert to tensor if needed
    #     if isinstance(sym_codes, (list, tuple)):
    #         sym_codes = torch.tensor(sym_codes, device=pred_points.device, dtype=torch.long)
    #     elif not isinstance(sym_codes, torch.Tensor):
    #         sym_codes = torch.as_tensor(sym_codes, device=pred_points.device, dtype=torch.long)
        
    #     B, S, H, W, C = pred_points.shape
        
    #     # Get symmetry transform matrix S [B, S, 3, 3]
    #     dummy_R = torch.eye(3, device=pred_points.device).unsqueeze(0).unsqueeze(0).expand(B, S, 3, 3)
    #     _, S_total = apply_symmetry_to_rotation_batch(dummy_R, sym_codes)
        
    #     # Apply symmetry efficiently: [B, S, H, W, 3] @ [B, S, 3, 3]
    #     # Reshape pred_points to [B, S, H*W, 3]
    #     pred_points_reshaped = pred_points.view(B, S, H * W, 3)
        
    #     # Apply transformation: [B, S, H*W, 3] @ [B, S, 3, 3]
    #     pred_points_sym = torch.matmul(pred_points_reshaped, S_total)
        
    #     # Reshape back to [B, S, H, W, 3]
    #     pred_points = pred_points_sym.view(B, S, H, W, 3)
    # ===================================================================
    
    # Compute confidence-weighted regression loss with optional gradient loss
    loss_conf, loss_grad, loss_reg = regression_loss(pred_points, gt_points, gt_points_mask, conf=pred_points_conf,
                                             gradient_loss_fn=gradient_loss_fn, gamma=gamma, alpha=alpha, valid_range=valid_range)
    
    loss_dict = {
        f"loss_conf_point": loss_conf,
        f"loss_reg_point": loss_reg,
        f"loss_grad_point": loss_grad,
    }
    
    return loss_dict


def compute_depth_loss(predictions, batch, gamma=1.0, alpha=0.2, gradient_loss_fn = None, valid_range=-1, **kwargs):
    """
    Compute depth loss.
    
    Args:
        predictions: Dict containing 'depth' and 'depth_conf'
        batch: Dict containing ground truth 'depths' and 'point_masks'
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        gradient_loss_fn: Type of gradient loss to apply
        valid_range: Quantile range for outlier filtering
    """
    pred_depth = predictions['depth']
    pred_depth_conf = predictions['depth_conf']

    gt_depth = batch['depths']
    gt_depth = check_and_fix_inf_nan(gt_depth, "gt_depth")
    gt_depth = gt_depth[..., None]              # (B, H, W, 1)
    gt_depth_mask = batch['point_masks'].clone()   # 3D points derived from depth map, so we use the same mask

    if gt_depth_mask.sum() < 100:
        # If there are less than 100 valid points, skip this batch
        dummy_loss = (0.0 * pred_depth).mean()
        loss_dict = {f"loss_conf_depth": dummy_loss,
                    f"loss_reg_depth": dummy_loss,
                    f"loss_grad_depth": dummy_loss,}
        return loss_dict

    # NOTE: we put conf inside regression_loss so that we can also apply conf loss to the gradient loss in a multi-scale manner
    # this is hacky, but very easier to implement
    loss_conf, loss_grad, loss_reg = regression_loss(pred_depth, gt_depth, gt_depth_mask, conf=pred_depth_conf,
                                             gradient_loss_fn=gradient_loss_fn, gamma=gamma, alpha=alpha, valid_range=valid_range)

    loss_dict = {
        f"loss_conf_depth": loss_conf,
        f"loss_reg_depth": loss_reg,    
        f"loss_grad_depth": loss_grad,
    }

    return loss_dict


def compute_absolute_scale_loss(predictions, batch, t_weight=1.0, s_weight=1.0, loss_type='l1', **kwargs):
    """
    Absolute scale loss for AbsoluteScaleHead.
    
    Instead of directly supervising scale, we supervise absolute translation and size predictions.
    Scale is then derived implicitly by comparing absolute and normalized predictions.
    
    Requires:
      - predictions["abs_translation"]: [B, S, 3] predicted absolute translation
      - predictions["abs_size"]: [B, S, 3] predicted absolute size
      - batch["abs_extrinsics"]: [B, S, 4, 4] ground truth poses (for translation)
      - batch["sizes"]: [B, S, 3] ground truth object sizes
    
    Returns:
        Dict containing individual loss components and total loss
    """
    if "abs_translation" not in predictions or "abs_size" not in predictions:
        # If predictions are missing, return zero loss
        anchor_src = next((predictions[x] for x in ("nocs", "world_points", "depth") if x in predictions), None)
        anchor = (anchor_src * 0).mean() if anchor_src is not None else torch.tensor(0.0, device="cuda" if torch.cuda.is_available() else "cpu")
        return {
            "loss_absolute_scale": anchor,
            "loss_abs_translation": anchor,
            "loss_abs_size": anchor,
        }
    
    pred_t = predictions["abs_translation"]  # [B, S, 3]
    pred_s = predictions["abs_size"]          # [B, S, 3]
    
    # Get ground truth translation and size
    if "abs_extrinsics" not in batch:
        zero = (pred_t * 0).mean()
        return {
            "loss_absolute_scale": zero,
            "loss_abs_translation": zero,
            "loss_abs_size": zero,
        }
    
    gt_extrinsics = batch["abs_extrinsics"]  # [B, S, 4, 4]
    gt_t = gt_extrinsics[:, :, :3, 3]         # [B, S, 3]
    gt_s = batch["sizes"]                     # [B, S, 3]
    
    # Compute losses
    if loss_type == 'l1':
        loss_t = F.l1_loss(pred_t, gt_t)
        loss_s = F.l1_loss(pred_s, gt_s)
    elif loss_type == 'l2':
        loss_t = F.mse_loss(pred_t, gt_t)
        loss_s = F.mse_loss(pred_s, gt_s)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")
    
    loss_t = check_and_fix_inf_nan(loss_t, "loss_abs_translation")
    loss_s = check_and_fix_inf_nan(loss_s, "loss_abs_size")
    
    # Total weighted loss
    total_loss = t_weight * loss_t + s_weight * loss_s
    total_loss = check_and_fix_inf_nan(total_loss, "loss_absolute_scale")
    
    return {
        "loss_absolute_scale": total_loss,
        "loss_abs_translation": loss_t,
        "loss_abs_size": loss_s,
    }


def compute_rel_scale_loss(predictions, batch, eps=1e-8, smooth_l1_beta=0.1, **kwargs):
    """
    Avg scale supervision for RelScaleHead (RGB-based scale estimation).
    
    Supervises log avg_scale prediction using Smooth L1 loss in log space.
    Smooth L1 balances precision (MSE for small errors) and robustness (L1 for outliers),
    which is important for RGB-based scale estimation that may have noisy observations.
    
    Requires:
      - predictions["log_avg_scale_per_frame"] : [B,S] predicted log scale per frame
      - batch["avg_scale"]                      : [B] or [B,S] ground truth scale
    
    Args:
      - smooth_l1_beta: Threshold for Smooth L1 loss (default 0.1 ≈ 10% scale error)
    
    Returns dict with loss_rel_scale and loss_rel_scale_logspace.
    """
    if "log_avg_scale_per_frame" not in predictions:
        # If predictions are missing, return zero loss
        anchor_src = next((predictions[x] for x in ("nocs", "world_points", "depth") if x in predictions), None)
        anchor = (anchor_src * 0).mean() if anchor_src is not None else torch.tensor(0.0, device="cuda" if torch.cuda.is_available() else "cpu")
        return {
            "loss_rel_scale": anchor,
            "loss_rel_scale_logspace": anchor,
        }
    
    # Predicted per-frame log avg-scale
    log_avg_hat_frames = predictions["log_avg_scale_per_frame"]  # [B,S]
    
    if ("avg_scale" not in batch) or (batch["avg_scale"] is None):
        zero = check_and_fix_inf_nan(log_avg_hat_frames.new_tensor(0.0), "loss_rel_scale_avg_missing")
        return {
            "loss_rel_scale": zero,
            "loss_rel_scale_logspace": zero,
        }
    
    # Ground-truth sequence avg scale
    avg_scale = batch["avg_scale"]
    
    # === Stage 1: Supervise scale prediction in log space ===
    # Handle both [B] and [B,S] formats
    if isinstance(avg_scale, torch.Tensor):
        if avg_scale.dim() == 1:  # [B]
            log_avg_gt = torch.log(avg_scale.clamp_min(eps))  # [B]
            target_per_frame = log_avg_gt.unsqueeze(1).expand_as(log_avg_hat_frames)  # [B,S]
        else:  # [B,S]
            log_avg_gt = torch.log(avg_scale.clamp_min(eps))
            target_per_frame = log_avg_gt
    else:
        # Handle scalar or list
        if isinstance(avg_scale, (int, float)):
            avg_scale = torch.tensor([avg_scale], device=log_avg_hat_frames.device)
        else:
            avg_scale = torch.tensor(avg_scale, device=log_avg_hat_frames.device)
        log_avg_gt = torch.log(avg_scale.clamp_min(eps))
        if avg_scale.numel() == 1:
            target_per_frame = log_avg_gt.expand_as(log_avg_hat_frames)
        else:
            target_per_frame = log_avg_gt.unsqueeze(1).expand_as(log_avg_hat_frames)
    
    target_per_frame = check_and_fix_inf_nan(target_per_frame, "loss_rel_scale_avg_gt")
    
    # Valid mask for finite predictions
    valid_mask = torch.isfinite(log_avg_hat_frames)
    
    if not torch.any(valid_mask):
        zero = check_and_fix_inf_nan(log_avg_hat_frames.new_tensor(0.0), "loss_rel_scale_avg_all_invalid")
        return {
            "loss_rel_scale": zero,
            "loss_rel_scale_logspace": zero,
        }
    
    # Replace invalid predictions with target to avoid NaNs
    log_avg_hat_clean = torch.where(valid_mask, log_avg_hat_frames, target_per_frame)
    
    # Use Smooth L1 for robustness to outliers while maintaining precision
    # Smooth L1 = MSE for small errors (precise convergence) + L1 for large errors (robust to outliers)
    # This is important for RGB-based scale estimation which may have noisy observations from
    # occlusions, motion blur, or poor lighting conditions
    per_frame_loss = F.smooth_l1_loss(
        log_avg_hat_clean,
        target_per_frame,
        beta=smooth_l1_beta,  # Treat errors > beta in log space as outliers (default 0.1 ≈ 10% scale error)
        reduction="none",
    )
    
    per_frame_loss = per_frame_loss * valid_mask.float()
    valid_counts = valid_mask.float().sum(dim=1).clamp_min(1.0)
    loss_per_sequence = per_frame_loss.sum(dim=1) / valid_counts
    loss_rel_scale_logspace = loss_per_sequence.mean()
    loss_rel_scale_logspace = check_and_fix_inf_nan(loss_rel_scale_logspace, "loss_rel_scale_logspace")
    
    # Return only the log-space scale loss (no metric consistency to avoid interfering with AbsoluteScaleHead)
    return {
        "loss_rel_scale": loss_rel_scale_logspace,
        "loss_rel_scale_logspace": loss_rel_scale_logspace,
    }


def _geodesic_rotation_loss(R_pred, R_gt):
    """
    Geodesic distance on SO(3): angle = arccos((tr(R_pred^T @ R_gt) - 1) / 2).
    
    This is the proper metric on the rotation manifold, unlike Frobenius norm
    which doesn't correspond to actual rotation angle error.
    
    Args:
        R_pred: [B, S, 3, 3] predicted rotation matrices
        R_gt: [B, S, 3, 3] ground truth rotation matrices
    Returns:
        Scalar mean geodesic loss (in radians)
    """
    # R_diff = R_pred^T @ R_gt  →  [B, S, 3, 3]
    R_diff = R_pred.transpose(-1, -2) @ R_gt
    # trace = sum of diagonal elements → [B, S]
    trace = R_diff.diagonal(dim1=-2, dim2=-1).sum(-1)
    # cos(angle) = (trace - 1) / 2, clamp for numerical safety
    cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    angle = torch.acos(cos_angle)  # [B, S] in radians
    return angle.mean()


def compute_cycle_consistency_loss(predictions, batch, **kwargs):
    """Cycle consistency between absolute pose (PoseNetHead) and relative pose (camera head).

    In normalized space (scheme 1): compare relative transforms from abs_pose with those
    implied by pose_enc extrinsics, without applying avg_scale to absolute translation.

    T_rel^{a→q} from abs: T_abs_q @ inv(T_abs_a). From camera head: T_cam_q @ inv(T_cam_a).
    Loss = geodesic(R_rel_abs, R_rel_cam) + weight_t * L2(t_rel_abs, t_rel_cam).

    Uses consecutive frame pairs (0,1), (1,2), ... when S >= 2.
    """
    rot_weight = kwargs.get("rotation_weight", 1.0)
    trans_weight = kwargs.get("translation_weight", 0.1)
    pair_type = kwargs.get("pair_type", "consecutive")  # "consecutive" or "all"

    if "abs_pose" not in predictions or "pose_enc" not in predictions:
        anchor_src = predictions.get("pose_enc") or predictions.get("abs_pose")
        if anchor_src is not None and isinstance(anchor_src, dict):
            anchor_src = anchor_src.get("rotation")
        if anchor_src is not None and isinstance(anchor_src, torch.Tensor):
            anchor = (anchor_src * 0).mean()
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            anchor = torch.tensor(0.0, device=device)
        return {"loss_cycle_consistency": anchor}

    pose = predictions["abs_pose"]
    R_abs = pose["rotation"]       # [B, S, 3, 3] canonical → camera
    t_abs = pose["translation"]    # [B, S, 3] normalized (do not apply avg_scale)
    pose_enc = predictions["pose_enc"]  # [B, S, 9]

    B, S = R_abs.shape[:2]
    if S < 2:
        device = R_abs.device
        return {"loss_cycle_consistency": R_abs.new_tensor(0.0)}

    image_hw = batch["images"].shape[-2:]
    extrinsics, _ = pose_encoding_to_extri_intri(
        pose_enc, image_hw, build_intrinsics=False
    )  # [B, S, 3, 4], camera-from-world [R|t]
    R_cam = extrinsics[..., :3, :3]   # [B, S, 3, 3]
    t_cam = extrinsics[..., :3, 3]    # [B, S, 3]

    if pair_type == "consecutive":
        pairs = [(i, i + 1) for i in range(S - 1)]
    else:
        pairs = [(a, q) for a in range(S) for q in range(S) if a < q]

    loss_R_list = []
    loss_t_list = []
    for a, q in pairs:
        # From absolute pose: T_rel = T_abs_q @ inv(T_abs_a)  (camera_a → camera_q)
        R_rel_abs = R_abs[:, q] @ R_abs[:, a].transpose(-1, -2)   # [B, 3, 3]
        t_rel_abs = t_abs[:, q] - (R_rel_abs @ t_abs[:, a].unsqueeze(-1)).squeeze(-1)  # [B, 3]
        # From camera head (same formula: E is camera-from-world)
        R_rel_cam = R_cam[:, q] @ R_cam[:, a].transpose(-1, -2)   # [B, 3, 3]
        t_rel_cam = t_cam[:, q] - (R_rel_cam @ t_cam[:, a].unsqueeze(-1)).squeeze(-1)  # [B, 3]
        loss_R_list.append(_geodesic_rotation_loss(
            R_rel_abs.unsqueeze(1), R_rel_cam.unsqueeze(1)
        ))
        loss_t_list.append(torch.mean(torch.norm(t_rel_abs - t_rel_cam, dim=-1)))

    loss_R = torch.stack(loss_R_list).mean()
    loss_t = torch.stack(loss_t_list).mean()
    total = rot_weight * loss_R + trans_weight * loss_t
    return {
        "loss_cycle_consistency": check_and_fix_inf_nan(total, "loss_cycle_consistency"),
        "loss_cycle_R": check_and_fix_inf_nan(loss_R, "loss_cycle_R"),
        "loss_cycle_t": check_and_fix_inf_nan(loss_t, "loss_cycle_t"),
    }


def compute_pose_loss(predictions, batch, **kwargs):
    """Multi-frame pose loss for PoseNetHead with symmetry awareness.

    Uses rotation supervision from all frame extrinsics for absolute pose estimation.

    For symmetric objects (sym_codes available), uses min-over-equivalents distance
    instead of canonicalization.  This avoids the torch.round discontinuity that
    caused loss_pose_R oscillation on Omni6D.

    Both geodesic and Frobenius reduce to the same trace optimization:
        min_{S_k in Sym} d(R_pred, R_gt @ S_k)
    where the trace tr(R_pred^T @ R_gt @ S_k) is shared, and only the final
    trace-to-distance conversion differs.

    Supports two rotation loss types:
        - "geodesic":   arccos((tr-1)/2)       — proper SO(3) metric (recommended)
        - "frobenius":  sqrt(6 - 2*tr)         — ||R_pred - R_gt||_F

    When ``multi_view_pose_aggregation`` is True the predicted pose is a single
    canonical -> frame-0 transform broadcast to [B, S, ...].  Supervision is
    performed against frame-0 GT only.
    """
    rot_component_weight = kwargs.get("rotation_weight", 1.0)
    trans_component_weight = kwargs.get("translation_weight", 0.1)
    size_component_weight = kwargs.get("size_weight", 0.05)
    rotation_loss_type = kwargs.get("rotation_loss_type", "geodesic")
    soft_min_temperature = kwargs.get("soft_min_temperature", 0.05)
    multi_view = kwargs.get("multi_view_pose_aggregation", False)

    def _apply_avg_scale(tensor: torch.Tensor) -> torch.Tensor:
        """Scale tensor by batch avg_scale if provided."""
        avg_scale = batch.get("avg_scale", None)
        if avg_scale is None:
            return tensor
        if isinstance(avg_scale, torch.Tensor):
            avg_scale = avg_scale.to(tensor.device)
            if avg_scale.numel() == 1:
                return tensor * avg_scale
            if avg_scale.dim() < tensor.dim():
                view_shape = avg_scale.shape + (1,) * (tensor.dim() - avg_scale.dim())
                avg_scale = avg_scale.reshape(view_shape)
            return tensor * avg_scale
        return tensor * avg_scale

    # Guard: required predictions
    if "abs_pose" not in predictions:
        anchor_src = next((predictions[x] for x in ("nocs", "world_points", "depth") if x in predictions), None)
        anchor = (anchor_src * 0).mean() if anchor_src is not None else torch.tensor(0.0, device="cuda" if torch.cuda.is_available() else "cpu")
        return {"loss_abs_pose": anchor}

    pose = predictions["abs_pose"]
    rotation = pose["rotation"]    # [B,S,3,3] predicted rotations per frame
    translation = pose["translation"]  # [B,S,3] predicted translations per frame
    scale = pose.get("scale", None)  # [B,S,1] optional isotropic scale
    size = pose.get("size", None)    # [B,S,3] optional anisotropic size

    # ===== Extract GT =====
    gt_extrinsics = batch["abs_extrinsics"]  # [B,S,3,4] pre-canonicalized in data loader
    gt_R = gt_extrinsics[:, :, :3, :3]  # [B,S,3,3]
    gt_t = gt_extrinsics[:, :, :3, 3]   # [B,S,3]
    gt_s = batch["sizes"]  # [B,S,3]

    # When multi-view aggregation is active the prediction is a single pose
    # (canonical -> frame 0) broadcast to S frames.  Supervise frame 0 only.
    # NOTE: abs_extrinsics[:, s] is canonical -> camera_s, which DIFFERS per
    # frame.  The pointmap-based kpt_3d are all in frame-0 space, so the
    # predicted transform is canonical -> frame_0 = abs_extrinsics[:, 0].
    # Using any other frame's GT would create contradictory gradients.
    if multi_view and rotation.shape[1] > 1:
        rotation = rotation[:, 0:1]
        translation = translation[:, 0:1]
        size = size[:, 0:1] if size is not None else None
        gt_R = gt_R[:, 0:1]
        gt_t = gt_t[:, 0:1]
        gt_s = gt_s[:, 0:1]

    # ===== Symmetry-aware rotation loss =====
    if "sym_codes" in batch:
        sym_codes = batch["sym_codes"]
        if isinstance(sym_codes, (list, tuple)):
            sym_codes = torch.tensor(sym_codes, device=rotation.device, dtype=torch.long)
        elif not isinstance(sym_codes, torch.Tensor):
            sym_codes = torch.as_tensor(sym_codes, device=rotation.device, dtype=torch.long)

        # Slice sym_codes to match the (possibly sliced) S dimension
        if sym_codes.dim() == 3 and sym_codes.shape[1] != rotation.shape[1]:
            sym_codes = sym_codes[:, 0:1]

        loss_R = symmetry_aware_rotation_loss(
            rotation, gt_R, sym_codes,
            loss_type=rotation_loss_type,
            soft_min_temperature=soft_min_temperature,
        )
    elif rotation_loss_type == "geodesic":
        loss_R = _geodesic_rotation_loss(rotation, gt_R)
    else:
        loss_R = torch.mean(torch.norm(rotation - gt_R, dim=(2, 3)))

    # Scale predicted translation by avg_scale if available
    pred_t_scaled = _apply_avg_scale(translation)
    pred_s_scaled = _apply_avg_scale(size)

    loss_t = torch.mean(torch.norm(pred_t_scaled - gt_t, dim=-1))
    loss_s = torch.mean(torch.norm(pred_s_scaled - gt_s, dim=-1))

    total_pose_loss = rot_component_weight * loss_R + trans_component_weight * loss_t + size_component_weight * loss_s
    return {
        "loss_abs_pose": check_and_fix_inf_nan(total_pose_loss, "loss_pose_total"),
        "loss_pose_R": check_and_fix_inf_nan(loss_R, "loss_pose_R"),
        "loss_pose_t": check_and_fix_inf_nan(loss_t, "loss_pose_t"),
        "loss_pose_s": check_and_fix_inf_nan(loss_s, "loss_pose_s"),
    }


def SmoothL1Dis(p1, p2, threshold=0.1):
    """
    Smooth L1 distance for sparse points.
    
    Args:
        p1: [B, N, 3] predicted sparse points  
        p2: [B, N, 3] ground truth sparse points
        threshold: Smooth L1 threshold
    
    Returns:
        Scalar smooth L1 distance loss
    """
    diff = torch.abs(p1 - p2)
    less = torch.pow(diff, 2) / (2.0 * threshold)
    higher = diff - threshold / 2.0
    dis = torch.where(diff > threshold, higher, less)
    dis = torch.mean(torch.sum(dis, dim=-1))  # Sum over 3D coords, mean over batch and points
    return dis


def _smooth_l1_per_point(p1: torch.Tensor, p2: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
    """Per-point smooth L1 distance (returns loss per point before mean)."""
    diff = torch.abs(p1 - p2)
    less = torch.pow(diff, 2) / (2.0 * threshold)
    higher = diff - threshold / 2.0
    dis = torch.where(diff > threshold, higher, less)
    return dis.sum(dim=-1)


def ChamferDis(p1, p2):
    """
    Chamfer distance for point clouds.
    
    Args:
        p1: [B, N1, 3] first point cloud
        p2: [B, N2, 3] second point cloud
    
    Returns:
        Scalar chamfer distance
    """
    # Compute pairwise distances: [B, N1, N2]
    dis = torch.norm(p1.unsqueeze(2) - p2.unsqueeze(1), dim=3)
    
    # Forward distance: min distance from p1 to p2
    dis1 = torch.min(dis, 2)[0]  # [B, N1]
    
    # Backward distance: min distance from p2 to p1
    dis2 = torch.min(dis, 1)[0]  # [B, N2]
    
    # Symmetric chamfer distance
    dis = 0.5 * dis1.mean(1) + 0.5 * dis2.mean(1)  # [B]
    return dis.mean()


def cd_dis_k2p(pts, pred_kpt_3d, use_smooth=True, smooth_threshold=0.05):
    """
    Compute average distance from each keypoint to its nearest point.
    
    This encourages keypoints to cover the sampled point cloud well.
    
    Args:
        pts: [BS, K, 3] sampled point cloud
        pred_kpt_3d: [BS, kpt_num, 3] predicted keypoints
        use_smooth: Apply smooth L1 loss instead of L2 for better gradient stability
        smooth_threshold: Threshold for smooth L1 loss
    
    Returns:
        Scalar loss - average distance from keypoints to nearest points
    """
    if pts.shape[1] == 0:
        return pts.new_zeros(())

    # Compute pairwise distances: [BS, K, kpt_num]
    dis = torch.norm(pts.unsqueeze(2) - pred_kpt_3d.unsqueeze(1), dim=3)
    
    # Find minimum distance from each keypoint to any point
    dis_min = torch.min(dis, dim=1)[0]  # [BS, kpt_num]
    
    if use_smooth:
        # Smooth L1 loss for better gradient behavior
        # For distances < threshold: 0.5 * d^2 / threshold
        # For distances >= threshold: d - 0.5 * threshold
        mask = dis_min < smooth_threshold
        loss_smooth = torch.where(
            mask,
            0.5 * dis_min.pow(2) / smooth_threshold,
            dis_min - 0.5 * smooth_threshold
        )
        return loss_smooth.mean()
    else:
        return dis_min.mean()


def diversity_loss_3d(data, threshold=0.02, loss_type="exponential", scale_factor=1.0):
    """
    Diversity loss to encourage keypoints to be spread out.
    
    Uses exponential decay to provide non-saturating gradients that continue
    encouraging keypoints to spread even beyond the threshold distance.
    
    Args:
        data: [B, kpt_num, 3] keypoint coordinates
        threshold: Characteristic distance scale for diversity (not a hard cutoff)
        loss_type: "exponential" (default, non-saturating) or "linear" (original)
        scale_factor: Adaptive scale multiplier based on object size
    
    Returns:
        Scalar diversity loss
    """
    b, kpt_num = data.shape[0], data.shape[1]
    
    # Compute pairwise distances between keypoints
    dis_mat = data.unsqueeze(2) - data.unsqueeze(1)  # [B, kpt_num, kpt_num, 3]
    dis_mat = torch.norm(dis_mat, p=2, dim=3, keepdim=False)  # [B, kpt_num, kpt_num]
    
    # Mask out diagonal (self-distances)
    eye_mask = torch.eye(kpt_num, device=dis_mat.device, dtype=torch.bool).unsqueeze(0)
    
    if loss_type == "exponential":
        # Exponential penalty: exp(-distance/threshold)
        # Provides non-saturating gradients that continue encouraging spread
        # Adaptive threshold based on object scale
        adaptive_threshold = threshold * scale_factor
        loss_mat = torch.exp(-dis_mat / adaptive_threshold)
        
        # Zero out diagonal
        loss_mat = loss_mat.masked_fill(eye_mask, 0.0)
        
        # Sum and normalize
        loss = torch.sum(loss_mat, dim=[1, 2])  # [B]
        loss = loss / (kpt_num * (kpt_num - 1))
        
    else:  # linear (original implementation)
        # Add identity to avoid self-distance penalty
        dis_mat = dis_mat + torch.eye(kpt_num, device=dis_mat.device).unsqueeze(0)
        
        # Clamp distances to threshold (causes saturation)
        dis_mat = torch.clamp(dis_mat, max=threshold)
        
        # Convert distance to loss: dis=0 -> loss=1, dis=threshold -> loss=0
        loss_mat = 1.0 - dis_mat / threshold
        
        # Sum and normalize
        loss = torch.sum(loss_mat, dim=[1, 2])  # [B]
        loss = loss / (kpt_num * (kpt_num - 1))
    
    return loss.mean()


def compute_keypoint_losses(predictions, batch, diversity_threshold=0.02, 
                           diversity_loss_type="exponential", 
                           use_adaptive_threshold=True, **kwargs):
    """
    Compute keypoint-specific losses: diversity and keypoint-to-point distance.
    
    Args:
        predictions: Dict containing 'pred_kpt_3d' [B,S,kpt_num,3] keypoint coordinates
            and optionally 'pts_3d_cam' [B,S,K,3] sampled camera-frame points
        batch: Dict containing 'cam_points' [B,S,H,W,3] (camera-frame dense points)
            and 'choose_indices' [B,S,K_dense]; falls back to 'world_points' if
            camera-frame points are unavailable when predictions do not provide
            sampled camera-frame points
        diversity_threshold: Base threshold for diversity loss (characteristic scale)
        diversity_loss_type: "exponential" (non-saturating) or "linear" (original)
        use_adaptive_threshold: Scale threshold based on object size if True
    
    Returns:
        Dict containing keypoint-specific losses and diagnostic statistics
    """
    pred_kpt_3d = predictions["pred_kpt_3d"]  # [B,S,kpt_num,3]
    B, S, kpt_num = pred_kpt_3d.shape[:3]
    BS = B * S
    pred_kpt_3d_flat = pred_kpt_3d.view(BS, kpt_num, 3)  # [BS, kpt_num, 3]
    
    # Compute scale factor for adaptive threshold
    scale_factor = 1.0
    if use_adaptive_threshold and "sizes" in batch:
        # Use object size to adapt the diversity threshold
        sizes = batch["sizes"]  # [B, S, 3]
        if sizes.dim() >= 2:
            # Average dimension of the object as characteristic scale
            obj_scale = sizes.view(BS, -1).mean(dim=-1, keepdim=True)  # [BS, 1]
            # Scale factor: larger objects should have proportionally larger thresholds
            # Use factor of 0.1 to convert from object scale to reasonable keypoint spread
            scale_factor = (obj_scale / 0.2).clamp(min=0.5, max=5.0)  # [BS, 1]
            scale_factor = scale_factor.mean().item()
    
    # Diversity loss with adaptive threshold
    loss_diversity = diversity_loss_3d(
        pred_kpt_3d_flat, 
        threshold=diversity_threshold,
        loss_type=diversity_loss_type,
        scale_factor=scale_factor
    )

    # Keypoint-to-point distance loss (coverage loss)
    loss_cd = torch.tensor(0.0, device=pred_kpt_3d.device)
    
    if "pts_3d_cam" in predictions:
        pts_3d_cam = predictions["pts_3d_cam"]  # [B,S,K,3]
        if pts_3d_cam.dim() == 4:
            pts_3d_cam = pts_3d_cam.view(BS, pts_3d_cam.shape[2], 3)
        
        # Use smooth L1 loss for better gradient stability
        cd_smooth_threshold = kwargs.get("cd_smooth_threshold", 0.05)
        loss_cd = cd_dis_k2p(pts_3d_cam, pred_kpt_3d_flat, 
                            use_smooth=True, 
                            smooth_threshold=cd_smooth_threshold)
    
    # Compute diagnostic statistics
    with torch.no_grad():
        # Average pairwise distance between keypoints
        dis_mat = pred_kpt_3d_flat.unsqueeze(2) - pred_kpt_3d_flat.unsqueeze(1)
        dis_mat = torch.norm(dis_mat, p=2, dim=3)
        
        # Exclude diagonal
        eye_mask = torch.eye(kpt_num, device=dis_mat.device, dtype=torch.bool).unsqueeze(0)
        dis_mat_no_diag = dis_mat.masked_fill(eye_mask, float('inf'))
        
        # Statistics
        avg_kpt_distance = dis_mat_no_diag[~eye_mask.expand_as(dis_mat_no_diag)].mean()
        min_kpt_distance = dis_mat_no_diag.min(dim=2)[0].mean()  # Average minimum distance per keypoint
        max_kpt_distance = dis_mat.max()
        
        # Keypoint spread (standard deviation)
        kpt_std = pred_kpt_3d_flat.std(dim=1).mean()
    
    return {
        "loss_diversity": check_and_fix_inf_nan(loss_diversity, "loss_diversity"),
        "loss_cd": check_and_fix_inf_nan(loss_cd, "loss_cd"),
        # Diagnostic statistics
        "stat_kpt_avg_distance": avg_kpt_distance,
        "stat_kpt_min_distance": min_kpt_distance,
        "stat_kpt_max_distance": max_kpt_distance,
        "stat_kpt_spread": kpt_std,
        "stat_kpt_scale_factor": torch.tensor(scale_factor, device=pred_kpt_3d.device),
    }


def compute_reconstruction_loss(predictions, batch, use_mask=False, **kwargs):
    """
    Compute reconstruction loss for point cloud reconstruction from keypoints.
    
    Uses Chamfer distance between reconstructed point cloud and input point cloud.
    Also includes a regularization term for reconstruction deltas.
    
    Args:
        predictions: Dict containing:
            - 'recon_model': [B, S, 3, pts_per_kpt*kpt_num] reconstructed point cloud
            - 'recon_delta': [B, S, pts_per_kpt*kpt_num, 3] reconstruction offsets
        batch: Dict containing ground truth data (optional masks)
        use_mask: Whether to use point cloud mask for valid points
    
    Returns:
        Dict containing reconstruction loss components
    """
    if "recon_model" not in predictions:
        # If predictions are missing, return zero loss
        anchor_src = next((predictions[x] for x in ("nocs", "world_points", "depth") if x in predictions), None)
        anchor = (anchor_src * 0).mean() if anchor_src is not None else torch.tensor(0.0, device="cuda" if torch.cuda.is_available() else "cpu")
        return {
            "loss_recon": anchor,
            "loss_delta": anchor,
        }
    
    recon_model = predictions["recon_model"]  # [B, S, 3, pts_per_kpt*kpt_num]
    recon_delta = predictions["recon_delta"]  # [B, S, pts_per_kpt*kpt_num, 3]
    
    # Get input point cloud in frame-0 (pointmap) space to match recon_model,
    # which is reconstructed from kpt_3d + center_pm (both in pointmap space).
    if "pts_3d_pm" in predictions:
        pts_input = predictions["pts_3d_pm"]  # [B, S, K, 3]
    elif "pts_3d_cam" in predictions:
        pts_input = predictions["pts_3d_cam"]  # [B, S, K, 3] fallback (frame-0 only correct)
    else:
        # Fallback: return zero loss if input points not available
        anchor = recon_model.new_tensor(0.0)
        return {
            "loss_recon": anchor,
            "loss_delta": anchor,
        }
    
    B, S = recon_model.shape[:2]
    BS = B * S
    
    # Reshape recon_model: [B, S, 3, pts_per_kpt*kpt_num] -> [BS, pts_per_kpt*kpt_num, 3]
    recon_model_reshaped = recon_model.permute(0, 1, 3, 2).contiguous()  # [B, S, pts_per_kpt*kpt_num, 3]
    recon_model_flat = recon_model_reshaped.view(BS, -1, 3)  # [BS, pts_per_kpt*kpt_num, 3]
    
    # Reshape input points: [B, S, K, 3] -> [BS, K, 3]
    pts_input_flat = pts_input.view(BS, -1, 3)  # [BS, K, 3]
    
    # Compute Chamfer distance
    if use_mask and "point_masks" in batch:
        # Use mask-aware Chamfer distance (similar to Net.py's ChamferDis_with_mask)
        point_masks = batch["point_masks"]  # [B, S, H, W] or similar
        # For now, we'll use a simplified version - can be enhanced later
        # If we have choose_indices, we can create a mask for sampled points
        loss_recon = ChamferDis(pts_input_flat, recon_model_flat)
    else:
        # Standard Chamfer distance
        loss_recon = ChamferDis(pts_input_flat, recon_model_flat)
    
    # Regularization loss: L2 norm of reconstruction deltas
    # recon_delta: [B, S, pts_per_kpt*kpt_num, 3]
    recon_delta_flat = recon_delta.view(BS, -1, 3)  # [BS, pts_per_kpt*kpt_num, 3]
    loss_delta = torch.norm(recon_delta_flat, p=2, dim=-1).mean()  # Mean over all points and batch
    
    loss_recon = check_and_fix_inf_nan(loss_recon, "loss_recon")
    loss_delta = check_and_fix_inf_nan(loss_delta, "loss_delta")
    
    return {
        "loss_recon": loss_recon,
        "loss_delta": loss_delta,
    }


def _compute_analytical_nocs_gt(predictions, batch):
    """
    Compute NOCS GT analytically from predicted keypoint 3D positions and GT pose.
    
    Following AG-Pose: nocs = (kpt_3d_cam - t_gt) / s_norm @ R_gt
    
    This is the inverse of the object-to-camera transform:
        p_cam = s * R @ p_nocs + t  =>  p_nocs = R^T @ (p_cam - t) / s
    In row-vector convention: p_nocs = (p_cam - t) / s @ R
    
    The GT is guaranteed to be mathematically consistent with the GT pose,
    unlike heatmap-sampled dense NOCS maps which may have discretization noise.
    
    Args:
        predictions: Dict with 'pred_kpt_3d' [B,S,kpt_num,3] in normalized camera space
        batch: Dict with 'abs_extrinsics' [B,S,3,4], 'sizes' [B,S,3] or [B,S], 'avg_scale' [B]
    
    Returns:
        gt_nocs_flat: [BS, kpt_num, 3] analytical NOCS ground truth
    """
    pred_kpt_3d = predictions["pred_kpt_3d"]  # [B, S, kpt_num, 3] in normalized camera space
    B, S, kpt_num, _ = pred_kpt_3d.shape
    
    # Convert from normalized camera space to original camera space
    avg_scale = batch["avg_scale"]  # [B]
    avg_scale_view = avg_scale.view(B, 1, 1, 1)  # [B, 1, 1, 1] for broadcasting
    kpt_3d_original = pred_kpt_3d * avg_scale_view  # [B, S, kpt_num, 3]
    
    # Get GT pose (already symmetry-canonicalized via extrinsics_sym)
    gt_extrinsics = batch["abs_extrinsics"]  # [B, S, 3, 4]
    gt_R = gt_extrinsics[:, :, :3, :3]  # [B, S, 3, 3]  object-to-camera rotation
    gt_t = gt_extrinsics[:, :, :3, 3]   # [B, S, 3]     translation
    
    # Get GT size and compute normalization factor
    gt_s = batch["sizes"]  # [B, S, 3] or [B, S] (bbox_side_len)
    if gt_s.dim() == 3:
        # [B, S, 3] -> compute L2 norm -> [B, S]
        s_norm = torch.norm(gt_s, dim=-1)
    elif gt_s.dim() == 2:
        # [B, S] scalar size
        s_norm = gt_s
    else:
        # [B] or scalar - expand to [B, S]
        s_norm = gt_s.unsqueeze(-1).expand(B, S) if gt_s.dim() == 1 else gt_s
    
    # Reshape for broadcasting: [B, S, 1, 1]
    s_norm = s_norm.view(B, S, 1, 1)
    
    # Analytical NOCS GT (row-vector convention):
    # nocs = (p_cam - t) / s_norm @ R
    # [B,S,kpt_num,3] - [B,S,1,3] = [B,S,kpt_num,3]
    kpt_centered = kpt_3d_original - gt_t.unsqueeze(2)
    kpt_scaled = kpt_centered / (s_norm + 1e-8)
    # [B,S,kpt_num,3] @ [B,S,3,3] = [B,S,kpt_num,3]
    gt_nocs = kpt_scaled @ gt_R
    
    # Flatten to [BS, kpt_num, 3]
    gt_nocs_flat = gt_nocs.reshape(B * S, kpt_num, 3)
    
    return gt_nocs_flat


def compute_nocs_loss(
    predictions,
    batch,
    smooth_l1_threshold=0.1,
    use_analytical_nocs_gt=False,
    nocs_symmetry_min_over_equivalents=True,
    n_continuous_samples=16,
    **kwargs
):
    """
    Compute NOCS loss for sparse keypoints.

    Supports three modes for NOCS ground truth:
    1. use_analytical_nocs_gt=True: Compute GT analytically from predicted keypoint 3D
       positions + GT pose (consistent with AG-Pose, guaranteed pose-NOCS consistency)
    2. 'kpt_nocs_gt' in predictions: Use precomputed GT from model forward (heatmap + dense map)
    3. Fallback: Compute from dense NOCS map + heatmap in loss function

    When sym_codes are available and nocs_symmetry_min_over_equivalents=True:
    - Discrete (code 2, 3): min-over-equivalents (enumerate all rotations)
    - Continuous (code 1) — 1 axis: invariant projection (r_perp, height), ignore free angle
    - Continuous (code 1) — 2+ axes (e.g. sphere): only supervise radial distance r
    
    Args:
        nocs_symmetry_min_over_equivalents: If True, use symmetry-aware loss
        n_continuous_samples: Deprecated, unused (kept for API compatibility)
    """
    pred_nocs = predictions["nocs"]

    if pred_nocs.dim() == 4:
        B, S, kpt_num, _ = pred_nocs.shape
        BS = B * S
        pred_nocs_flat = pred_nocs.view(BS, kpt_num, 3)
    else:
        pred_nocs_flat = pred_nocs
        BS, kpt_num, _ = pred_nocs_flat.shape
        if "nocs" in batch:
            B, S = batch["nocs"].shape[:2]
            assert BS == B * S, f"Batch size mismatch: BS={BS}, B*S={B*S}"
        else:
            B = batch.get("images").shape[0]
            S = batch.get("images").shape[1]

    # ===== Compute NOCS ground truth =====
    if use_analytical_nocs_gt and "pred_kpt_3d" in predictions and "abs_extrinsics" in batch:
        # Mode 1: Analytical NOCS GT from GT pose (AG-Pose style)
        # Guarantees mathematical consistency between NOCS GT and pose GT.
        # abs_extrinsics is symmetry-canonicalized in the data loader.
        gt_nocs_flat = _compute_analytical_nocs_gt(predictions, batch)
    elif "kpt_nocs_gt" in predictions:
        # Mode 2: Precomputed from model forward (heatmap + dense NOCS map)
        kpt_nocs_gt = predictions["kpt_nocs_gt"]  # [B,S,kpt_num,3]
        if kpt_nocs_gt.dim() == 4:
            B, S, kpt_num, _ = kpt_nocs_gt.shape
            BS = B * S
            gt_nocs_flat = kpt_nocs_gt.view(BS, kpt_num, 3)
        else:
            gt_nocs_flat = kpt_nocs_gt
    else:
        # Mode 3: Fallback — compute from dense NOCS map and heatmap
        heat_map = predictions["heat_map"]
        gt_nocs_dense = batch["nocs"]  # [B,S,H,W,3]
        choose_indices = batch["choose_indices"]  # [B,S,K_dense]

        # Get dimensions from prediction tensor
        B, S = gt_nocs_dense.shape[:2]
        H, W = gt_nocs_dense.shape[2:4]
        K_dense = choose_indices.shape[-1]  # 1024

        # Verify that BS = B * S
        BS = B * S
        assert pred_nocs_flat.shape[0] == BS, f"Batch size mismatch: BS={pred_nocs_flat.shape[0]}, B*S={BS}"

        # heat_map may come in [B, S, kpt_num, K_dense]; flatten if needed
        if heat_map.dim() == 4:
            heat_map_flat = heat_map.view(BS, kpt_num, K_dense)
        else:
            heat_map_flat = heat_map
        choose_indices_flat = choose_indices.view(BS, K_dense)  # [BS, K_dense]
        gt_nocs_flat_dense = gt_nocs_dense.view(BS, H*W, 3)  # [BS, H*W, 3]

        # Clamp indices to valid range 
        max_spatial_idx = H * W - 1
        choose_indices_flat = torch.clamp(choose_indices_flat, 0, max_spatial_idx)
        
        # Gather dense NOCS at chosen locations 
        gather_idx = choose_indices_flat.unsqueeze(-1).expand(-1, -1, 3)  # [BS, K_dense, 3]
        nocs_dense_sampled = torch.gather(gt_nocs_flat_dense, 1, gather_idx)  # [BS, K_dense, 3]
        
        # Compute weighted GT NOCS using heatmap attention 
        gt_nocs_flat = torch.bmm(heat_map_flat.detach(), nocs_dense_sampled)  # [BS, kpt_num, 3]

    gt_nocs_flat = check_and_fix_inf_nan(gt_nocs_flat, "gt_nocs")

    # ===== Symmetry-aware NOCS loss (min-over-equivalents) =====
    if "sym_codes" in batch and nocs_symmetry_min_over_equivalents:
        sym_codes = batch["sym_codes"]
        if isinstance(sym_codes, (list, tuple)):
            sym_codes = torch.tensor(sym_codes, device=pred_nocs_flat.device, dtype=torch.long)
        elif not isinstance(sym_codes, torch.Tensor):
            sym_codes = torch.as_tensor(sym_codes, device=pred_nocs_flat.device, dtype=torch.long)

        if sym_codes.dim() == 2 and sym_codes.shape == (B, 3):
            sym_codes = sym_codes.unsqueeze(1).expand(B, S, 3)
        sym_flat = sym_codes.reshape(BS, 3)

        unique_codes, inverse_idx = torch.unique(sym_flat, dim=0, return_inverse=True)
        device = pred_nocs_flat.device
        dtype = pred_nocs_flat.dtype

        losses_per_sample = torch.zeros(BS, device=device, dtype=dtype)

        for ui in range(unique_codes.shape[0]):
            code = unique_codes[ui]
            mask = (inverse_idx == ui)
            if not mask.any():
                continue

            pred_m = pred_nocs_flat[mask]  # [M, kpt_num, 3]
            gt_m = gt_nocs_flat[mask]      # [M, kpt_num, 3]

            continuous_axes = [ax for ax, c in zip('xyz', code) if int(c.item()) == 1]

            S_k = _generate_symmetry_group_for_nocs(
                code, device, dtype, n_continuous_samples=n_continuous_samples
            )  # [K, 3, 3]

            # gt_nocs @ S_k: row vectors [M,kpt_num,3] @ [K,3,3] -> [M,kpt_num,K,3]
            gt_m_k = torch.einsum("mpd,kdc->mpkc", gt_m, S_k)
            K = gt_m_k.shape[2]

            if len(continuous_axes) == 1:
                # Single continuous axis (e.g. bottle): supervise (r_perp, height)
                axis = continuous_axes[0]
                pred_proj = _project_nocs_to_invariant_continuous(pred_m, axis)  # [M, kpt_num, 2]
                gt_proj = _project_nocs_to_invariant_continuous(gt_m_k, axis)    # [M, kpt_num, K, 2]
                pred_proj_exp = pred_proj.unsqueeze(2).expand(-1, -1, K, -1)
                loss_mpk = _smooth_l1_per_point(
                    pred_proj_exp, gt_proj, threshold=smooth_l1_threshold
                )  # [M, kpt_num, K]
            elif len(continuous_axes) >= 2:
                # Multi-continuous (e.g. sphere): only supervise radial distance r = sqrt(x²+y²+z²)
                pred_proj = _project_nocs_to_radial(pred_m)   # [M, kpt_num, 1]
                gt_proj = _project_nocs_to_radial(gt_m_k)     # [M, kpt_num, K, 1]
                pred_proj_exp = pred_proj.unsqueeze(2).expand(-1, -1, K, -1)
                loss_mpk = _smooth_l1_per_point(
                    pred_proj_exp, gt_proj, threshold=smooth_l1_threshold
                )  # [M, kpt_num, K]
            else:
                # Discrete only (no continuous): use full 3D
                pred_m_exp = pred_m.unsqueeze(2).expand(-1, -1, K, -1)
                loss_mpk = _smooth_l1_per_point(
                    pred_m_exp, gt_m_k, threshold=smooth_l1_threshold
                )  # [M, kpt_num, K]

            loss_m = loss_mpk.min(dim=-1).values.mean(dim=-1)  # [M]
            losses_per_sample[mask] = loss_m

        loss_reg = losses_per_sample.mean()
    else:
        loss_reg = SmoothL1Dis(pred_nocs_flat, gt_nocs_flat, threshold=smooth_l1_threshold)

    return {
        "loss_conf_nocs": torch.tensor(0.0, device=pred_nocs.device),
        "loss_reg_nocs": loss_reg,
        "loss_grad_nocs": torch.tensor(0.0, device=pred_nocs.device),
    }


def regression_loss(pred, gt, mask, conf=None, gradient_loss_fn=None, gamma=1.0, alpha=0.2, valid_range=-1):
    """
    Core regression loss function with confidence weighting and optional gradient loss.
    
    Computes:
    1. gamma * ||pred - gt||^2 * conf - alpha * log(conf)
    2. Optional gradient loss
    
    Args:
        pred: (B, S, H, W, C) predicted values
        gt: (B, S, H, W, C) ground truth values
        mask: (B, S, H, W) valid pixel mask
        conf: (B, S, H, W) confidence weights (optional)
        gradient_loss_fn: Type of gradient loss ("normal", "grad", etc.)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        valid_range: Quantile range for outlier filtering
    
    Returns:
        loss_conf: Confidence-weighted loss
        loss_grad: Gradient loss (0 if not specified)
        loss_reg: Regular L2 loss
    """
    bb, ss, hh, ww, nc = pred.shape

    # Ensure boolean mask dtype (avoid uint8 deprecation warnings in indexing)
    mask = mask.bool()

    # Normalize gradient loss flag
    if gradient_loss_fn is None:
        gradient_loss_fn = ""

    # Compute L2 distance between predicted and ground truth points
    loss_reg = torch.norm(gt[mask] - pred[mask], dim=-1)
    loss_reg = check_and_fix_inf_nan(loss_reg, "loss_reg")

    # Confidence-weighted loss: gamma * loss * conf - alpha * log(conf)
    # This encourages the model to be confident on easy examples and less confident on hard ones
    loss_conf = gamma * loss_reg * conf[mask] - alpha * torch.log(conf[mask])
    loss_conf = check_and_fix_inf_nan(loss_conf, "loss_conf")
        
    # Initialize gradient loss
    loss_grad = 0

    # Prepare confidence for gradient loss if needed
    if "conf" in gradient_loss_fn:
        to_feed_conf = conf.reshape(bb*ss, hh, ww)
    else:
        to_feed_conf = None

    # Compute gradient loss if specified for spatial smoothness
    if "normal" in gradient_loss_fn:
        # Surface normal-based gradient loss
        loss_grad = gradient_loss_multi_scale_wrapper(
            pred.reshape(bb*ss, hh, ww, nc),
            gt.reshape(bb*ss, hh, ww, nc),
            mask.reshape(bb*ss, hh, ww),
            gradient_loss_fn=normal_loss,
            scales=3,
            conf=to_feed_conf,
        )
    elif "grad" in gradient_loss_fn:
        # Standard gradient-based loss
        loss_grad = gradient_loss_multi_scale_wrapper(
            pred.reshape(bb*ss, hh, ww, nc),
            gt.reshape(bb*ss, hh, ww, nc),
            mask.reshape(bb*ss, hh, ww),
            gradient_loss_fn=gradient_loss,
            conf=to_feed_conf,
        )

    # Process confidence-weighted loss
    if loss_conf.numel() > 0:
        # Filter out outliers using quantile-based thresholding
        if valid_range>0:
            loss_conf = filter_by_quantile(loss_conf, valid_range)

        loss_conf = check_and_fix_inf_nan(loss_conf, f"loss_conf_depth")
        loss_conf = loss_conf.mean()
    else:
        loss_conf = (0.0 * pred).mean()

    # Process regular regression loss
    if loss_reg.numel() > 0:
        # Filter out outliers using quantile-based thresholding
        if valid_range>0:
            loss_reg = filter_by_quantile(loss_reg, valid_range)

        loss_reg = check_and_fix_inf_nan(loss_reg, f"loss_reg_depth")
        loss_reg = loss_reg.mean()
    else:
        loss_reg = (0.0 * pred).mean()

    return loss_conf, loss_grad, loss_reg


def gradient_loss_multi_scale_wrapper(prediction, target, mask, scales=4, gradient_loss_fn = None, conf=None):
    """
    Multi-scale gradient loss wrapper. Applies gradient loss at multiple scales by subsampling the input.
    This helps capture both fine and coarse spatial structures.
    
    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values  
        mask: (B, H, W) valid pixel mask
        scales: Number of scales to use
        gradient_loss_fn: Gradient loss function to apply
        conf: (B, H, W) confidence weights (optional)
    """
    total = 0
    for scale in range(scales):
        step = pow(2, scale)  # Subsample by 2^scale

        total += gradient_loss_fn(
            prediction[:, ::step, ::step],
            target[:, ::step, ::step],
            mask[:, ::step, ::step],
            conf=conf[:, ::step, ::step] if conf is not None else None
        )

    total = total / scales
    return total


def normal_loss(prediction, target, mask, cos_eps=1e-8, conf=None, gamma=1.0, alpha=0.2):
    """
    Surface normal-based loss for geometric consistency.
    
    Computes surface normals from 3D point maps using cross products of neighboring points,
    then measures the angle between predicted and ground truth normals.
    
    Args:
        prediction: (B, H, W, 3) predicted 3D coordinates/points
        target: (B, H, W, 3) ground-truth 3D coordinates/points
        mask: (B, H, W) valid pixel mask
        cos_eps: Epsilon for numerical stability in cosine computation
        conf: (B, H, W) confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
    """
    # Convert point maps to surface normals using cross products
    pred_normals, pred_valids = point_map_to_normal(prediction, mask, eps=cos_eps)
    gt_normals,   gt_valids   = point_map_to_normal(target,     mask, eps=cos_eps)

    # Only consider regions where both predicted and GT normals are valid
    all_valid = pred_valids & gt_valids  # shape: (4, B, H, W)

    # Early return if not enough valid points
    divisor = torch.sum(all_valid)
    if divisor < 10:
        return 0

    # Extract valid normals
    pred_normals = pred_normals[all_valid].clone()
    gt_normals = gt_normals[all_valid].clone()

    # Compute cosine similarity between corresponding normals
    dot = torch.sum(pred_normals * gt_normals, dim=-1)

    # Clamp dot product to [-1, 1] for numerical stability
    dot = torch.clamp(dot, -1 + cos_eps, 1 - cos_eps)

    # Compute loss as 1 - cos(theta), instead of arccos(dot) for numerical stability
    loss = 1 - dot

    # Return mean loss if we have enough valid points
    if loss.numel() < 10:
        return 0
    else:
        loss = check_and_fix_inf_nan(loss, "normal_loss")

        if conf is not None:
            # Apply confidence weighting
            conf = conf[None, ...].expand(4, -1, -1, -1)
            conf = conf[all_valid].clone()

            loss = gamma * loss * conf - alpha * torch.log(conf)
            return loss.mean()
        else:
            return loss.mean()


def gradient_loss(prediction, target, mask, conf=None, gamma=1.0, alpha=0.2):
    """
    Gradient-based loss. Computes the L1 difference between adjacent pixels in x and y directions.
    
    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values
        mask: (B, H, W) valid pixel mask
        conf: (B, H, W) confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
    """
    # Expand mask to match prediction channels
    mask = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])
    M = torch.sum(mask, (1, 2, 3))

    # Compute difference between prediction and target
    diff = prediction - target
    diff = torch.mul(mask, diff)

    # Compute gradients in x direction (horizontal)
    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    # Compute gradients in y direction (vertical)
    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    # Clamp gradients to prevent outliers
    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)

    # Apply confidence weighting if provided
    if conf is not None:
        conf = conf[..., None].expand(-1, -1, -1, prediction.shape[-1])
        conf_x = conf[:, :, 1:]
        conf_y = conf[:, 1:, :]

        grad_x = gamma * grad_x * conf_x - alpha * torch.log(conf_x)
        grad_y = gamma * grad_y * conf_y - alpha * torch.log(conf_y)

    # Sum gradients and normalize by number of valid pixels
    grad_loss = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        grad_loss = torch.sum(grad_loss) / divisor

    return grad_loss


def point_map_to_normal(point_map, mask, eps=1e-6):
    """
    Convert 3D point map to surface normal vectors using cross products.
    
    Computes normals by taking cross products of neighboring point differences.
    Uses 4 different cross-product directions for robustness.
    
    Args:
        point_map: (B, H, W, 3) 3D points laid out in a 2D grid
        mask: (B, H, W) valid pixels (bool)
        eps: Epsilon for numerical stability in normalization
    
    Returns:
        normals: (4, B, H, W, 3) normal vectors for each of the 4 cross-product directions
        valids: (4, B, H, W) corresponding valid masks
    """
    with torch.amp.autocast("cuda", enabled=False):
        # Pad inputs to avoid boundary issues
        padded_mask = F.pad(mask, (1, 1, 1, 1), mode='constant', value=0)
        pts = F.pad(point_map.permute(0, 3, 1, 2), (1,1,1,1), mode='constant', value=0).permute(0, 2, 3, 1)

        # Get neighboring points for each pixel
        center = pts[:, 1:-1, 1:-1, :]   # B,H,W,3
        up     = pts[:, :-2,  1:-1, :]
        left   = pts[:, 1:-1, :-2 , :]
        down   = pts[:, 2:,   1:-1, :]
        right  = pts[:, 1:-1, 2:,   :]

        # Compute direction vectors from center to neighbors
        up_dir    = up    - center
        left_dir  = left  - center
        down_dir  = down  - center
        right_dir = right - center

        # Compute four cross products for different normal directions
        n1 = torch.cross(up_dir,   left_dir,  dim=-1)  # up x left
        n2 = torch.cross(left_dir, down_dir,  dim=-1)  # left x down
        n3 = torch.cross(down_dir, right_dir, dim=-1)  # down x right
        n4 = torch.cross(right_dir,up_dir,    dim=-1)  # right x up

        # Validity masks - require both direction pixels to be valid
        v1 = padded_mask[:, :-2,  1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, :-2]
        v2 = padded_mask[:, 1:-1, :-2 ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 2:,   1:-1]
        v3 = padded_mask[:, 2:,   1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, 2:]
        v4 = padded_mask[:, 1:-1, 2:  ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, :-2,  1:-1]

        # Stack normals and validity masks
        normals = torch.stack([n1, n2, n3, n4], dim=0)  # shape [4, B, H, W, 3]
        valids  = torch.stack([v1, v2, v3, v4], dim=0)  # shape [4, B, H, W]

        # Normalize normal vectors
        normals = F.normalize(normals, p=2, dim=-1, eps=eps)

    return normals, valids


def filter_by_quantile(loss_tensor, valid_range, min_elements=1000, hard_max=100):
    """
    Filter loss tensor by keeping only values below a certain quantile threshold.
    
    This helps remove outliers that could destabilize training.
    
    Args:
        loss_tensor: Tensor containing loss values
        valid_range: Float between 0 and 1 indicating the quantile threshold
        min_elements: Minimum number of elements required to apply filtering
        hard_max: Maximum allowed value for any individual loss
    
    Returns:
        Filtered and clamped loss tensor
    """
    if loss_tensor.numel() <= min_elements:
        # Too few elements, just return as-is
        return loss_tensor

    # Randomly sample if tensor is too large to avoid memory issues
    if loss_tensor.numel() > 100000000:
        # Flatten and randomly select 1M elements
        indices = torch.randperm(loss_tensor.numel(), device=loss_tensor.device)[:1_000_000]
        loss_tensor = loss_tensor.view(-1)[indices]

    # First clamp individual values to prevent extreme outliers
    loss_tensor = loss_tensor.clamp(max=hard_max)

    # Compute quantile threshold
    quantile_thresh = torch_quantile(loss_tensor.detach(), valid_range)
    quantile_thresh = min(quantile_thresh, hard_max)

    # Apply quantile filtering if enough elements remain
    quantile_mask = loss_tensor < quantile_thresh
    if quantile_mask.sum() > min_elements:
        return loss_tensor[quantile_mask]
    return loss_tensor


def torch_quantile(
    input,
    q,
    dim = None,
    keepdim: bool = False,
    *,
    interpolation: str = "nearest",
    out: torch.Tensor = None,
) -> torch.Tensor:
    """Better torch.quantile for one SCALAR quantile.

    Using torch.kthvalue. Better than torch.quantile because:
        - No 2**24 input size limit (pytorch/issues/67592),
        - Much faster, at least on big input sizes.

    Arguments:
        input (torch.Tensor): See torch.quantile.
        q (float): See torch.quantile. Supports only scalar input
            currently.
        dim (int | None): See torch.quantile.
        keepdim (bool): See torch.quantile. Supports only False
            currently.
        interpolation: {"nearest", "lower", "higher"}
            See torch.quantile.
        out (torch.Tensor | None): See torch.quantile. Supports only
            None currently.
    """
    # https://github.com/pytorch/pytorch/issues/64947
    # Sanitization: q
    try:
        q = float(q)
        assert 0 <= q <= 1
    except Exception:
        raise ValueError(f"Only scalar input 0<=q<=1 is currently supported (got {q})!")

    # Handle dim=None case
    if dim_was_none := dim is None:
        dim = 0
        input = input.reshape((-1,) + (1,) * (input.ndim - 1))

    # Set interpolation method
    if interpolation == "nearest":
        inter = round
    elif interpolation == "lower":
        inter = floor
    elif interpolation == "higher":
        inter = ceil
    else:
        raise ValueError(
            "Supported interpolations currently are {'nearest', 'lower', 'higher'} "
            f"(got '{interpolation}')!"
        )

    # Validate out parameter
    if out is not None:
        raise ValueError(f"Only None value is currently supported for out (got {out})!")

    # Compute k-th value
    k = inter(q * (input.shape[dim] - 1)) + 1
    out = torch.kthvalue(input, k, dim, keepdim=True, out=out)[0]

    # Handle keepdim and dim=None cases
    if keepdim:
        return out
    if dim_was_none:
        return out.squeeze()
    else:
        return out.squeeze(dim)

    return out


########################################################################################
########################################################################################

# Dirty code for tracking loss:

########################################################################################
########################################################################################

'''
def _compute_losses(self, coord_preds, vis_scores, conf_scores, batch):
    """Compute tracking losses using sequence_loss"""
    gt_tracks = batch["tracks"]  # B, S, N, 2
    gt_track_vis_mask = batch["track_vis_mask"]  # B, S, N

    # if self.training and hasattr(self, "train_query_points"):
    train_query_points = coord_preds[-1].shape[2]
    gt_tracks = gt_tracks[:, :, :train_query_points]
    gt_tracks = check_and_fix_inf_nan(gt_tracks, "gt_tracks", hard_max=None)

    gt_track_vis_mask = gt_track_vis_mask[:, :, :train_query_points]

    # Create validity mask that filters out tracks not visible in first frame
    valids = torch.ones_like(gt_track_vis_mask)
    mask = gt_track_vis_mask[:, 0, :] == True
    valids = valids * mask.unsqueeze(1)



    if not valids.any():
        print("No valid tracks found in first frame")
        print("seq_name: ", batch["seq_name"])
        print("ids: ", batch["ids"])
        print("time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        dummy_coord = coord_preds[0].mean() * 0          # keeps graph & grads
        dummy_vis = vis_scores.mean() * 0
        if conf_scores is not None:
            dummy_conf = conf_scores.mean() * 0
        else:
            dummy_conf = 0
        return dummy_coord, dummy_vis, dummy_conf                # three scalar zeros


    # Compute tracking loss using sequence_loss
    track_loss = sequence_loss(
        flow_preds=coord_preds,
        flow_gt=gt_tracks,
        vis=gt_track_vis_mask,
        valids=valids,
        **self.loss_kwargs
    )

    vis_loss = F.binary_cross_entropy_with_logits(vis_scores[valids], gt_track_vis_mask[valids].float())

    vis_loss = check_and_fix_inf_nan(vis_loss, "vis_loss", hard_max=None)


    # within 3 pixels
    if conf_scores is not None:
        gt_conf_mask = (gt_tracks - coord_preds[-1]).norm(dim=-1) < 3
        conf_loss = F.binary_cross_entropy_with_logits(conf_scores[valids], gt_conf_mask[valids].float())
        conf_loss = check_and_fix_inf_nan(conf_loss, "conf_loss", hard_max=None)
    else:
        conf_loss = 0

    return track_loss, vis_loss, conf_loss



def reduce_masked_mean(x, mask, dim=None, keepdim=False):
    for a, b in zip(x.size(), mask.size()):
        assert a == b
    prod = x * mask

    if dim is None:
        numer = torch.sum(prod)
        denom = torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = torch.sum(mask, dim=dim, keepdim=keepdim)

    mean = numer / denom.clamp(min=1)
    mean = torch.where(denom > 0,
                       mean,
                       torch.zeros_like(mean))
    return mean


def sequence_loss(flow_preds, flow_gt, vis, valids, gamma=0.8, vis_aware=False, huber=False, delta=10, vis_aware_w=0.1, **kwargs):
    """Loss function defined over sequence of flow predictions"""
    B, S, N, D = flow_gt.shape
    assert D == 2
    B, S1, N = vis.shape
    B, S2, N = valids.shape
    assert S == S1
    assert S == S2
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        flow_pred = flow_preds[i]

        i_loss = (flow_pred - flow_gt).abs()  # B, S, N, 2
        i_loss = check_and_fix_inf_nan(i_loss, f"i_loss_iter_{i}", hard_max=None)

        i_loss = torch.mean(i_loss, dim=3) # B, S, N

        # Combine valids and vis for per-frame valid masking.
        combined_mask = torch.logical_and(valids, vis)

        num_valid_points = combined_mask.sum()

        if vis_aware:
            combined_mask = combined_mask.float() * (1.0 + vis_aware_w)  # Add, don't add to the mask itself.
            flow_loss += i_weight * reduce_masked_mean(i_loss, combined_mask)
        else:
            if num_valid_points > 2:
                i_loss = i_loss[combined_mask]
                flow_loss += i_weight * i_loss.mean()
            else:
                i_loss = check_and_fix_inf_nan(i_loss, f"i_loss_iter_safe_check_{i}", hard_max=None)
                flow_loss += 0 * i_loss.mean()

    # Avoid division by zero if n_predictions is 0 (though it shouldn't be).
    if n_predictions > 0:
        flow_loss = flow_loss / n_predictions

    return flow_loss
'''
