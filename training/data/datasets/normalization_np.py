# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Project-owned portions of this file are licensed under CC-BY-NC-SA-4.0.
# See LICENSE and NOTICE for details. Third-party notices remain applicable.

import numpy as np
import logging
from typing import Tuple, Optional

def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
        R (optional): Nx3x3 array or tensor of rotation matrices.
        T (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.

    Shapes:
        se3: (N, 4, 4)
        R: (N, 3, 3)
        T: (N, 3, 1)
    """
    # Assuming numpy for this context
    is_numpy = isinstance(se3, np.ndarray)
    if not is_numpy:
        raise ValueError("This NumPy version assumes numpy arrays.")

    # Validate shapes
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4) or (N,3,4), got {se3.shape}.")

    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # Transpose R
    R_transposed = np.transpose(R, (0, 2, 1))
    # -R^T t for NumPy
    top_right = -np.matmul(R_transposed, T)
    inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix

def check_valid_tensor(input_tensor: Optional[np.ndarray], name: str = "tensor") -> None:
    """
    Check if a tensor contains NaN or Inf values and log a warning if found.
    
    Args:
        input_tensor: The tensor to check
        name: Name of the tensor for logging purposes
    """
    if input_tensor is not None:
        if np.isnan(input_tensor).any() or np.isinf(input_tensor).any():
            logging.warning(f"NaN or Inf found in tensor: {name}")

def check_and_fix_inf_nan(input_tensor, loss_name="default", hard_max=100):
    """
    Checks if 'input_tensor' contains inf or nan values and clamps extreme values.
    
    Args:
        input_tensor (np.ndarray): The loss tensor to check and fix.
        loss_name (str): Name of the loss (for diagnostic prints).
        hard_max (float, optional): Maximum absolute value allowed. Values outside 
                                  [-hard_max, hard_max] will be clamped. If None, 
                                  no clamping is performed. Defaults to 100.
    """
    if input_tensor is None:
        return input_tensor
    
    # Check for inf/nan values
    has_inf_nan = np.isnan(input_tensor).any() or np.isinf(input_tensor).any()
    if has_inf_nan:
        logging.warning(f"Tensor {loss_name} contains inf or nan values. Replacing with zeros.")
        input_tensor = np.nan_to_num(input_tensor, nan=0.0, posinf=0.0, neginf=0.0)

    # Apply hard clamping if specified
    if hard_max is not None:
        input_tensor = np.clip(input_tensor, min=-hard_max, max=hard_max)

    return input_tensor

def normalize_camera_extrinsics_and_points_batch_numpy(
    extrinsics: np.ndarray,
    cam_points: Optional[np.ndarray] = None,
    world_points: Optional[np.ndarray] = None,
    depths: Optional[np.ndarray] = None,
    scale_by_points: bool = False,
    point_masks: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Normalize camera extrinsics and corresponding 3D points.
    
    This function transforms the coordinate system to be centered at the first camera
    and optionally scales the scene to have unit average distance.
    
    Args:
        extrinsics: Camera extrinsic matrices of shape (B, S, 3, 4)
        cam_points: 3D points in camera coordinates of shape (B, S, H, W, 3) or (*,3)
        world_points: 3D points in world coordinates of shape (B, S, H, W, 3) or (*,3)
        depths: Depth maps of shape (B, S, H, W)
        scale_by_points: Whether to normalize the scale based on point distances
        point_masks: Boolean masks for valid points of shape (B, S, H, W)
    
    Returns:
        Tuple containing:
        - Normalized camera extrinsics of shape (B, S, 3, 4)
        - Normalized camera points (same shape as input cam_points)
        - Normalized world points (same shape as input world_points)
        - Normalized depths (same shape as input depths)
    """
    # Validate inputs
    check_valid_tensor(extrinsics, "extrinsics")
    check_valid_tensor(cam_points, "cam_points")
    check_valid_tensor(world_points, "world_points")
    check_valid_tensor(depths, "depths")

    B, S = extrinsics.shape[:2]

    # Convert extrinsics to homogeneous form: (B, S, 4, 4)
    extrinsics_homog = np.concatenate(
        [
            extrinsics,
            np.zeros((B, S, 1, 4)),
        ],
        axis=-2,
    )
    extrinsics_homog[:, :, -1, -1] = 1.0

    # first_cam_extrinsic_inv, the inverse of the first camera's extrinsic matrix
    # which can be also viewed as the cam_to_world extrinsic matrix
    first_cam_extrinsic_inv = closed_form_inverse_se3(extrinsics_homog[:, 0])
    new_extrinsics = np.matmul(extrinsics_homog, np.expand_dims(first_cam_extrinsic_inv, axis=1))  # (B,S,4,4)

    if world_points is not None:
        # since we are transforming the world points to the first camera's coordinate system
        # we directly use the cam_from_world extrinsic matrix of the first camera
        # instead of using the inverse of the first camera's extrinsic matrix
        R = extrinsics[:, 0, :3, :3]
        t = extrinsics[:, 0, :3, 3]
        R_t = np.transpose(R, (0, 2, 1))[:, np.newaxis, np.newaxis, :, :]
        t_exp = t[:, np.newaxis, np.newaxis, np.newaxis, :]
        new_world_points = np.matmul(world_points, R_t) + t_exp
    else:
        new_world_points = None

    new_cam_points = cam_points
    new_depths = depths

    if scale_by_points:
        if new_cam_points is not None:
            new_cam_points = np.copy(new_cam_points)
        if new_depths is not None:
            new_depths = np.copy(new_depths)
        if point_masks is None:
            raise ValueError("point_masks required when scale_by_points is True")

        dist = np.linalg.norm(new_world_points, axis=-1)
        dist_sum = np.sum(dist * point_masks, axis=(1, 2, 3), dtype=np.float32)
        valid_count = np.sum(point_masks, axis=(1, 2, 3))
        avg_scale = np.clip(dist_sum / (valid_count + 1e-3), a_min=1e-6, a_max=1e6)

        print("avg_scale:", avg_scale)

        new_world_points = new_world_points / avg_scale[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        new_extrinsics[:, :, :3, 3] = new_extrinsics[:, :, :3, 3] / avg_scale[:, np.newaxis, np.newaxis]
        if depths is not None:
            new_depths = new_depths / avg_scale[:, np.newaxis, np.newaxis, np.newaxis]
        if cam_points is not None:
            new_cam_points = new_cam_points / avg_scale[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    else:
        return new_extrinsics[:, :, :3], cam_points, new_world_points, depths, 1.0

    new_extrinsics = new_extrinsics[:, :, :3] # 4x4 -> 3x4
    new_extrinsics = check_and_fix_inf_nan(new_extrinsics, "new_extrinsics", hard_max=None)
    new_cam_points = check_and_fix_inf_nan(new_cam_points, "new_cam_points", hard_max=None)
    new_world_points = check_and_fix_inf_nan(new_world_points, "new_world_points", hard_max=None)
    new_depths = check_and_fix_inf_nan(new_depths, "new_depths", hard_max=None)

    if scale_by_points:
        return new_extrinsics, new_cam_points, new_world_points, new_depths, avg_scale
    else:
        return new_extrinsics, new_cam_points, new_world_points, new_depths, 1.0*np.ones((B,))