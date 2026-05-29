# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Project-owned portions of this file are licensed under CC-BY-NC-SA-4.0.
# See LICENSE and NOTICE for details. Third-party notices remain applicable.


import os
import cv2
import glob
import argparse
import numpy as np
import torch
import sys
from PIL import Image
from pathlib import Path
from torchvision import transforms as TF
import struct
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle as cPickle
from tqdm import tqdm
import logging
from torch.utils.data import Dataset, DataLoader
from utils.housecat6d_eval_utils import evaluate_housecat

sys.path.append("opt/")
sys.path.append("utils/")

from opt.models.opt import OPT
from opt.utils.geometry import closed_form_inverse_se3
from typing import Optional, Tuple
from types import SimpleNamespace
from training.data.datasets.housecat import HouseCat6DTestDataset

cat_name_2_id = {
    "BG": 0, 
    "box": 1, 
    "bottle": 2, 
    "can": 3, 
    "cup": 4, 
    "remote": 5, 
    "teapot": 6, 
    "cutlery": 7, 
    "glass": 8, 
    "shoe": 9, 
    "tube": 10
}

def check_valid_tensor(input_tensor: Optional[torch.Tensor], name: str = "tensor") -> None:
    """
    Check if a tensor contains NaN or Inf values and log a warning if found.
    
    Args:
        input_tensor: The tensor to check
        name: Name of the tensor for logging purposes
    """
    if input_tensor is not None:
        if torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any():
            logging.warning(f"NaN or Inf found in tensor: {name}")

def check_and_fix_inf_nan(input_tensor, loss_name="default", hard_max=100):
    """
    Checks if 'input_tensor' contains inf or nan values and clamps extreme values.
    
    Args:
        input_tensor (torch.Tensor): The loss tensor to check and fix.
        loss_name (str): Name of the loss (for diagnostic prints).
        hard_max (float, optional): Maximum absolute value allowed. Values outside 
                                  [-hard_max, hard_max] will be clamped. If None, 
                                  no clamping is performed. Defaults to 100.
    """
    if input_tensor is None:
        return input_tensor
    
    # Check for inf/nan values
    has_inf_nan = torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any()
    if has_inf_nan:
        logging.warning(f"Tensor {loss_name} contains inf or nan values. Replacing with zeros.")
        input_tensor = torch.where(
            torch.isnan(input_tensor) | torch.isinf(input_tensor),
            torch.zeros_like(input_tensor),
            input_tensor
        )

    # Apply hard clamping if specified
    if hard_max is not None:
        input_tensor = torch.clamp(input_tensor, min=-hard_max, max=hard_max)

    return input_tensor

def normalize_camera_extrinsics_and_points_batch(
    extrinsics: torch.Tensor,
    cam_points: Optional[torch.Tensor] = None,
    world_points: Optional[torch.Tensor] = None,
    depths: Optional[torch.Tensor] = None,
    scale_by_points: bool = True,
    point_masks: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
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


    B, S, _, _ = extrinsics.shape
    device = extrinsics.device
    assert device == torch.device("cpu")


    # Convert extrinsics to homogeneous form: (B, N,4,4)
    extrinsics_homog = torch.cat(
        [
            extrinsics,
            torch.zeros((B, S, 1, 4), device=device),
        ],
        dim=-2,
    )
    extrinsics_homog[:, :, -1, -1] = 1.0

    # first_cam_extrinsic_inv, the inverse of the first camera's extrinsic matrix
    # which can be also viewed as the cam_to_world extrinsic matrix
    first_cam_extrinsic_inv = closed_form_inverse_se3(extrinsics_homog[:, 0])
    # new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv)
    new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv.unsqueeze(1))  # (B,N,4,4)


    if world_points is not None:
        # since we are transforming the world points to the first camera's coordinate system
        # we directly use the cam_from_world extrinsic matrix of the first camera
        # instead of using the inverse of the first camera's extrinsic matrix
        R = extrinsics[:, 0, :3, :3]
        t = extrinsics[:, 0, :3, 3]
        new_world_points = (world_points @ R.transpose(-1, -2).unsqueeze(1).unsqueeze(2)) + t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    else:
        new_world_points = None


    if scale_by_points:
        new_cam_points = cam_points.clone()
        new_depths = depths.clone()

        dist = new_world_points.norm(dim=-1)
        dist_sum = (dist * point_masks).sum(dim=[1,2,3])
        valid_count = point_masks.sum(dim=[1,2,3])
        avg_scale = (dist_sum / (valid_count + 1e-3)).clamp(min=1e-6, max=1e6)


        new_world_points = new_world_points / avg_scale.view(-1, 1, 1, 1, 1)
        new_extrinsics[:, :, :3, 3] = new_extrinsics[:, :, :3, 3] / avg_scale.view(-1, 1, 1)
        if depths is not None:
            new_depths = new_depths / avg_scale.view(-1, 1, 1, 1)
        if cam_points is not None:
            new_cam_points = new_cam_points / avg_scale.view(-1, 1, 1, 1, 1)
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
        return new_extrinsics, new_cam_points, new_world_points, new_depths, 1.0

def load_opt_model(checkpoint_path: str, device: str = "cuda"):
    """Load and initialize the OPT model from a checkpoint."""
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available.")
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Using device: {device}")
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if isinstance(ckpt, dict):
        state_dict = ckpt.get("state_dict", ckpt)
    else:
        state_dict = ckpt

    if not isinstance(state_dict, dict):
        raise TypeError("Invalid checkpoint format: expected a state_dict mapping.")

    # Keep a single deterministic normalization path.
    if all(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    model = OPT(enable_pose=True, enable_nocs=True, enable_metric_scale=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={len(missing)}, unexpected={len(unexpected)}. "
            "Please use the matching OPT checkpoint."
        )

    model.eval()
    model = model.to(device)
    print("Model loaded")
    return model, device

def infer_opt_outputs(model: OPT, batch: dict, device: str = "cuda", use_gt_intrinsics: bool = False):
    with torch.inference_mode():
        cat_labels = batch.get("cat_id")
        cat_labels = cat_labels.to(device) if cat_labels is not None else None
        choose_indices = batch.get("choose_indices")
        choose_indices = choose_indices.to(device) if choose_indices is not None else None
        nocs_gt = batch.get("nocs")
        nocs_gt = nocs_gt.to(device) if nocs_gt is not None else None
        depth_sensor = batch.get("depths_sensor")
        depth_sensor = depth_sensor.to(device) if depth_sensor is not None else None
        preds = model(
            images=batch["images"].to(device),
            cat_labels=cat_labels,
            choose_indices=choose_indices,
            nocs_gt=nocs_gt,
            depth_sensor=depth_sensor,
            category_names=batch.get("cat_name"),
            intrinsics=batch.get("intrinsics"),
            use_gt_intrinsics=use_gt_intrinsics,
        )

    nocs = preds.get("nocs", None) # [B,S=1,K,3]
    nocs = nocs.squeeze(1).detach().cpu().numpy() if nocs is not None else None # [B,K,3]
    world_points = preds.get("world_points", None) # [B,S=1,H,W,3]
    world_points = world_points.squeeze(1).detach().cpu().numpy() if world_points is not None else None # [B,H,W,3]
    depth_pred = preds.get("depth", None) # [B,S=1,H,W,1]
    depth_pred = depth_pred.squeeze(1).detach().cpu().numpy() if depth_pred is not None else None # [B,H,W,1]
    pred_kpt_3d = preds.get("pred_kpt_3d", None) # [B,S=1,K,3]
    pred_kpt_3d = pred_kpt_3d.squeeze(1).detach().cpu().numpy() if pred_kpt_3d is not None else None # [B,K,3]
    heat_map = preds.get("heat_map", None) # [B,S=1,K,K]
    heat_map = heat_map.squeeze(1).detach().cpu().numpy() if heat_map is not None else None # [B,K,K]
    kpt_nocs_gt = preds.get("kpt_nocs_gt", None) # [B,S=1,K,3]
    kpt_nocs_gt = kpt_nocs_gt.squeeze(1).detach().cpu().numpy() if kpt_nocs_gt is not None else None # [B,K,3]
    
    # Rotation from pose_head (normalized predictions)
    pred_R = preds.get("abs_pose", None)["rotation"] # [B,S=1,3,3]
    pred_R = pred_R.squeeze(1).detach().cpu().numpy() # [B,3,3]
    
    # Normalized translation and size from pose_head
    pred_t_norm = preds.get("abs_pose", None)["translation"] # [B,S=1,3]
    pred_t_norm = pred_t_norm.squeeze(1).detach().cpu().numpy() # [B,3]
    pred_size_norm = preds.get("abs_pose", None)["size"] # [B,S=1,3]
    pred_size_norm = pred_size_norm.squeeze(1).detach().cpu().numpy() # [B,3]
    
    # Absolute translation and size from absolute_scale_head (metric scale)
    abs_translation = preds.get("abs_translation", None)
    abs_translation = abs_translation.squeeze(1).detach().cpu().numpy() if abs_translation is not None else None # [B,3]
    abs_size = preds.get("abs_size", None)
    abs_size = abs_size.squeeze(1).detach().cpu().numpy() if abs_size is not None else None # [B,3]
    
    # Derived scale from comparison of absolute vs normalized predictions
    derived_scale = preds.get("derived_scale", None)
    derived_scale = derived_scale.squeeze(1).detach().cpu().numpy() if derived_scale is not None else None # [B,]
    
    return {
        'nocs': nocs,
        'world_points': world_points,
        'depth_pred': depth_pred,
        'pred_kpt_3d': pred_kpt_3d,
        'heat_map': heat_map,
        'kpt_nocs_gt': kpt_nocs_gt,
        'pred_R': pred_R,
        'pred_t_norm': pred_t_norm,
        'pred_size_norm': pred_size_norm,
        'abs_translation': abs_translation,
        'abs_size': abs_size,
        'derived_scale': derived_scale,
    }

def to_homogeneous(T_3x4):
    """Convert 3x4 transformation matrix to 4x4 homogeneous form."""
    batch_size = T_3x4.shape[0]
    T_4x4 = torch.zeros(batch_size, 4, 4, dtype=T_3x4.dtype, device=T_3x4.device)
    T_4x4[:, :3, :] = T_3x4
    T_4x4[:, 3, 3] = 1
    return T_4x4

def umeyama_similarity_transform(X, Y, point_conf=None, ransac=False, ransac_iters=200, inlier_thresh=0.02, fix_scale=False):
    """
    Weighted Umeyama / Procrustes estimator with optional RANSAC.

    Args:
        X: (N,3) source points
        Y: (N,3) target points
        point_conf: (N,) optional confidences/weights for points. If provided, used as weights and sampling probs for RANSAC.
        ransac: bool, whether to run RANSAC to robustly estimate the transform.
        ransac_iters: number of RANSAC iterations.
        inlier_thresh: distance threshold (meters) to consider an inlier.
        fix_scale: if True, enforce s=1 (pure SE3). If False, estimate scale (Sim3).

    Returns:
        R: (3,3) rotation
        t: (3,) translation
        s: scalar scale (1.0 if fix_scale=True)
        T: (4,4) homogeneous transform (T[:3,:3] = s*R)
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    assert X.shape == Y.shape, "Point sets must have same shape"

    N = X.shape[0]
    if N == 0:
        # empty, return identity
        R = np.eye(3)
        t = np.zeros(3)
        s = 1.0
        T = np.eye(4)
        return R, t, s, T

    if point_conf is None:
        weights = np.ones(N, dtype=np.float64)
    else:
        pc = np.asarray(point_conf, dtype=np.float64)
        if pc.shape[0] != N:
            # try to flatten
            pc = pc.flatten()[:N]
        weights = pc.copy()
        weights[weights < 0] = 0.0

    # Prevent all-zero weights
    if weights.sum() <= 0:
        weights = np.ones(N, dtype=np.float64)

    def compute_transform_from_inliers(idx, use_weights=True):
        Xs = X[idx]
        Ys = Y[idx]
        ws = weights[idx] if use_weights else np.ones(len(idx), dtype=np.float64)
        W = ws.sum()
        # weighted centroids
        mean_X = (ws[:, None] * Xs).sum(axis=0) / W
        mean_Y = (ws[:, None] * Ys).sum(axis=0) / W

        Xc = Xs - mean_X
        Yc = Ys - mean_Y

        H = Xc.T @ (ws[:, None] * Yc)
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        if fix_scale:
            s = 1.0
        else:
            denom = (ws * np.sum(Xc ** 2, axis=1)).sum()
            if denom > 1e-12:
                s = S.sum() / denom
            else:
                s = 1.0

        t = mean_Y - s * (R @ mean_X)

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = s * R
        T[:3, 3] = t
        return R, t, s, T

    if ransac and N >= 3:
        best_inliers = None
        best_inlier_count = 0
        best_model = None

        # sampling probabilities from weights
        probs = weights / weights.sum()

        for _ in range(ransac_iters):
            try:
                sample_k = min(3, N)
                sample_idx = np.random.choice(N, size=sample_k, replace=False, p=probs)
            except Exception:
                sample_idx = np.random.choice(N, size=min(3, N), replace=False)

            R_s, t_s, s_s, T_s = compute_transform_from_inliers(sample_idx, use_weights=False)

            # compute residuals for all points
            X_trans = (s_s * (R_s @ X.T).T) + t_s
            residuals = np.linalg.norm(X_trans - Y, axis=1)
            inliers = residuals < inlier_thresh
            inlier_count = int(inliers.sum())

            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inliers = inliers
                best_model = (R_s, t_s, s_s, T_s)

                # early exit if all points are inliers
                if inlier_count == N:
                    break

        if best_inliers is None:
            # fallback to full fit
            R, t, s, T = compute_transform_from_inliers(np.arange(N), use_weights=True)
            return R, t, s, T

        # refine using all inliers and use weights
        inlier_idx = np.nonzero(best_inliers)[0]
        R, t, s, T = compute_transform_from_inliers(inlier_idx, use_weights=True)
        return R, t, s, T
    else:
        # single closed-form weighted solution
        R, t, s, T = compute_transform_from_inliers(np.arange(N), use_weights=True)
        return R, t, s, T

def build_common_conf(training=True, debug=False):
    """Build common configuration compatible with BaseDataset."""
    augs = SimpleNamespace(
        scales=[1.0],  # No scale augmentation for inference
        cojitter=False,
        cojitter_ratio=0.5,
        color_jitter=0.0,
        gray_scale=0.0,
        gau_blur=0.0,
        aspects=[1.0],  # No aspect ratio augmentation for inference
    )
    return SimpleNamespace(
        img_size=518,
        patch_size=14,
        augs=augs,
        rescale=True,
        rescale_aug=False,  # Disable rescale augmentation for inference
        landscape_check=True,
        debug=debug,
        training=training,
        get_nearby=False,
        load_depth=True,
        inside_random=False,
        allow_duplicate_img=True,
        fix_img_num=0,
        fix_aspect_ratio=1.0,
        load_track=False,
        track_num=0,
    )

def get_logger(level_print, level_save, path_file, name_logger = "logger"):
    # level: logging.INFO / logging.WARN
    logger = logging.getLogger(name_logger)
    logger.setLevel(level = logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    # set file handler
    handler_file = logging.FileHandler(path_file)
    handler_file.setLevel(level_save)
    handler_file.setFormatter(formatter)
    logger.addHandler(handler_file)
    # set console holder
    handler_view = logging.StreamHandler()
    handler_view.setFormatter(formatter)
    handler_view.setLevel(level_print)
    logger.addHandler(handler_view)
    return logger


def collate_single_item(batch_list):
    """DataLoader collate_fn for `batch_size=1` without stacking dict fields."""
    return batch_list[0]


class HouseCat6DTestDataLoader(Dataset):
    """
    Wrap `HouseCat6DTestDataset` to provide `__len__`/`__getitem__` so we can use
    a PyTorch `DataLoader`.

    Note: `HouseCat6DTestDataset.get_data()` already returns a fully-formed batch
    dict (with variable-length lists), so the DataLoader should not try to
    "stack" fields; we handle this via `collate_fn` in the caller.
    """

    def __init__(self, dataset: HouseCat6DTestDataset, aspect_ratio: float = 1.0):
        self.dataset = dataset
        self.aspect_ratio = aspect_ratio

    def __len__(self) -> int:
        return int(getattr(self.dataset, "len_train"))

    def __getitem__(self, idx: int) -> dict:
        # `sample_index` maps 1:1 to `unique_keys` entries inside the dataset.
        return self.dataset.get_data(sample_index=int(idx), aspect_ratio=self.aspect_ratio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Absolute pose estimation on HouseCat6D using dataset loader')
    parser.add_argument('--data_root', type=str, required=True, help='Path to the HouseCat6D dataset root')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--img_per_seq', type=int, default=1, help='Number of frames per sequence to load')
    parser.add_argument('--seq_index', type=int, default=None, help='Sequence index to evaluate; random if None')
    parser.add_argument('--aspect', type=float, default=1.0, help='Target aspect ratio for crop')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the HouseCat6D checkpoint')
    parser.add_argument('--num_seqs', type=int, default=10, help='Number of sequences to iterate (ignored if --seq_index is set)')
    parser.add_argument('--save_path', type=str, default='outputs/housecat6d_abs', help='Path to save results')
    parser.add_argument('--save_pkl', type=bool, default=True, help='Save results as pkl')
    parser.add_argument('--eval_only', type=bool, default=False, help='Only evaluate the results')
    parser.add_argument(
        '--use_gt_intrinsics',
        action='store_true',
        help='At eval, replace predicted FoV in pose_enc with GT intrinsics (converted to pose_enc format).',
    )
    args = parser.parse_args()

    torch.cuda.set_device(0)
    os.makedirs(args.save_path, exist_ok=True)
    logger = get_logger(
        level_print=logging.INFO, level_save=logging.WARNING, path_file=os.path.join(args.save_path, f"test_logger_{args.checkpoint.split('/')[-1].split('.')[0]}.log")
    )

    if args.eval_only:
        evaluate_housecat(args.save_path, logger)
        logger.info(f'Completed evaluation on unseen objects from checkpoint {args.checkpoint}.')
        sys.exit(0)

    # Build dataset
    conf = build_common_conf(training=False, debug=False)
    ds = HouseCat6DTestDataset(
        common_conf=conf,
        data_root=args.data_root,
        split=args.split,
        force_rebuild_cache=False,
        sample_num=1024,
    )

    logger.info(f"Found {len(ds.unique_keys)} frames in HouseCat6D {args.split}")
    logger.info(f"Total test samples (objects): {ds.len_train}")
    
    if ds.len_train == 0:
        logger.error("No test samples found. Check --data_root and dataset layout.")
    
    model, device = load_opt_model(checkpoint_path=args.checkpoint)

    # `HouseCat6DTestDataset` does not implement `__getitem__`, so we wrap it.
    wrapper = HouseCat6DTestDataLoader(ds, aspect_ratio=args.aspect)
    loader = DataLoader(
        wrapper,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # dataset code uses CUDA internally; avoid worker processes
        collate_fn=collate_single_item,
    )

    for batch in tqdm(loader, total=ds.len_train, desc="Evaluating sequences", leave=False):

        seq_name = batch["seq_name"]
        scene_name = seq_name.rsplit('_', 1)[0]

        save_path = os.path.join(args.save_path, scene_name)
        os.makedirs(save_path, exist_ok=True)

        # Convert category name(s) to ID(s)
        def get_cat_id(name):
            if name not in cat_name_2_id:
                print(f"Warning: Unknown category '{name}', using default ID 0")
                return 0
            return cat_name_2_id[name]
        
        cat_name = batch.get("cat_name", None)
        cat_id = None
        if cat_name is not None:
            cat_id = [get_cat_id(n) for n in cat_name] if isinstance(cat_name, list) else get_cat_id(cat_name)

        images = torch.from_numpy(np.stack(batch["images"]).astype(np.float32)).contiguous()
        images = images.permute(0,3,1,2).to(torch.get_default_dtype()).div(255)
        images = images.unsqueeze(1)
        depths_sensor = torch.from_numpy(np.stack(batch["depths_sensor"]).astype(np.float32)).unsqueeze(1)
        extrinsics = torch.from_numpy(np.stack(batch["extrinsics"]).astype(np.float32)).unsqueeze(1)
        intrinsics = torch.from_numpy(np.stack(batch["intrinsics"]).astype(np.float32)).unsqueeze(1)
        bboxes = np.stack(batch["bboxes"]).astype(np.int64)

        # Convert category ID(s) to tensor
        if cat_id is not None:
            B = len(images)
            if isinstance(cat_id, list):
                if len(cat_id) != B:
                    raise ValueError(f"Length of cat_id list ({len(cat_id)}) doesn't match batch size ({B})")
                batch["cat_id"] = torch.tensor(cat_id, dtype=torch.long)
            else:
                batch["cat_id"] = torch.full((B,), cat_id, dtype=torch.long)

        if "nocs" in batch:
            nocs = torch.from_numpy(np.stack(batch["nocs"]).astype(np.float32)).unsqueeze(1)
            batch["nocs"] = nocs
        if "inst_masks" in batch:
            inst_masks = torch.from_numpy(np.stack(batch["inst_masks"]).astype(np.uint8)).bool().unsqueeze(1)
            batch["inst_masks"] = inst_masks
        if "choose_indices" in batch:
            choose_indices = torch.from_numpy(np.stack(batch["choose_indices"]).astype(np.int64)).unsqueeze(1)
            batch["choose_indices"] = choose_indices
        if "intrinsics" in batch:
            batch["intrinsics"] = intrinsics

        batch["images"] = images
        batch["abs_extrinsics"] = extrinsics.clone()  # [B, S, 3, 4]
        batch["depths_sensor"] = depths_sensor
        batch["scales"] = np.stack(batch['scales'])

        predictions = infer_opt_outputs(model, batch, device=device, use_gt_intrinsics=args.use_gt_intrinsics)

        result = {}

        B = len(images)  # Number of object frames
        # Decode class IDs from category names using cat_name_2_id dictionary
        cat_names = batch['cat_name']
        if isinstance(cat_names, list):
            result['gt_class_ids'] = np.array([cat_name_2_id.get(name, 0) for name in cat_names])
        else:
            result['gt_class_ids'] = np.array([cat_name_2_id.get(cat_names, 0)])
        gt_RTs = to_homogeneous(batch['abs_extrinsics'][:,0, ...]).numpy() # [B,4,4]
        RTs = []
        s = np.linalg.norm(np.stack(batch['sizes']), axis=1)
        for each_idx in range(B):
            matrix = np.identity(4)
            matrix[:3, :3] = gt_RTs[:, :3, :3][each_idx] * s[each_idx]
            matrix[:3, 3] = gt_RTs[:, :3, 3][each_idx]
            RTs.append(matrix)
        RTs = np.stack(RTs, 0)

        result['gt_RTs'] = RTs
        result['gt_scales'] = batch["sizes"] / s[:, np.newaxis]
        result['gt_handle_visibility'] = None

        result['pred_class_ids'] = result['gt_class_ids']
        result['pred_scores'] = np.ones_like(result['gt_class_ids'])
        result['pred_bboxes'] = bboxes

        # Use absolute predictions from absolute_scale_head
        pred_rotation = torch.from_numpy(predictions['pred_R'])

        pred_translation = predictions.get('abs_translation', None)
        pred_size = predictions.get('abs_size', None)
        pred_translation = torch.from_numpy(np.asarray(pred_translation))
        pred_size = torch.from_numpy(np.asarray(pred_size))

        # Construct pred_RTs: T = [sR | t; 0 0 0 1]
        # Note: In HouseCat6D format, the scale is applied to the rotation part
        pred_RTs = torch.eye(4).unsqueeze(0).repeat(B, 1, 1).float()
        pred_RTs[:, :3, 3] = pred_translation
        
        # Compute scale factor as norm of predicted size
        s_pred = torch.norm(pred_size, dim=1, keepdim=True)
        pred_RTs[:, :3, :3] = pred_rotation * s_pred[:, :, None]
        
        # Normalized size (unit norm)
        pred_scales = pred_size / s_pred

        result['pred_RTs'] = pred_RTs.detach().cpu().numpy()
        result['pred_scales'] = pred_scales.detach().cpu().numpy()

        pkl_path =  batch["filepaths"][0].split('/')[-1].replace('.png', '.pkl')

        if args.save_pkl:
            with open(os.path.join(save_path, pkl_path), 'wb') as f:
                cPickle.dump(result, f)
            print(f"Saved result to {os.path.join(save_path, pkl_path)}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    logger.info(f'Completed evaluation of checkpoint {args.checkpoint}.')
    logger.info("Running evaluation...")
    evaluate_housecat(args.save_path, logger)
    logger.info("Evaluation complete.")
