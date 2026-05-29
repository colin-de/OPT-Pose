# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Project-owned portions of this file are licensed under CC-BY-NC-SA-4.0.
# See LICENSE and NOTICE for details. Third-party notices remain applicable.

import os
import sys
import json
import cv2
import math
import numpy as np
import torch
import torch.nn.functional as F
import glob
import _pickle as cPickle
from typing import Tuple, List, Dict, Optional
from collections import defaultdict
import hashlib
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
"""
Ensure imports work both when this file is imported as a module and when run directly.
We prefer absolute package imports, with a minimal fallback.
"""
# Allow running this file directly as a script by adding repo root to sys.path
_CURR_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_CURR_DIR, "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Prefer absolute imports so both contexts work; fallback to local if needed
try:
    from training.data.datasets.normalization_np import normalize_camera_extrinsics_and_points_batch_numpy
    from training.data.datasets.solve_sim3 import umeyama_similarity_transform
except ImportError:
    _TRAIN_DIR = os.path.abspath(os.path.join(_CURR_DIR, "../.."))
    if _TRAIN_DIR not in sys.path:
        sys.path.insert(0, _TRAIN_DIR)
    from normalization_np import normalize_camera_extrinsics_and_points_batch_numpy
    from solve_sim3 import umeyama_similarity_transform

from training.data.base_dataset import BaseDataset
from training.data.dataset_util import threshold_depth_map
from opt.utils.geometry import depth_to_world_coords_points
from training.data.datasets.hash_utils import stable_seq_id
from training.data.datasets.structs import CameraModel, AlignedBox2f
from training.data.datasets.cache_utils import resolve_cache_dir
# Import misc module with multiple fallback strategies
import importlib
import importlib.util
try:
    misc = importlib.import_module('training.data.datasets.misc')
except ImportError:
    try:
        if _CURR_DIR not in sys.path:
            sys.path.insert(0, _CURR_DIR)
        misc = importlib.import_module('misc')
    except ImportError:
        # Last resort - direct import from file
        misc_path = os.path.join(_CURR_DIR, 'misc.py')
        spec = importlib.util.spec_from_file_location("misc", misc_path)
        misc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(misc)

# Import HouseCat6D specific utilities
from training.data.datasets.housecat6d_utils import (
    load_housecat_depth,
    load_housecat_depth_sensor,
    fill_missing,
    get_bbox_from_mask,
)


class CameraWarper:
    """PyTorch + CUDA accelerated camera warping"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.grid_cache = {}  # Cache for reusable grids
    
    def compute_warp_grid_torch(
        self,
        src_camera,
        dst_camera,
        use_half_pixel=True,
        depth_check=True,
    ):
        """
        Compute warping grid using PyTorch on GPU
        
        Returns:
            grid: [1, H, W, 2] normalized grid for F.grid_sample
            valid_mask: [H, W] boolean mask for valid pixels
        """
        W, H = dst_camera.width, dst_camera.height
        
        # Create pixel grid on GPU
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=self.device, dtype=torch.float32),
            torch.arange(W, device=self.device, dtype=torch.float32),
            indexing='ij'
        )
        
        if use_half_pixel:
            dst_win_pts = torch.stack([grid_x + 0.5, grid_y + 0.5], dim=-1)
        else:
            dst_win_pts = torch.stack([grid_x, grid_y], dim=-1)
        
        dst_win_pts_flat = dst_win_pts.reshape(-1, 2)
        
        # Convert cameras to torch tensors
        K_dst = torch.tensor([
            [dst_camera.f[0], 0, dst_camera.c[0]],
            [0, dst_camera.f[1], dst_camera.c[1]],
            [0, 0, 1]
        ], device=self.device, dtype=torch.float32)
        
        K_src = torch.tensor([
            [src_camera.f[0], 0, src_camera.c[0]],
            [0, src_camera.f[1], src_camera.c[1]],
            [0, 0, 1]
        ], device=self.device, dtype=torch.float32)
        
        T_dst_world = torch.from_numpy(dst_camera.T_world_from_eye).to(self.device).float()
        T_src_world = torch.from_numpy(src_camera.T_world_from_eye).to(self.device).float()
        
        # Unproject destination pixels to rays
        K_dst_inv = torch.inverse(K_dst)
        ones = torch.ones((dst_win_pts_flat.shape[0], 1), device=self.device)
        pixel_coords_h = torch.cat([dst_win_pts_flat, ones], dim=1)  # [N, 3]
        
        rays_dst = torch.matmul(pixel_coords_h, K_dst_inv.T)  # [N, 3]
        
        # Transform to world space
        rays_dst_h = torch.cat([rays_dst, ones], dim=1)  # [N, 4]
        rays_world = torch.matmul(rays_dst_h, T_dst_world.T)[:, :3]  # [N, 3]
        
        # Transform to source camera space
        T_world_src = torch.inverse(T_src_world)
        rays_world_h = torch.cat([rays_world, ones], dim=1)
        rays_src = torch.matmul(rays_world_h, T_world_src.T)[:, :3]  # [N, 3]
        
        # Depth check
        valid_mask = torch.ones(H * W, device=self.device, dtype=torch.bool)
        if depth_check:
            valid_mask = rays_src[:, 2] > 0
        
        # Project to source camera pixels
        pixel_src = torch.matmul(rays_src, K_src.T)  # [N, 3]
        pixel_src = pixel_src[:, :2] / (pixel_src[:, 2:3] + 1e-8)  # [N, 2]
        
        if use_half_pixel:
            pixel_src -= 0.5
        
        # Normalize to [-1, 1] for grid_sample
        pixel_src[:, 0] = 2.0 * pixel_src[:, 0] / (src_camera.width - 1) - 1.0
        pixel_src[:, 1] = 2.0 * pixel_src[:, 1] / (src_camera.height - 1) - 1.0
        
        # Invalidate out-of-bound pixels
        valid_mask &= (pixel_src[:, 0] >= -1) & (pixel_src[:, 0] <= 1)
        valid_mask &= (pixel_src[:, 1] >= -1) & (pixel_src[:, 1] <= 1)
        
        # Set invalid pixels to out of bounds
        pixel_src[~valid_mask] = -2.0
        
        grid = pixel_src.reshape(1, H, W, 2)
        valid_mask = valid_mask.reshape(H, W)
        
        return grid, valid_mask
    
    def warp_image_torch(
        self,
        src_image,
        grid,
        mode='bilinear',
        mask_invalid=True,
        valid_mask=None,
    ):
        """
        Warp image using precomputed grid
        
        Args:
            src_image: numpy array [H, W] or [H, W, C] or torch tensor
            grid: [1, H, W, 2] normalized grid
            mode: 'bilinear' or 'nearest'
            mask_invalid: whether to mask invalid pixels as 0
            valid_mask: [H, W] boolean mask
        
        Returns:
            warped: torch tensor [C, H, W] or [H, W]
        """
        # Convert to torch if needed
        if isinstance(src_image, np.ndarray):
            src_image = torch.from_numpy(src_image).to(self.device)
        
        # Handle different input shapes
        original_shape = src_image.shape
        if src_image.ndim == 2:  # [H, W]
            src_image = src_image.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            squeeze_output = True
        elif src_image.ndim == 3:  # [H, W, C]
            src_image = src_image.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
            squeeze_output = False
        else:
            squeeze_output = False
        
        # Convert to float if needed
        if src_image.dtype == torch.uint8:
            src_image = src_image.float()
        elif src_image.dtype == torch.bool:
            src_image = src_image.float()
        
        # Perform warping
        warped = F.grid_sample(
            src_image,
            grid,
            mode=mode,
            padding_mode='zeros',
            align_corners=True
        )
        
        # Apply valid mask
        if mask_invalid and valid_mask is not None:
            if warped.shape[1] == 1:  # Single channel
                warped[0, 0, ~valid_mask] = 0
            else:  # Multi-channel
                warped[0, :, ~valid_mask] = 0
        
        # Squeeze and return
        if squeeze_output:
            return warped.squeeze()  # [H, W]
        else:
            return warped.squeeze(0)  # [C, H, W]
    
    def warp_depth_torch(
        self,
        src_camera,
        dst_camera,
        src_depth,
        grid,
        valid_mask,
    ):
        """
        Warp depth image with proper depth value transformation
        
        Args:
            src_depth: numpy array [H, W] or torch tensor
            grid: [1, H, W, 2] normalized grid
            valid_mask: [H, W] boolean mask
        
        Returns:
            warped_depth: torch tensor [H, W]
        """
        # Convert to torch if needed
        if isinstance(src_depth, np.ndarray):
            src_depth = torch.from_numpy(src_depth).to(self.device).float()
        
        # Check if camera extrinsics changed
        T_src = torch.from_numpy(src_camera.T_world_from_eye).to(self.device).float()
        T_dst = torch.from_numpy(dst_camera.T_world_from_eye).to(self.device).float()
        
        if not torch.allclose(T_src, T_dst, atol=1e-6):
            # Need to transform depth values
            H, W = src_depth.shape
            
            # Get valid depth pixels
            depth_valid_mask = src_depth > 0
            valid_ys, valid_xs = torch.nonzero(depth_valid_mask, as_tuple=True)
            
            if len(valid_xs) > 0:
                # Unproject to 3D in source camera
                K_src = torch.tensor([
                    [src_camera.f[0], 0, src_camera.c[0]],
                    [0, src_camera.f[1], src_camera.c[1]],
                    [0, 0, 1]
                ], device=self.device, dtype=torch.float32)
                
                K_src_inv = torch.inverse(K_src)
                
                if hasattr(src_camera, 'USE_HALF_PIXEL') and src_camera.USE_HALF_PIXEL:
                    pixel_coords = torch.stack([
                        valid_xs.float() + 0.5,
                        valid_ys.float() + 0.5,
                        torch.ones_like(valid_xs, dtype=torch.float32)
                    ], dim=1)
                else:
                    pixel_coords = torch.stack([
                        valid_xs.float(),
                        valid_ys.float(),
                        torch.ones_like(valid_xs, dtype=torch.float32)
                    ], dim=1)
                
                rays = torch.matmul(pixel_coords, K_src_inv.T)
                depths = src_depth[valid_ys, valid_xs].unsqueeze(1)
                pts_src = rays * (depths / rays[:, 2:3])
                
                # Transform to world
                ones = torch.ones((pts_src.shape[0], 1), device=self.device)
                pts_src_h = torch.cat([pts_src, ones], dim=1)
                pts_world = torch.matmul(pts_src_h, T_src.T)[:, :3]
                
                # Transform to destination camera
                T_world_dst = torch.inverse(T_dst)
                pts_world_h = torch.cat([pts_world, ones], dim=1)
                pts_dst = torch.matmul(pts_world_h, T_world_dst.T)[:, :3]
                
                # Update depth values
                src_depth = src_depth.clone()
                src_depth[valid_ys, valid_xs] = pts_dst[:, 2]
        
        # Warp the depth image
        warped_depth = self.warp_image_torch(
            src_depth,
            grid,
            mode='nearest',
            mask_invalid=True,
            valid_mask=valid_mask,
        )
        
        return warped_depth

class HouseCat6DPoseDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        data_root: str,
        split: str = "train",
        min_num_images: int = 4,
        force_rebuild_cache: bool = False,
        seq_length: int = -1,  # -1 means all sequences
        img_length: int = -1,  # -1 means all images per sequence
        sample_num: int = 1024,  # Number of points to sample for choose indices
    ):
        super().__init__(common_conf=common_conf)
        
        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.load_depth = True
        self.load_nocs = True
        self.depth_scale = 1000.0
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.data_root = data_root
        self.split = split
        self.min_num_images = min_num_images
        self.seq_length = seq_length
        self.img_length = img_length
        self.force_rebuild_cache = force_rebuild_cache
        self.sample_num = sample_num
        self._cache_dir = None
        
        # HouseCat6D specific parameters
        self.xmap = np.array([[i for i in range(1096)] for j in range(852)])
        self.ymap = np.array([[j for i in range(1096)] for j in range(852)])
        self.sym_ids = [1, 2, 7]  # Symmetric object categories
        
        # Load object models
        self.models = self._load_models()

        # Build all annotations first (from all available data)
        all_annotations: Dict[str, List[Dict]] = self._build_all_annotations()
        
        # Filter annotations based on split
        self.annotations: Dict[str, List[Dict]] = self._filter_annotations_by_split(all_annotations)
        self.chunks: Dict[str, List[Dict]] = {}
        self.seq_names: List[str] = []
        
        # Organize annotations into sequences (object-centric)
        total_samples = 0
        for obj_id, entries in self.annotations.items():
            if len(entries) < self.min_num_images:
                continue
            name = f"{obj_id}"
            self.chunks[name] = entries
            self.seq_names.append(name)
            total_samples += len(entries)

        if not self.training or self.split in ["val", "test"]:
            self.len_train = len(self.seq_names)
        else:
            self.len_train = total_samples

        print(f"Found {len(self.seq_names)} object-centric sequences in HouseCat6D {self.split}")
        print(f"Total training samples: {self.len_train}")
        
    def _load_models(self) -> Dict:
        """Load HouseCat6D object models."""
        models = {}
        model_path = os.path.join(self.data_root, 'obj_models_small_size_final/objects.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                models.update(cPickle.load(f))
            print(f'{len(models)} models loaded.')
        else:
            print(f"Warning: Model file not found at {model_path}")
        return models
    
    def _get_cache_path(self) -> str:
        """Generate cache file path based on dataset parameters."""
        # Use "all" since we cache annotations from all splits, mirroring NOCS
        cache_key = f"{self.data_root}_all_{self.min_num_images}_{self.seq_length}_{self.img_length}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        if self._cache_dir is None:
            preferred_dir = os.path.join(self.data_root, "cache")
            self._cache_dir = resolve_cache_dir(preferred_dir, "housecat", self.data_root)
        cache_dir = self._cache_dir
        return os.path.join(cache_dir, f"housecat_annotations_{cache_hash}.npz")
    
    def _is_cache_valid(self) -> bool:
        """Check if cache exists and is valid."""
        cache_path = self._get_cache_path()
        if not os.path.exists(cache_path):
            return False
        
        cache_mtime = os.path.getmtime(cache_path)
        
        # Check if source data is newer than cache
        try:
            split_dir = os.path.join(self.data_root, self.split)
            if os.path.exists(split_dir) and os.path.getmtime(split_dir) > cache_mtime:
                print(f"Cache invalid: {split_dir} newer than cache")
                return False
        except (OSError, IOError) as e:
            print(f"Error checking cache validity: {e}")
            return False
            
        return True
    
    def _load_cached_annotations(self) -> Optional[Dict[str, List[Dict]]]:
        """Load annotations from cache if available."""
        cache_path = self._get_cache_path()
        if os.path.exists(cache_path):
            try:
                print(f"Loading cached annotations from {cache_path}")
                data = np.load(cache_path, allow_pickle=True)
                
                annotations = {}
                for seq_id in data['seq_ids']:
                    annotations[seq_id] = data[f'annotations_{seq_id}'].tolist()
                
                return annotations
            except Exception as e:
                print(f"Failed to load cache: {e}")
        return None
    
    def _save_cached_annotations(self, annotations: Dict[str, List[Dict]]):
        """Save annotations to cache."""
        cache_path = self._get_cache_path()
        try:
            print(f"Saving annotations to cache: {cache_path}")
            
            save_data = {}
            seq_ids = list(annotations.keys())
            save_data['seq_ids'] = np.array(seq_ids, dtype=object)
            
            for seq_id, annotation_list in annotations.items():
                save_data[f'annotations_{seq_id}'] = np.array(annotation_list, dtype=object)
            
            np.savez_compressed(cache_path, **save_data)
            
            cache_size = os.path.getsize(cache_path) / (1024 * 1024)
            print(f"Cache saved successfully. Size: {cache_size:.2f} MB")
            
        except Exception as e:
            print(f"Failed to save cache: {e}")
    
    def _build_all_annotations(self) -> Dict[str, List[Dict]]:
        """Build annotations from HouseCat6D dataset structure for ALL available data.
        Mirrors NOCS logic: build once from both train and test, then filter per split.
        """
        # Check cache first
        if not self.force_rebuild_cache and self._is_cache_valid():
            print("Using valid cached annotations...")
            cached_annotations = self._load_cached_annotations()
            if cached_annotations is not None:
                return cached_annotations

        print("Building annotations from scratch...")
        annotations = defaultdict(list)
        
        # Process both splits: ('train', pattern 'scene*') and ('test', pattern 'test_scene*')
        split_specs = [
            ("train", os.path.join(self.data_root, "train", 'scene*')),
            ("test", os.path.join(self.data_root, "test", 'test_scene*')),
        ]
        
        total_scenes = 0
        for split_name, scene_pattern in split_specs:
            scene_dirs = sorted(glob.glob(scene_pattern))
            if self.seq_length != -1:
                scene_dirs = scene_dirs[:self.seq_length]
            total_scenes += len(scene_dirs)
            print(f"Split {split_name}: found {len(scene_dirs)} scenes")
            
            for scene_dir in tqdm(scene_dirs, desc=f"Processing scenes ({split_name})"):
                scene_name = os.path.basename(scene_dir)
                
                # Load scene intrinsics
                intrinsics_path = os.path.join(scene_dir, 'intrinsics.txt')
                if not os.path.exists(intrinsics_path):
                    continue
                try:
                    intrinsics = np.loadtxt(intrinsics_path).reshape(3, 3)
                except Exception:
                    # Fallback if file contains a single line of 9 numbers
                    intr = np.loadtxt(intrinsics_path)
                    intrinsics = intr.reshape(3, 3)
                
                # Get all RGB images in the scene
                rgb_dir = os.path.join(scene_dir, 'rgb')
                if not os.path.exists(rgb_dir):
                    continue
                
                img_paths = sorted(glob.glob(os.path.join(rgb_dir, '*.png')))
                if self.img_length != -1:
                    img_paths = img_paths[:self.img_length]
                
                # Process each image in the scene
                for img_path in img_paths:
                    frame_id = os.path.basename(img_path).replace('.png', '')
                    
                    # Load labels (ground truth)
                    label_path = img_path.replace('rgb', 'labels').replace('.png', '_label.pkl')
                    if not os.path.exists(label_path):
                        continue
                    
                    with open(label_path, 'rb') as f:
                        gts = cPickle.load(f)
                    
                    # Process each object instance in the frame
                    num_instances = len(gts['instance_ids'])
                    for inst_idx in range(num_instances):
                        # Create object ID combining scene and instance (object-centric sequence)
                        class_id = gts['class_ids'][inst_idx] - 1  # Convert to 0-indexed
                        instance_id = gts['instance_ids'][inst_idx]
                        model_name = gts['model_list'][inst_idx] if 'model_list' in gts else f"model_{class_id}_{instance_id}"
                        obj_id = model_name
                        
                        # Create annotation entry
                        annotation_entry = {
                            "scene": scene_name,
                            "frame_id": frame_id,
                            "paths": {
                                "color": img_path,
                                "depth": img_path,
                                "mask": img_path.replace('rgb', 'instance'),
                                "coord": img_path.replace('rgb', 'nocs'),
                                "label": label_path,
                            },
                            "inst_idx": inst_idx,
                            "inst_id": instance_id,
                            "class_id": class_id,
                            "class_name": model_name.split('-')[0],
                            "bbox": gts['bboxes'][inst_idx],  # [y1, x1, y2, x2]
                            "rotation": gts['rotations'][inst_idx],
                            "translation": gts['translations'][inst_idx],
                            "size": gts.get('gt_scales', [1.0] * num_instances)[inst_idx],
                            "intrinsics": {
                                "fx": float(intrinsics[0, 0]),
                                "fy": float(intrinsics[1, 1]),
                                "cx": float(intrinsics[0, 2]),
                                "cy": float(intrinsics[1, 2]),
                            },
                            "split_source": split_name,
                        }
                        
                        annotations[obj_id].append(annotation_entry)
        
        # Convert to regular dict and sort
        annotations = dict(annotations)
        for obj_id, items in annotations.items():
            annotations[obj_id] = sorted(items, key=lambda e: (e["scene"], e["frame_id"]))
        
        print(f"Found {len(annotations)} total object sequences in HouseCat6D across all splits")
        # Save to cache
        self._save_cached_annotations(annotations)
        return annotations

    def _filter_annotations_by_split(self, all_annotations: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """Filter annotations based on requested split (train uses train source; val/test use test)."""
        if self.split == 'train':
            filtered = {}
            for obj_id, entries in all_annotations.items():
                train_entries = [e for e in entries if e['split_source'] == 'train']
                if len(train_entries) >= self.min_num_images:
                    filtered[obj_id] = train_entries
            print(f"Selected {len(filtered)} object sequences for TRAIN split")
            return filtered
        elif self.split in ['val', 'test']:
            filtered = {}
            for obj_id, entries in all_annotations.items():
                test_entries = [e for e in entries if e['split_source'] == 'test']
                if len(test_entries) >= self.min_num_images:
                    filtered[obj_id] = test_entries
            print(f"Selected {len(filtered)} object sequences for {self.split.upper()} split")
            return filtered
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train', 'val', or 'test'")
    
    def _load_housecat_depth(self, img_path: str) -> np.ndarray:
        """Load depth map for HouseCat6D dataset using utility function."""
        depth = load_housecat_depth(img_path)
        # depth = fill_missing(depth, self.depth_scale, 1)
        return depth

    def _load_housecat_depth_sensor(self, img_path: str) -> np.ndarray:
        """Load depth map for HouseCat6D dataset using utility function."""
        depth = load_housecat_depth_sensor(img_path)
        depth = fill_missing(depth, self.depth_scale, 1)
        return depth
    
    def _quaternion_to_matrix(self, quaternions: torch.Tensor) -> torch.Tensor:
        """Convert quaternions to rotation matrices (if needed)."""
        # HouseCat6D already provides rotation matrices, so this might not be needed
        # But keeping for compatibility with potential quaternion representations
        if quaternions.shape[-1] != 4:
            return quaternions  # Already a matrix
            
        r, i, j, k = torch.unbind(quaternions, -1)
        two_s = 2.0 / (quaternions * quaternions).sum(-1)
        
        o = torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            -1,
        )
        return o.reshape(quaternions.shape[:-1] + (3, 3))
    
    def _camera_from_meta(self, entry: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Extract camera intrinsics and object pose from annotation entry."""
        # Extract camera intrinsics
        intr = entry["intrinsics"]
        fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        
        # Extract object pose (already in matrix form for HouseCat6D)
        R = entry["rotation"].astype(np.float32)
        U, _, Vt = np.linalg.svd(R)
        R_fix = (U @ Vt).astype(np.float32)
        if np.linalg.det(R_fix) < 0:
            U[:, -1] *= -1
            R_fix = (U @ Vt).astype(np.float32)
        R = R_fix
        t = entry["translation"].astype(np.float32)
        
        # Create camera-to-object transform
        c2o = np.eye(4, dtype=np.float32)
        c2o[:3, :3] = R
        c2o[:3, 3] = t
        
        return K, c2o

    def _process_crop_image(self, image, depth, depth_sensor, inst_mask, nocs, T_obj_to_cam, K_original,
                            target_image_shape, viewport_rel_pad=0.1):
        """Process and crop images focused on the object."""
        orig_mask_modal = inst_mask.astype(np.uint8)
        orig_image_np_hwc = image.copy()
        orig_depth_map = depth.copy()
        orig_depth_sensor = depth_sensor.copy()
        
        # Create camera model
        orig_camera_c2w = CameraModel(
            width=image.shape[1],
            height=image.shape[0],
            f=(K_original[0, 0], K_original[1, 1]),
            c=(K_original[0, 2], K_original[1, 2]),
            T_world_from_eye=np.linalg.inv(T_obj_to_cam)
        )
        
        # Get bbox from mask using HouseCat6D utility
        # Note: HouseCat6D images are 852x1096 (height x width)
        rmin, rmax, cmin, cmax = get_bbox_from_mask(orig_mask_modal, img_width=852, img_length=1096)
        # Create AlignedBox2f object for misc functions
        # AlignedBox2f expects (left, top, right, bottom) which is (x1, y1, x2, y2)
        orig_box_amodal = AlignedBox2f(left=cmin, top=rmin, right=cmax, bottom=rmax)
        
        # Get box for cropping
        crop_box = misc.calc_crop_box(
            box=orig_box_amodal,
            make_square=True,
        )
        
        # Construct virtual camera focused on the crop
        crop_camera_model_c2w = misc.construct_crop_camera(
            box=crop_box,
            camera_model_c2w=orig_camera_c2w,
            viewport_size=target_image_shape,
            viewport_rel_pad=viewport_rel_pad,
        )
        
        # Compute warping maps once
        # map_x, map_y, valid_mask = misc.compute_warp_maps(
        #     src_camera=orig_camera_c2w,
        #     dst_camera=crop_camera_model_c2w,
        #     depth_check=True,
        # )

        # # Warp all images using the same maps
        # image_ = misc.warp_image_with_maps(orig_image_np_hwc, map_x, map_y, cv2.INTER_LINEAR)
        # nocs_ = misc.warp_image_with_maps(nocs, map_x, map_y, cv2.INTER_LINEAR)
        # mask_ = misc.warp_image_with_maps(orig_mask_modal, map_x, map_y, cv2.INTER_NEAREST)
        # depth_ = misc.warp_depth_image_with_maps(
        #     orig_camera_c2w, crop_camera_model_c2w, 
        #     orig_depth_map.astype(np.float32), 
        #     map_x, map_y
        # )

        # Warp images to the virtual camera
        image_ = misc.warp_image(
            src_camera=orig_camera_c2w,
            dst_camera=crop_camera_model_c2w,
            src_image=orig_image_np_hwc,
            interpolation=cv2.INTER_LINEAR,
        )
        
        nocs_ = misc.warp_image(
            src_camera=orig_camera_c2w,
            dst_camera=crop_camera_model_c2w,
            src_image=nocs,
            interpolation=cv2.INTER_LINEAR,
        )
        
        mask_ = misc.warp_image(
            src_camera=orig_camera_c2w,
            dst_camera=crop_camera_model_c2w,
            src_image=orig_mask_modal,
            interpolation=cv2.INTER_NEAREST,
        )
        
        depth_ = misc.warp_depth_image(
            src_camera=orig_camera_c2w,
            dst_camera=crop_camera_model_c2w,
            src_depth_image=orig_depth_map.astype(np.float32),
        )

        depth_sensor_ = misc.warp_depth_image(
            src_camera=orig_camera_c2w,
            dst_camera=crop_camera_model_c2w,
            src_depth_image=orig_depth_sensor.astype(np.float32),
        )

        # Extract new camera parameters
        extri_ = np.linalg.inv(crop_camera_model_c2w.T_world_from_eye)
        fx, fy = crop_camera_model_c2w.f
        cx, cy = crop_camera_model_c2w.c
        intri_ = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape(3, 3)
        crop_info = misc.crop_box_to_normalized_features(crop_box, image.shape[:2])
        return image_, depth_.astype(np.float16), depth_sensor_.astype(np.float16), mask_, nocs_, extri_, intri_, crop_info

    def _process_crop_image_torch(self, image, depth, depth_sensor, inst_mask, nocs, T_obj_to_cam, K_original,
        target_image_shape, viewport_rel_pad=0.1, device='cuda'
    ):
        """
        Process and crop images focused on the object using PyTorch + CUDA
        
        Args:
            image: [H, W, 3] numpy array
            depth: [H, W] numpy array
            depth_sensor: [H, W] numpy array
            inst_mask: [H, W] numpy array
            nocs: [H, W, 3] numpy array
            T_obj_to_cam: [4, 4] numpy array
            K_original: [3, 3] numpy array
            target_image_shape: (width, height)
            viewport_rel_pad: float
            device: 'cuda' or 'cpu'
        
        Returns:
            image_, depth_, mask_, nocs_, extri_, intri_
        """
        # Initialize warper (you can make this a class member to avoid reinitializing)
        if not hasattr(self, 'warper'):
            self.warper = CameraWarper(device=device)
        
        orig_mask_modal = inst_mask.astype(np.uint8)
        orig_image_np_hwc = image.copy()
        orig_depth_map = depth.copy()
        
        # Create camera model
        orig_camera_c2w = CameraModel(
            width=image.shape[1],
            height=image.shape[0],
            f=(K_original[0, 0], K_original[1, 1]),
            c=(K_original[0, 2], K_original[1, 2]),
            T_world_from_eye=np.linalg.inv(T_obj_to_cam)
        )
        
        # Get bbox from mask
        rmin, rmax, cmin, cmax = get_bbox_from_mask(orig_mask_modal, img_width=852, img_length=1096)
        orig_box_amodal = AlignedBox2f(left=cmin, top=rmin, right=cmax, bottom=rmax)
        
        # Get box for cropping
        crop_box = misc.calc_crop_box(
            box=orig_box_amodal,
            make_square=True,
        )
        
        # Construct virtual camera focused on the crop
        crop_camera_model_c2w = misc.construct_crop_camera(
            box=crop_box,
            camera_model_c2w=orig_camera_c2w,
            viewport_size=target_image_shape,
            viewport_rel_pad=viewport_rel_pad,
        )
        
        # Compute warping grid once (this is the key optimization)
        grid, valid_mask = self.warper.compute_warp_grid_torch(
            src_camera=orig_camera_c2w,
            dst_camera=crop_camera_model_c2w,
            use_half_pixel=True,
            depth_check=True,
        )
        
        # Warp all images using the same grid
        image_ = self.warper.warp_image_torch(
            orig_image_np_hwc,
            grid,
            mode='bilinear',
            mask_invalid=False,
            valid_mask=valid_mask,
        )
        
        nocs_ = self.warper.warp_image_torch(
            nocs,
            grid,
            mode='bilinear',
            mask_invalid=False,
            valid_mask=valid_mask,
        )
        
        mask_ = self.warper.warp_image_torch(
            orig_mask_modal,
            grid,
            mode='nearest',
            mask_invalid=True,
            valid_mask=valid_mask,
        )
        
        depth_ = self.warper.warp_depth_torch(
            src_camera=orig_camera_c2w,
            dst_camera=crop_camera_model_c2w,
            src_depth=orig_depth_map.astype(np.float32),
            grid=grid,
            valid_mask=valid_mask,
        )

        depth_sensor_ = self.warper.warp_depth_torch(
            src_camera=orig_camera_c2w,
            dst_camera=crop_camera_model_c2w,
            src_depth=depth_sensor.astype(np.float32),
            grid=grid,
            valid_mask=valid_mask,
        )

        # Extract new camera parameters
        extri_ = np.linalg.inv(crop_camera_model_c2w.T_world_from_eye)
        fx, fy = crop_camera_model_c2w.f
        cx, cy = crop_camera_model_c2w.c
        intri_ = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape(3, 3)
        
        # Convert back to numpy if needed
        image_ = image_.permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # [H, W, 3]
        nocs_ = nocs_.permute(1, 2, 0).cpu().numpy().astype(np.float32)  # [H, W, 3]
        mask_ = mask_.cpu().numpy().astype(np.uint8)    # [H, W]
        depth_ = depth_.cpu().numpy().astype(np.float16)  # [H, W]
        depth_sensor_ = depth_sensor_.cpu().numpy().astype(np.float16)  # [H, W]
        
        crop_info = misc.crop_box_to_normalized_features(crop_box, image.shape[:2])

        return image_, depth_, depth_sensor_, mask_, nocs_, extri_, intri_, crop_info

    def _apply_symmetry_transform(self, rotation, nocs, cat_id):
        """Apply symmetry transformation for symmetric objects."""
        if cat_id in self.sym_ids:
            theta_x = rotation[0, 0] + rotation[2, 2]
            theta_y = rotation[0, 2] - rotation[2, 0]
            r_norm = math.sqrt(theta_x**2 + theta_y**2)
            if r_norm > 0:
                s_map = np.array([
                    [theta_x/r_norm, 0.0, -theta_y/r_norm],
                    [0.0, 1.0, 0.0],
                    [theta_y/r_norm, 0.0, theta_x/r_norm]
                ])
                rotation = rotation @ s_map
                nocs = nocs @ s_map
        return rotation, nocs
    
    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        """Get data for a sequence of images."""
        if self.inside_random:
            seq_index = np.random.randint(0, len(self.seq_names))
        
        # Robustly clamp/wrap seq_index (align with NOCS behavior)
        if seq_index is None and seq_name is None:
            seq_index = np.random.randint(0, len(self.seq_names))
        if seq_name is None:
            seq_index = int(seq_index) % max(1, len(self.seq_names))
            seq_name = self.seq_names[seq_index]
        
        entries = self.chunks[seq_name]
        
        if ids is None:
            ids = np.random.choice(
                len(entries), img_per_seq, replace=self.allow_duplicate_img
            )
        
        target_image_shape = self.get_target_shape(aspect_ratio)
        
        # Initialize output lists
        images, depths, depths_sensor, inst_masks, nocs_list = [], [], [], [], []
        extrinsics, intrinsics, extrinsics_sym = [], [], []
        world_points, cam_points, point_masks = [], [], []
        ori_sizes, filepaths, sizes = [], [], []
        choose_indices = []  # Random sampled indices
        models = []
        crop_boxes = []
        
        def process_record(e: Dict) -> Optional[tuple]:
            """Process a single record."""
            # Load image data
            image = cv2.imread(e["paths"]["color"])[:, :, ::-1]  # BGR to RGB, shape (852, 1096, 3)
            depth = self._load_housecat_depth(e["paths"]["color"]) / self.depth_scale
            depth_sensor = self._load_housecat_depth_sensor(e["paths"]["color"]) / self.depth_scale

            # Load mask - HouseCat6D stores instance masks in the blue channel
            mask = cv2.imread(e["paths"]["mask"])[:, :, 2]  # Get blue channel
            inst_mask = np.equal(mask, e["inst_id"])
            inst_mask = np.logical_and(inst_mask, depth > 0)
            
            # Load NOCS coordinates
            coord = cv2.imread(e["paths"]["coord"])[:, :, :3]
            coord = np.array(coord, dtype=np.float32) / 255.0
            nocs = coord - 0.5  # Center NOCS coordinates
            
            # Apply mask
            if inst_mask.sum() < 1024:
                return None

            # Threshold/prune depth values similarly to NOCS
            depth = threshold_depth_map(depth, min_percentile=1, max_percentile=99)
            depth_sensor = threshold_depth_map(depth_sensor, min_percentile=1, max_percentile=99)

            image[~inst_mask] = 0
            nocs[~inst_mask] = 0
            depth[~inst_mask] = 0
            depth_sensor[~inst_mask] = 0
            
            # Get camera parameters and object pose
            intri, extri = self._camera_from_meta(e)
                       
            # Store original size
            original_size = np.array(image.shape[:2])

            size = e['size']
            
            # Process and crop image
            image_, depth_, depth_sensor_, inst_mask_, nocs_, extri_, intri_, crop_info = self._process_crop_image(
                image, depth, depth_sensor, inst_mask, nocs, extri, intri,
                target_image_shape=target_image_shape,
                viewport_rel_pad=0.1,
            )

            # image_, depth_, depth_sensor_, inst_mask_, nocs_, extri_, intri_ = self._process_crop_image_torch(
            #     image, depth, depth_sensor, inst_mask, nocs, extri, intri,
            #     target_image_shape=target_image_shape,
            #     viewport_rel_pad=0.1,
            #     device='cuda'
            # )

            # Apply symmetry transformation if needed
            rotation_sym, nocs_ = self._apply_symmetry_transform(
                extri_[:3, :3], nocs_, e["class_id"]
            )
            extri_sym = extri_.copy()
            extri_sym[:3, :3] = rotation_sym

            # Convert depth to world and camera coordinates
            world_coords_points, cam_coords_points, point_mask = (
                depth_to_world_coords_points(depth_, extri_sym, intri_)
            )
            
            H, W = inst_mask_.shape[:2]
            mask_flat = inst_mask_.flatten()
            choose = mask_flat.nonzero()[0]
            
            if len(choose) <= 0:
                # If no valid points, return None to skip this frame
                return None
            elif len(choose) <= self.sample_num:
                choose_idx = np.random.choice(np.arange(len(choose)), self.sample_num)
            else:
                choose_idx = np.random.choice(np.arange(len(choose)), self.sample_num, replace=False)
            choose = choose[choose_idx].astype(np.int64)
            model = self.models[seq_name]
            
            return (
                image_.astype(np.uint8).copy(),
                depth_.astype(np.float32).copy() if depth_ is not None else np.zeros(image_.shape[:2], np.float32),
                depth_sensor_.astype(np.float16).copy() if depth_sensor_ is not None else np.zeros(image_.shape[:2], np.float16),
                inst_mask_.astype(np.uint8).copy(),
                nocs_.astype(np.float16).copy(),
                extri_[:3, :].astype(np.float16).copy(),
                extri_sym[:3, :].astype(np.float16).copy(), # sym-canonicalized pose
                intri_.astype(np.float16).copy(),
                world_coords_points.astype(np.float16).copy(),
                cam_coords_points.astype(np.float16).copy(),
                point_mask.astype(np.uint8).copy(),
                original_size,
                e["paths"]["color"],
                size.astype(np.float16).copy(),
                choose,  # Random sampled indices
                model.astype(np.float16).copy(),
                crop_info.astype(np.float32).copy(),
            )
        
        # Process selected frames
        valid_items = []
        selected_indices = []
        for i in ids:
            packed = process_record(entries[int(i)])
            if packed is not None:
                valid_items.append(packed)
                selected_indices.append(int(i))
        
        # If not enough valid items, sample more (no scale consistency needed for HouseCat)
        tries = 0
        while len(valid_items) < len(ids) and tries < len(entries) * 2:
            rnd = int(np.random.choice(len(entries)))
            packed = process_record(entries[rnd])
            if packed is not None:
                valid_items.append(packed)
                selected_indices.append(rnd)
            tries += 1
        
        # Stack outputs
        for item in valid_items[:len(ids)]:
            (image, depth, depth_sensor, inst_mask, nocs_item, extri, extri_sym, intri,
             world_coords_points, cam_coords_points, point_mask,
             original_size, filepath, size, choose, model, crop_info) = item
            
            images.append(image)
            depths.append(depth)
            depths_sensor.append(depth_sensor)
            inst_masks.append(inst_mask)
            nocs_list.append(nocs_item)
            extrinsics.append(extri)
            extrinsics_sym.append(extri_sym)
            intrinsics.append(intri)
            world_points.append(world_coords_points)
            cam_points.append(cam_coords_points)
            point_masks.append(point_mask)
            ori_sizes.append(original_size)
            filepaths.append(filepath)
            sizes.append(size)
            choose_indices.append(choose)
            models.append(model)
            crop_boxes.append(crop_info)

        # Update ids
        if len(selected_indices) >= len(ids):
            ids = np.asarray(selected_indices[:len(ids)], dtype=np.int32)
        else:
            ids = np.asarray(ids, dtype=np.int32)
        
        # Create deterministic seq_id for instance consistency across DDP workers.
        seq_id = stable_seq_id(seq_name)
        
        batch = {
            "seq_name": seq_name,
            "seq_id": seq_id,
            "cat_name": seq_name.split("-")[0],
            "ids": ids,
            "frame_num": len(images),
            "images": images,
            "depths": depths,
            "depths_sensor": depths_sensor,
            "inst_masks": inst_masks,
            "nocs": nocs_list,
            "extrinsics": extrinsics,
            "extrinsics_sym": extrinsics_sym,
            "intrinsics": intrinsics,
            "world_points": world_points,
            "cam_points": cam_points,
            "point_masks": point_masks,
            "original_sizes": ori_sizes,
            "filepaths": filepaths,
            "sizes": sizes,
            "choose_indices": choose_indices,
            "crop_boxes": crop_boxes,
            "models": models,
        }
        if len(batch["images"]) == 0:
            print(f"Empty batch at {filepaths}")

        return batch


class HouseCat6DTestDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        data_root: str,
        split: str = "test",
        min_num_images: int = 4,
        force_rebuild_cache: bool = True,
        seq_length: int = -1,  # -1 means all sequences
        img_length: int = -1,  # -1 means all images per sequence
        sample_num: int = 1024,  # Number of points to sample for choose indices
        visibility_threshold: float = 0.001,  # Minimum visibility ratio for objects
    ):
        super().__init__(common_conf=common_conf)
        
        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.load_depth = True
        self.load_nocs = True
        self.depth_scale = 1000.0
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.data_root = data_root
        self.split = split
        self.min_num_images = min_num_images
        self.seq_length = seq_length
        self.img_length = img_length
        self.force_rebuild_cache = force_rebuild_cache
        self.sample_num = sample_num
        self.visibility_threshold = visibility_threshold
        self._cache_dir = None
        
        # HouseCat6D specific parameters
        self.xmap = np.array([[i for i in range(1096)] for j in range(852)])
        self.ymap = np.array([[j for i in range(1096)] for j in range(852)])
        self.sym_ids = [1, 2, 7]  # Symmetric object categories
        
        # Load object models
        self.models = self._load_models()

        # Build annotations organized by unique object instances
        self.annotations: Dict[str, List[Dict]] = self._build_frame_annotations()
        self.unique_keys: List[str] = []
        
        # Each unique key represents one object instance
        self.unique_keys = list(self.annotations.keys())
        
        # For testing, we want each object to be a separate sample
        self.len_train = len(self.unique_keys)

        print(f"Found {len(self.unique_keys)} unique object instances in HouseCat6D {self.split}")
        print(f"Total test samples (objects): {self.len_train}")
        
    def _load_models(self) -> Dict:
        """Load HouseCat6D object models."""
        models = {}
        model_path = os.path.join(self.data_root, 'obj_models_small_size_final/objects.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                models.update(cPickle.load(f))
            print(f'{len(models)} models loaded.')
        else:
            print(f"Warning: Model file not found at {model_path}")
        return models
    
    def _get_cache_path(self) -> str:
        """Generate cache file path based on dataset parameters."""
        # Create a unique cache key for the test dataset to avoid conflicts with training dataset
        cache_key = f"{self.data_root}_test_dataset_frames_{self.min_num_images}_{self.seq_length}_{self.img_length}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        if self._cache_dir is None:
            preferred_dir = os.path.join(self.data_root, "cache")
            self._cache_dir = resolve_cache_dir(preferred_dir, "housecat_test", self.data_root)
        cache_dir = self._cache_dir
        return os.path.join(cache_dir, f"housecat_test_dataset_frames_{cache_hash}.npz")
    
    def _is_cache_valid(self) -> bool:
        """Check if cache exists and is valid."""
        cache_path = self._get_cache_path()
        if not os.path.exists(cache_path):
            return False
        
        cache_mtime = os.path.getmtime(cache_path)
        
        # Check if source data is newer than cache
        try:
            # Check the specified split directory
            split_dir = os.path.join(self.data_root, self.split)
            if os.path.exists(split_dir) and os.path.getmtime(split_dir) > cache_mtime:
                print(f"Test dataset cache invalid: {split_dir} newer than cache")
                return False
            
            # Also check if any scene directories are newer
            if self.split == 'test':
                scenes_pattern = os.path.join(split_dir, 'test_scene*')
            else:
                scenes_pattern = os.path.join(split_dir, 'scene*')
            scene_dirs = glob.glob(scenes_pattern)
            for scene_dir in scene_dirs:
                if os.path.exists(scene_dir) and os.path.getmtime(scene_dir) > cache_mtime:
                    print(f"Test dataset cache invalid: {scene_dir} newer than cache")
                    return False
                    
        except (OSError, IOError) as e:
            print(f"Error checking test dataset cache validity: {e}")
            return False
            
        return True
    
    def _load_cached_annotations(self) -> Optional[Dict[str, List[Dict]]]:
        """Load annotations from cache if available."""
        cache_path = self._get_cache_path()
        if os.path.exists(cache_path):
            try:
                print(f"Loading cached test dataset frame annotations from {cache_path}")
                data = np.load(cache_path, allow_pickle=True)
                
                # Validate cache structure and type
                if 'unique_keys' not in data:
                    print("Invalid test dataset cache: missing 'unique_keys'")
                    return None
                
                # Check if this is actually a test dataset cache
                if 'dataset_type' in data:
                    dataset_type = data['dataset_type'].tolist()
                    if dataset_type != ['test_dataset']:
                        print(f"Invalid test dataset cache: wrong dataset type '{dataset_type}'")
                        return None
                
                annotations = {}
                for unique_key in data['unique_keys']:
                    annotation_key = f'annotations_{unique_key}'
                    if annotation_key not in data:
                        print(f"Invalid test dataset cache: missing '{annotation_key}'")
                        return None
                    annotations[unique_key] = data[annotation_key].tolist()
                
                print(f"Successfully loaded {len(annotations)} unique object instances from test dataset cache")
                return annotations
            except Exception as e:
                print(f"Failed to load test dataset cache: {e}")
                # Remove corrupted cache file
                try:
                    os.remove(cache_path)
                    print(f"Removed corrupted cache file: {cache_path}")
                except:
                    pass
        return None
    
    def _save_cached_annotations(self, annotations: Dict[str, List[Dict]]):
        """Save annotations to cache."""
        cache_path = self._get_cache_path()
        try:
            print(f"Saving test dataset frame annotations to cache: {cache_path}")
            
            # Create temporary file first to avoid corruption
            temp_cache_path = cache_path + ".tmp"
            
            save_data = {}
            unique_keys = list(annotations.keys())
            save_data['unique_keys'] = np.array(unique_keys, dtype=object)
            
            # Add metadata to identify this as a test dataset cache
            save_data['dataset_type'] = np.array(['test_dataset'], dtype=object)
            save_data['split'] = np.array([self.split], dtype=object)
            
            for unique_key, annotation_list in annotations.items():
                save_data[f'annotations_{unique_key}'] = np.array(annotation_list, dtype=object)
            
            # Save to temporary file first
            np.savez_compressed(temp_cache_path, **save_data)
            
            # Move to final location
            os.rename(temp_cache_path, cache_path)
            
            cache_size = os.path.getsize(cache_path) / (1024 * 1024)
            print(f"Test dataset cache saved successfully. Size: {cache_size:.2f} MB")
            print(f"  - Cached {len(annotations)} unique object instances")
            print(f"  - Each instance has its own unique key")
            
        except Exception as e:
            print(f"Failed to save test dataset cache: {e}")
            # Clean up temporary file if it exists
            temp_cache_path = cache_path + ".tmp"
            if os.path.exists(temp_cache_path):
                try:
                    os.remove(temp_cache_path)
                except:
                    pass
    
    def clear_cache(self):
        """Clear the test dataset cache."""
        cache_path = self._get_cache_path()
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
                print(f"Cleared test dataset cache: {cache_path}")
            except Exception as e:
                print(f"Failed to clear test dataset cache: {e}")
    
    def _build_frame_annotations(self) -> Dict[str, List[Dict]]:
        """Build annotations organized by image frame (not object sequence)."""
        # Check cache first
        if not self.force_rebuild_cache and self._is_cache_valid():
            print("Using valid cached test dataset annotations...")
            cached_annotations = self._load_cached_annotations()
            if cached_annotations is not None:
                print(f"Loaded {len(cached_annotations)} unique object instances from cache")
                return cached_annotations
            else:
                print("Cache loading failed, rebuilding from scratch...")

        print("Building test dataset frame annotations from scratch...")
        annotations = defaultdict(list)
        
        # Process the specified split for test dataset
        scene_pattern = os.path.join(self.data_root, self.split, 'test_scene*')
        scene_dirs = sorted(glob.glob(scene_pattern))
        if self.seq_length != -1:
            scene_dirs = scene_dirs[:self.seq_length]
        print(f"Found {len(scene_dirs)} scenes in {self.split} split")
        
        for scene_dir in tqdm(scene_dirs, desc=f"Processing {self.split} scenes"):
            scene_name = os.path.basename(scene_dir)
            
            # Load scene intrinsics
            intrinsics_path = os.path.join(scene_dir, 'intrinsics.txt')
            if not os.path.exists(intrinsics_path):
                continue
            try:
                intrinsics = np.loadtxt(intrinsics_path).reshape(3, 3)
            except Exception:
                # Fallback if file contains a single line of 9 numbers
                intr = np.loadtxt(intrinsics_path)
                intrinsics = intr.reshape(3, 3)
            
            # Get all RGB images in the scene
            rgb_dir = os.path.join(scene_dir, 'rgb')
            if not os.path.exists(rgb_dir):
                continue
            
            img_paths = sorted(glob.glob(os.path.join(rgb_dir, '*.png')))
            if self.img_length != -1:
                img_paths = img_paths[:self.img_length]
            
            # Process each image in the scene
            for img_path in img_paths:
                frame_id = os.path.basename(img_path).replace('.png', '')
                
                # Load labels (ground truth)
                label_path = img_path.replace('rgb', 'labels').replace('.png', '_label.pkl')
                if not os.path.exists(label_path):
                    continue
                
                with open(label_path, 'rb') as f:
                    gts = cPickle.load(f)
                
                # Create unique key for this frame: scene_frame_id
                unique_key = f"{scene_name}_{frame_id}"
                
                # Process all object instances in the frame
                num_instances = len(gts['instance_ids'])
                frame_objects = []
                
                for inst_idx in range(num_instances):
                    # Create annotation entry for this object in this frame
                    annotation_entry = {
                        "scene": scene_name,
                        "frame_id": frame_id,
                        "inst_idx": inst_idx,
                        "unique_key": unique_key,
                        "paths": {
                            "color": img_path,
                            "depth": img_path,
                            "mask": img_path.replace('rgb', 'instance'),
                            "coord": img_path.replace('rgb', 'nocs'),
                            "label": label_path,
                        },
                        "inst_id": gts['instance_ids'][inst_idx],
                        "class_id": gts['class_ids'][inst_idx] - 1,  # Convert to 0-indexed
                        "class_name": gts['model_list'][inst_idx].split('-')[0] if 'model_list' in gts else f"model_{gts['class_ids'][inst_idx]-1}",
                        "bbox": gts['bboxes'][inst_idx],  # [y1, x1, y2, x2]
                        "rotation": gts['rotations'][inst_idx],
                        "translation": gts['translations'][inst_idx],
                        "size": gts.get('gt_scales', [1.0] * num_instances)[inst_idx],
                        "intrinsics": {
                            "fx": float(intrinsics[0, 0]),
                            "fy": float(intrinsics[1, 1]),
                            "cx": float(intrinsics[0, 2]),
                            "cy": float(intrinsics[1, 2]),
                        },
                    }
                    
                    frame_objects.append(annotation_entry)
                
                # Store all objects for this frame under the frame key
                annotations[unique_key] = frame_objects
        
        # Convert to regular dict
        annotations = dict(annotations)
        
        print(f"Built test dataset annotations:")
        print(f"  - Found {len(annotations)} unique frames in HouseCat6D {self.split} split")
        print(f"  - Each frame has a unique key: scene_frame_id")
        total_objects = sum(len(objects) for objects in annotations.values())
        print(f"  - Total objects across all frames: {total_objects}")
        
        if len(annotations) == 0:
            print(f"Warning: No frames found in {self.split} dataset!")
            return annotations
        
        # Save to cache
        self._save_cached_annotations(annotations)
        return annotations

    def _load_housecat_depth(self, img_path: str) -> np.ndarray:
        """Load depth map for HouseCat6D dataset using utility function."""
        depth = load_housecat_depth(img_path)
        return depth

    def _load_housecat_depth_sensor(self, img_path: str) -> np.ndarray:
        """Load depth map for HouseCat6D dataset using utility function."""
        depth = load_housecat_depth_sensor(img_path)
        depth = fill_missing(depth, self.depth_scale, 1)
        return depth

    def _quaternion_to_matrix(self, quaternions: torch.Tensor) -> torch.Tensor:
        """Convert quaternions to rotation matrices (if needed)."""
        if quaternions.shape[-1] != 4:
            return quaternions  # Already a matrix
            
        r, i, j, k = torch.unbind(quaternions, -1)
        two_s = 2.0 / (quaternions * quaternions).sum(-1)
        
        o = torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            -1,
        )
        return o.reshape(quaternions.shape[:-1] + (3, 3))
    
    def _camera_from_meta(self, entry: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Extract camera intrinsics and object pose from annotation entry."""
        # Extract camera intrinsics
        intr = entry["intrinsics"]
        fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        
        # Extract object pose (already in matrix form for HouseCat6D)
        R = entry["rotation"].astype(np.float32)
        U, _, Vt = np.linalg.svd(R)
        R_fix = (U @ Vt).astype(np.float32)
        if np.linalg.det(R_fix) < 0:
            U[:, -1] *= -1
            R_fix = (U @ Vt).astype(np.float32)
        R = R_fix
        t = entry["translation"].astype(np.float32)
        
        # Create camera-to-object transform
        c2o = np.eye(4, dtype=np.float32)
        c2o[:3, :3] = R
        c2o[:3, 3] = t
        
        return K, c2o

    def _process_crop_image(self, image, depth, depth_sensor, inst_mask, nocs, T_obj_to_cam, K_original,
                            target_image_shape, viewport_rel_pad=0.1):
        """Process and crop images focused on the object."""
        orig_mask_modal = inst_mask.astype(np.uint8)
        orig_image_np_hwc = image.copy()
        orig_depth_map = depth.copy()
        orig_depth_sensor = depth_sensor.copy()
        # Create camera model
        orig_camera_c2w = CameraModel(
            width=image.shape[1],
            height=image.shape[0],
            f=(K_original[0, 0], K_original[1, 1]),
            c=(K_original[0, 2], K_original[1, 2]),
            T_world_from_eye=np.linalg.inv(T_obj_to_cam)
        )
        # Get bbox from mask using HouseCat6D utility
        rmin, rmax, cmin, cmax = get_bbox_from_mask(orig_mask_modal, img_width=852, img_length=1096)
        # Create AlignedBox2f object for misc functions
        orig_box_amodal = AlignedBox2f(left=cmin, top=rmin, right=cmax, bottom=rmax)
        
        # Get box for cropping
        crop_box = misc.calc_crop_box(
            box=orig_box_amodal,
            make_square=True,
        )
        # Construct virtual camera focused on the crop
        crop_camera_model_c2w = misc.construct_crop_camera(
            box=crop_box,
            camera_model_c2w=orig_camera_c2w,
            viewport_size=target_image_shape,
            viewport_rel_pad=viewport_rel_pad,
        )

        # Compute warping maps once
        map_x, map_y, valid_mask = misc.compute_warp_maps(
            src_camera=orig_camera_c2w,
            dst_camera=crop_camera_model_c2w,
            depth_check=True,
        )

        # Warp all images using the same maps
        image_ = misc.warp_image_with_maps(orig_image_np_hwc, map_x, map_y, cv2.INTER_LINEAR)
        nocs_ = misc.warp_image_with_maps(nocs, map_x, map_y, cv2.INTER_LINEAR)
        mask_ = misc.warp_image_with_maps(orig_mask_modal, map_x, map_y, cv2.INTER_NEAREST)
        depth_ = misc.warp_depth_image_with_maps(
            orig_camera_c2w, crop_camera_model_c2w, 
            orig_depth_map.astype(np.float32), 
            map_x, map_y
        )
        depth_sensor_ = misc.warp_depth_image_with_maps(
            orig_camera_c2w, crop_camera_model_c2w, 
            orig_depth_sensor.astype(np.float32), 
            map_x, map_y
        )

        # Warp images to the virtual camera
        # image_ = misc.warp_image(
        #     src_camera=orig_camera_c2w,
        #     dst_camera=crop_camera_model_c2w,
        #     src_image=orig_image_np_hwc,
        #     interpolation=cv2.INTER_LINEAR,
        # )
        # nocs_ = misc.warp_image(
        #     src_camera=orig_camera_c2w,
        #     dst_camera=crop_camera_model_c2w,
        #     src_image=nocs,
        #     interpolation=cv2.INTER_LINEAR,
        # )
        # mask_ = misc.warp_image(
        #     src_camera=orig_camera_c2w,
        #     dst_camera=crop_camera_model_c2w,
        #     src_image=orig_mask_modal,
        #     interpolation=cv2.INTER_NEAREST,
        # )
        # depth_ = misc.warp_depth_image(
        #     src_camera=orig_camera_c2w,
        #     dst_camera=crop_camera_model_c2w,
        #     src_depth_image=orig_depth_map.astype(np.float32),
        # )
        # Extract new camera parameters
        extri_ = np.linalg.inv(crop_camera_model_c2w.T_world_from_eye)
        fx, fy = crop_camera_model_c2w.f
        cx, cy = crop_camera_model_c2w.c
        intri_ = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape(3, 3)
        
        return image_, depth_.astype(np.float16), depth_sensor_.astype(np.float16), mask_, nocs_, extri_, intri_
    
    def _process_crop_image_torch(self, image, depth, depth_sensor, inst_mask, nocs, T_obj_to_cam, K_original,
        target_image_shape, viewport_rel_pad=0.1, device='cuda'
    ):
        """
        Process and crop images focused on the object using PyTorch + CUDA
        
        Args:
            image: [H, W, 3] numpy array
            depth: [H, W] numpy array
            depth_sensor: [H, W] numpy array
            inst_mask: [H, W] numpy array
            nocs: [H, W, 3] numpy array
            T_obj_to_cam: [4, 4] numpy array
            K_original: [3, 3] numpy array
            target_image_shape: (width, height)
            viewport_rel_pad: float
            device: 'cuda' or 'cpu'
        
        Returns:
            image_, depth_, mask_, nocs_, extri_, intri_
        """
        # Initialize warper (you can make this a class member to avoid reinitializing)
        if not hasattr(self, 'warper'):
            self.warper = CameraWarper(device=device)
        
        orig_mask_modal = inst_mask.astype(np.uint8)
        orig_image_np_hwc = image.copy()
        orig_depth_map = depth.copy()
        
        # Create camera model
        orig_camera_c2w = CameraModel(
            width=image.shape[1],
            height=image.shape[0],
            f=(K_original[0, 0], K_original[1, 1]),
            c=(K_original[0, 2], K_original[1, 2]),
            T_world_from_eye=np.linalg.inv(T_obj_to_cam)
        )
        
        # Get bbox from mask
        rmin, rmax, cmin, cmax = get_bbox_from_mask(orig_mask_modal, img_width=852, img_length=1096)
        orig_box_amodal = AlignedBox2f(left=cmin, top=rmin, right=cmax, bottom=rmax)
        
        # Get box for cropping
        crop_box = misc.calc_crop_box(
            box=orig_box_amodal,
            make_square=True,
        )
        
        # Construct virtual camera focused on the crop
        crop_camera_model_c2w = misc.construct_crop_camera(
            box=crop_box,
            camera_model_c2w=orig_camera_c2w,
            viewport_size=target_image_shape,
            viewport_rel_pad=viewport_rel_pad,
        )
        
        # Compute warping grid once (this is the key optimization)
        grid, valid_mask = self.warper.compute_warp_grid_torch(
            src_camera=orig_camera_c2w,
            dst_camera=crop_camera_model_c2w,
            use_half_pixel=True,
            depth_check=True,
        )
        
        # Warp all images using the same grid
        image_ = self.warper.warp_image_torch(
            orig_image_np_hwc,
            grid,
            mode='bilinear',
            mask_invalid=False,
            valid_mask=valid_mask,
        )
        
        nocs_ = self.warper.warp_image_torch(
            nocs,
            grid,
            mode='bilinear',
            mask_invalid=False,
            valid_mask=valid_mask,
        )
        
        mask_ = self.warper.warp_image_torch(
            orig_mask_modal,
            grid,
            mode='nearest',
            mask_invalid=True,
            valid_mask=valid_mask,
        )
        
        depth_ = self.warper.warp_depth_torch(
            src_camera=orig_camera_c2w,
            dst_camera=crop_camera_model_c2w,
            src_depth=orig_depth_map.astype(np.float32),
            grid=grid,
            valid_mask=valid_mask,
        )

        depth_sensor_ = self.warper.warp_depth_torch(
            src_camera=orig_camera_c2w,
            dst_camera=crop_camera_model_c2w,
            src_depth=depth_sensor.astype(np.float32),
            grid=grid,
            valid_mask=valid_mask,
        )

        # Extract new camera parameters
        extri_ = np.linalg.inv(crop_camera_model_c2w.T_world_from_eye)
        fx, fy = crop_camera_model_c2w.f
        cx, cy = crop_camera_model_c2w.c
        intri_ = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape(3, 3)
        
        # Convert back to numpy if needed
        image_ = image_.permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # [H, W, 3]
        nocs_ = nocs_.permute(1, 2, 0).cpu().numpy().astype(np.float32)  # [H, W, 3]
        mask_ = mask_.cpu().numpy().astype(np.uint8)    # [H, W]
        depth_ = depth_.cpu().numpy().astype(np.float16)  # [H, W]
        depth_sensor_ = depth_sensor_.cpu().numpy().astype(np.float16)  # [H, W]
        
        return image_, depth_, depth_sensor_, mask_, nocs_, extri_, intri_

    def _apply_symmetry_transform(self, rotation, nocs, cat_id):
        """Apply symmetry transformation for symmetric objects."""
        if cat_id in self.sym_ids:
            theta_x = rotation[0, 0] + rotation[2, 2]
            theta_y = rotation[0, 2] - rotation[2, 0]
            r_norm = math.sqrt(theta_x**2 + theta_y**2)
            if r_norm > 0:
                s_map = np.array([
                    [theta_x/r_norm, 0.0, -theta_y/r_norm],
                    [0.0, 1.0, 0.0],
                    [theta_y/r_norm, 0.0, theta_x/r_norm]
                ])
                rotation = rotation @ s_map
                nocs = nocs @ s_map
        return rotation, nocs
    
    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
        sample_index: int = None,
    ) -> dict:
        """Get data for a single object sample (not a sequence)."""
        # For test dataset, use sample_index if provided, otherwise use seq_index
        if sample_index is not None:
            pass  # Use the provided sample_index
        elif seq_index is not None:
            sample_index = seq_index
        else:
            sample_index = np.random.randint(0, self.len_train)
        
        # Find which unique frame this sample corresponds to
        if sample_index >= len(self.unique_keys):
            # Fallback to last frame if index is out of bounds
            unique_key = self.unique_keys[-1]
        else:
            unique_key = self.unique_keys[sample_index]
        
        # Get all object annotations for this frame
        frame_objects = self.annotations[unique_key]
        
        target_image_shape = self.get_target_shape(aspect_ratio)
        
        # Load shared image data (same for all objects in the frame)
        first_obj = frame_objects[0]
        image = cv2.imread(first_obj["paths"]["color"])[:, :, ::-1]  # BGR to RGB
        depth = self._load_housecat_depth(first_obj["paths"]["color"]) / self.depth_scale
        depth_sensor = self._load_housecat_depth_sensor(first_obj["paths"]["color"]) / self.depth_scale
        
        # Load mask - HouseCat6D stores instance masks in the blue channel
        mask = cv2.imread(first_obj["paths"]["mask"])[:, :, 2]  # Get blue channel
        
        # Load NOCS coordinates
        coord = cv2.imread(first_obj["paths"]["coord"])[:, :, :3]
        coord = np.array(coord, dtype=np.float32) / 255.0
        nocs = coord - 0.5  # Center NOCS coordinates
        
        # Threshold/prune depth values
        # depth = threshold_depth_map(depth, min_percentile=1, max_percentile=99)
        # depth_sensor = threshold_depth_map(depth_sensor, min_percentile=1, max_percentile=99)
        
        # Initialize output lists for all objects in the frame
        images, depths, depths_sensor, inst_masks, nocs_list = [], [], [], [], []
        extrinsics, intrinsics = [], []
        world_points, cam_points, point_masks = [], [], []
        ori_sizes, filepaths, sizes, scales = [], [], [], []
        choose_indices = []
        models = []
        cat_names = []
        ids = []
        bboxes = []
        crop_boxes = []
        
        # Process each object in the frame
        for obj_idx, obj_annotation in enumerate(frame_objects):
            # Create instance mask for this object
            inst_mask = np.equal(mask, obj_annotation["inst_id"])
            # inst_mask = np.logical_and(inst_mask, depth > 0)
            if not np.any(inst_mask):
                # Skip invalid objects that have no visible pixels in this frame.
                continue
            
            # Apply mask to shared data
            obj_image = image.copy()
            obj_nocs = nocs.copy()
            obj_depth = depth.copy()
            obj_depth_sensor = depth_sensor.copy()
            
            obj_image[~inst_mask] = 0
            obj_nocs[~inst_mask] = 0
            obj_depth[~inst_mask] = 0
            obj_depth_sensor[~inst_mask] = 0

            # Get camera parameters and object pose
            intri, extri = self._camera_from_meta(obj_annotation)
            
            # Apply symmetry transformation if needed
            rotation, obj_nocs = self._apply_symmetry_transform(
                obj_annotation["rotation"], obj_nocs, obj_annotation["class_id"]
            )
            
            # Store original size
            original_size = np.array(obj_image.shape[:2])
            
            # Process and crop image
            image_, depth_, depth_sensor_, inst_mask_, nocs_, extri_, intri_ = self._process_crop_image(
                obj_image, obj_depth, obj_depth_sensor, inst_mask, obj_nocs, extri, intri,
                target_image_shape=target_image_shape,
                viewport_rel_pad=0.1,
            )

            # image_, depth_, depth_sensor_, inst_mask_, nocs_, extri_, intri_ = self._process_crop_image_torch(
            #     obj_image, obj_depth, obj_depth_sensor, inst_mask, nocs, extri, intri,
            #     target_image_shape=target_image_shape,
            #     viewport_rel_pad=0.1,
            #     device='cuda'
            # )
            
            # Convert depth to world and camera coordinates
            world_coords_points, cam_coords_points, point_mask = (
                depth_to_world_coords_points(depth_, extri_, intri_)
            )
            
            H, W = inst_mask_.shape[:2]
            mask_flat = inst_mask_.flatten()
            choose = mask_flat.nonzero()[0]
            if len(choose) == 0:
                continue
            
            if len(choose) <= self.sample_num:
                choose_idx = np.random.choice(np.arange(len(choose)), self.sample_num)
            else:
                choose_idx = np.random.choice(np.arange(len(choose)), self.sample_num, replace=False)
            choose = choose[choose_idx].astype(np.int64)
            
            # Get model for this object
            model_name = obj_annotation.get("class_name", f"model_{obj_annotation['class_id']}")
            model = self.models.get(model_name, np.zeros((100, 3), dtype=np.float32))
            
            # Append to lists
            images.append(image_.astype(np.uint8).copy())
            depths.append(depth_.astype(np.float32).copy() if depth_ is not None else np.zeros(image_.shape[:2], np.float32))
            depths_sensor.append(depth_sensor_.astype(np.float16).copy() if depth_sensor_ is not None else np.zeros(image_.shape[:2], np.float16))
            inst_masks.append(inst_mask_.astype(np.uint8).copy())
            nocs_list.append(nocs_.astype(np.float16).copy())
            extrinsics.append(extri_[:3, :].astype(np.float16).copy())
            intrinsics.append(intri_.astype(np.float16).copy())
            world_points.append(world_coords_points.astype(np.float16).copy())
            cam_points.append(cam_coords_points.astype(np.float16).copy())
            point_masks.append(point_mask.astype(np.uint8).copy())
            ori_sizes.append(original_size)
            filepaths.append(obj_annotation["paths"]["color"])
            sizes.append(obj_annotation.get("size", 1.0))
            scales.append(np.linalg.norm(obj_annotation.get("size", 1.0)))
            choose_indices.append(choose)
            models.append(model.astype(np.float16).copy())
            cat_names.append(obj_annotation["class_name"])
            ids.append(obj_idx)
            bboxes.append(obj_annotation.get("bbox", [0, 0, 0, 0]))
            # crop_boxes.append(crop_info.astype(np.float32).copy())
        
        batch = {
            "seq_name": unique_key,  # Frame key: scene_frame_id
            "cat_name": cat_names,  # list of per object's category
            "ids": np.array(ids, dtype=np.int32),
            "frame_num": len(images),  # Number of objects in this frame
            "images": images,
            "depths": depths,
            "depths_sensor": depths_sensor,
            "inst_masks": inst_masks,
            "nocs": nocs_list,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "world_points": world_points,
            "cam_points": cam_points,
            "point_masks": point_masks,
            "original_sizes": ori_sizes,
            "filepaths": filepaths,
            "sizes": sizes,
            "scales": scales,
            "choose_indices": choose_indices,
            "models": models,
            "bboxes": bboxes,
            # "crop_boxes": crop_boxes,
        }

        return batch


if __name__ == "__main__":
    import argparse
    from types import SimpleNamespace
    import random
    
    def build_common_conf(training=True, debug=False):
        """Build common configuration for testing."""
        augs = SimpleNamespace(
            scales=[0.8, 1.2],
            cojitter=False,
            cojitter_ratio=0.5,
            color_jitter=0.0,
            gray_scale=0.0,
            gau_blur=0.0,
            aspects=[0.75, 1.0],
        )
        return SimpleNamespace(
            img_size=518,
            patch_size=14,
            augs=augs,
            rescale=True,
            rescale_aug=True,
            landscape_check=True,
            debug=debug,
            training=training,
            get_nearby=False,
            load_depth=True,
            inside_random=False,
            allow_duplicate_img=False,
            fix_img_num=0,
            fix_aspect_ratio=1.0,
            load_track=False,
            track_num=0,
        )
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="Path to HouseCat6D dataset root")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--img_per_seq", type=int, default=4)
    parser.add_argument("--aspect", type=float, default=1.0)
    parser.add_argument("--force_rebuild", type=bool, default=True,
                        help="Force rebuild cache")
    parser.add_argument("--test_dataset", type=bool, default=False,
                        help="Test the new HouseCat6DTestDataset instead of training dataset")
    args = parser.parse_args()
    
    # For test dataset, force split to be "test"
    if args.test_dataset and args.split != "test":
        print(f"Warning: Test dataset only works with test split. Changing split from '{args.split}' to 'test'")
        args.split = "test"
    
    # Create dataset
    conf = build_common_conf(training=(args.split == "train"), debug=False)
    
    if args.test_dataset:
        # Test the new HouseCat6DTestDataset
        ds = HouseCat6DTestDataset(
            common_conf=conf,
            data_root=args.data_root,
            split=args.split,
            force_rebuild_cache=args.force_rebuild,
            sample_num=1024,  # Default sample size
        )
        
        print(f"Found {len(ds.unique_keys)} frames in HouseCat6D {args.split}")
        print(f"Total test samples (objects): {ds.len_train}")
        
        if ds.len_train == 0:
            raise SystemExit("No test samples found. Check --data_root and dataset layout.")
        
        import time
        # Test loading a few samples
        for i in range(min(100, ds.len_train)):
            # sample_index = random.randint(0, ds.len_train - 1)
            sample_index = i
            print(f"Sample index: {sample_index}")
            print(f"Sample key: {ds.unique_keys[sample_index]}")
            start_time = time.time()
            batch = ds.get_data(sample_index=sample_index, aspect_ratio=args.aspect)
            end_time = time.time()
            print(f"Time taken: {end_time - start_time} seconds")
                
            print(f"\nTest Sample {i+1}:")
            print("  seq_name (frame_key):", batch["seq_name"])
            print("  cat_name:", batch["cat_name"])
            print("  ids:", batch["ids"])
            print("  num objects in frame:", batch["frame_num"])
            print("  image shape:", batch["images"][0].shape)
            print("  depth shape:", batch["depths"][0].shape)
            print("  nocs shape:", batch["nocs"][0].shape)
            print("  K shape:", batch["intrinsics"][0].shape)
            print("  extrinsic shape:", batch["extrinsics"][0].shape)
            print("  Batch structure: B=1, S={} (objects per frame)".format(batch["frame_num"]))
            print("  filepath:", batch["filepaths"][0])
    else:
        # Test the original HouseCat6DPoseDataset
        ds = HouseCat6DPoseDataset(
            common_conf=conf,
            data_root=args.data_root,
            split=args.split,
            force_rebuild_cache=args.force_rebuild,
            sample_num=1024,  # Default sample size
        )
        
        print(f"Found {len(ds.seq_names)} object-centric sequences in HouseCat6D {args.split}")
        
        if len(ds.seq_names) == 0:
            raise SystemExit("No sequences found. Check --data_root and dataset layout.")
        
        # Test loading a few batches
        for i in range(min(10, len(ds.seq_names))):
            seq_index = random.randint(0, len(ds.seq_names) - 1)
            batch = ds.get_data(seq_index=seq_index, img_per_seq=args.img_per_seq, aspect_ratio=args.aspect)
            
            print(f"\nBatch {i+1}:")
            print("  seq_name:", batch["seq_name"])
            print("  cat_name:", batch["cat_name"])
            print("  ids:", batch["ids"])
            print("  num frames:", batch["frame_num"])
            print("  image shape:", batch["images"][0].shape)
            print("  depth shape:", batch["depths"][0].shape)
            print("  nocs shape:", batch["nocs"][0].shape)
            print("  K shape:", batch["intrinsics"][0].shape)
            print("  extrinsic shape:", batch["extrinsics"][0].shape)

            # Stack the lists into arrays and add batch dimension (B=1)
            extrinsics = np.expand_dims(np.stack(batch['extrinsics'], axis=0), axis=0)  # Shape: (1, 4, 3, 4)
            cam_points = np.expand_dims(np.stack(batch['cam_points'], axis=0), axis=0)  # Shape: (1, 4, H, W, 3)
            world_points = np.expand_dims(np.stack(batch['world_points'], axis=0), axis=0)  # Shape: (1, 4, H, W, 3)
            depths = np.expand_dims(np.stack(batch['depths'], axis=0), axis=0)  # Shape: (1, 4, H, W)
            point_masks = np.expand_dims(np.stack(batch['point_masks'], axis=0), axis=0)  # Shape: (1, 4, H, W)

            # Convert extrinsics to homogeneous form: (B, S, 4, 4)
            extrinsics_homog = np.concatenate(
                [
                    extrinsics,
                    np.zeros((1, 4, 1, 4)),
                ],
                axis=-2,
            )
            extrinsics_homog[:, :, -1, -1] = 1.0

            # Call the function
            new_extrinsics, new_cam_points, new_world_points, new_depths, avg_scale = \
                normalize_camera_extrinsics_and_points_batch_numpy(
                    extrinsics=extrinsics,
                    cam_points=cam_points,
                    world_points=world_points,
                    depths=depths,
                    scale_by_points=True,
                    point_masks=point_masks
                )

            # Convert extrinsics to homogeneous form: (B, S, 4, 4)
            new_extrinsics_homog = np.concatenate(
                [
                    new_extrinsics,
                    np.zeros((1, 4, 1, 4)),
                ],
                axis=-2,
            )
            new_extrinsics_homog[:, :, -1, -1] = 1.0

            # Now you can use the returned values, e.g., print shapes
            print(" new extrinsic 0: \n", new_extrinsics[0][0].astype(np.float16))  # Should be (1, 4, 3, 4)

            new_sizes = batch['sizes'] / avg_scale

            # Recover the absolute object pose from pointmap and nocsmap
            new_extrinsics_0 = new_extrinsics[0][0]
            new_cam_points_0 = new_cam_points[0][0]
            new_cam_points_1 = new_cam_points[0][1]
            new_cam_points_2 = new_cam_points[0][2]
            new_cam_points_3 = new_cam_points[0][3]
            cam_points_0 = cam_points[0][0]
            cam_points_1 = cam_points[0][1]
            cam_points_2 = cam_points[0][2]
            cam_points_3 = cam_points[0][3]
            new_world_points_0 = new_world_points[0][0]
            new_world_points_1 = new_world_points[0][1]
            new_world_points_2 = new_world_points[0][2]
            new_world_points_3 = new_world_points[0][3]
            new_depths_0 = new_depths[0][0]
            new_point_masks_0 = point_masks[0][0]
            new_point_masks_1 = point_masks[0][1]
            new_point_masks_2 = point_masks[0][2]
            new_point_masks_3 = point_masks[0][3]

            point_mask_0 = new_point_masks_0.astype(bool)
            point_mask_1 = new_point_masks_1.astype(bool)
            point_mask_2 = new_point_masks_2.astype(bool)
            point_mask_3 = new_point_masks_3.astype(bool)
            pointmap_0 = new_world_points_0[point_mask_0]
            pointmap_1 = new_world_points_1[point_mask_1]
            pointmap_2 = new_world_points_2[point_mask_2]
            pointmap_3 = new_world_points_3[point_mask_3]
            nocs_map_0 = batch["nocs"][0][point_mask_0]
            nocs_map_1 = batch["nocs"][1][point_mask_1]
            nocs_map_2 = batch["nocs"][2][point_mask_2]
            nocs_map_3 = batch["nocs"][3][point_mask_3]
            conf_pm_0 = None
            conf_nocs_0 = None
            # abs_scale_0 = batch['scales'][0]
            # print(" abs_scale_0:", abs_scale_0)
            print(" avg_scale:", avg_scale)

            # R_a, t_a, s_a, sim3_a = umeyama_similarity_transform(pointmap_0 * abs_scale_0, nocs_map_0, point_conf=None, ransac=True, ransac_iters=300, inlier_thresh=0.02, fix_scale=False)


            pointmap_0_original = world_points[0][0][point_mask_0]
            cammap_0_original = cam_points[0][0][point_mask_0]
            cammap_0 = new_cam_points_0[point_mask_0]
            cammap_2 = new_cam_points_2[point_mask_2]
            cammap_3 = new_cam_points_3[point_mask_3]
            cammap_1 = new_cam_points_1[point_mask_1]
            cammap_0_original = cam_points_0[point_mask_0]
            cammap_1_original = cam_points_1[point_mask_1]
            cammap_2_original = cam_points_2[point_mask_2]
            cammap_3_original = cam_points_3[point_mask_3]

            import time
            R_, t_, s_, sim3_a_original_ = umeyama_similarity_transform(pointmap_0, nocs_map_0, point_conf=None, ransac=False, ransac_iters=300, inlier_thresh=0.02, fix_scale=False)
            Rc, tc, sc, sim3_a_original_c = umeyama_similarity_transform(nocs_map_0, cammap_0, point_conf=None, ransac=True, ransac_iters=300, inlier_thresh=0.02, fix_scale=False)
            Rc1, tc1, sc1, sim3_a_original_c1 = umeyama_similarity_transform(nocs_map_1, cammap_1, point_conf=None, ransac=True, ransac_iters=300, inlier_thresh=0.02, fix_scale=False)
            Rc2, tc2, sc2, sim3_a_original_c2 = umeyama_similarity_transform(nocs_map_2, cammap_2, point_conf=None, ransac=True, ransac_iters=300, inlier_thresh=0.02, fix_scale=False)
            Rc3, tc3, sc3, sim3_a_original_c3 = umeyama_similarity_transform(nocs_map_3, cammap_3, point_conf=None, ransac=True, ransac_iters=300, inlier_thresh=0.02, fix_scale=False)

            Rc0_original, tc0_original, sc0_original, sim3_a_original_c0 = umeyama_similarity_transform(nocs_map_0, cammap_0_original, point_conf=None, ransac=True, ransac_iters=300, inlier_thresh=0.02, fix_scale=False)
            Rc1_original, tc1_original, sc1_original, sim3_a_original_c1 = umeyama_similarity_transform(nocs_map_1, cammap_1_original, point_conf=None, ransac=True, ransac_iters=300, inlier_thresh=0.02, fix_scale=False)
            Rc2_original, tc2_original, sc2_original, sim3_a_original_c2 = umeyama_similarity_transform(nocs_map_2, cammap_2_original, point_conf=None, ransac=True, ransac_iters=300, inlier_thresh=0.02, fix_scale=False)
            Rc3_original, tc3_original, sc3_original, sim3_a_original_c3 = umeyama_similarity_transform(nocs_map_3, cammap_3_original, point_conf=None, ransac=True, ransac_iters=300, inlier_thresh=0.02, fix_scale=False)

            Rp, tp, sp, sim3_a_original_c = umeyama_similarity_transform(pointmap_0, cammap_0, point_conf=None, ransac=True, ransac_iters=300, inlier_thresh=0.02, fix_scale=False)
            Rp1, tp1, sp1, sim3_a_original_c1 = umeyama_similarity_transform(pointmap_1, cammap_1, point_conf=None, ransac=True, ransac_iters=300, inlier_thresh=0.02, fix_scale=False)
            Rp2, tp2, sp2, sim3_a_original_c2 = umeyama_similarity_transform(pointmap_2, cammap_2, point_conf=None, ransac=True, ransac_iters=300, inlier_thresh=0.02, fix_scale=False)
            Rp3, tp3, sp3, sim3_a_original_c3 = umeyama_similarity_transform(pointmap_3, cammap_3, point_conf=None, ransac=True, ransac_iters=300, inlier_thresh=0.02, fix_scale=False)

            print(" filepaths:", batch["filepaths"][0])
