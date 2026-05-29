# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Project-owned portions of this file are licensed under CC-BY-NC-SA-4.0.
# See LICENSE and NOTICE for details. Third-party notices remain applicable.

import os
import sys

# CRITICAL: Set OpenGL environment variables BEFORE any imports
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["PYOPENGL_USE_ACCELERATE"] = "False"
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'

# Block OpenGL_accelerate from loading
import builtins
_original_import = builtins.__import__

def _blocked_import(name, *args, **kwargs):
    if 'OpenGL_accelerate' in name:
        raise ImportError(f"OpenGL_accelerate is blocked: {name}")
    return _original_import(name, *args, **kwargs)

builtins.__import__ = _blocked_import

# Configure OpenGL before any other imports
try:
    import OpenGL
    OpenGL.ERROR_CHECKING = False
    OpenGL.ERROR_LOGGING = False
    OpenGL.ERROR_ON_COPY = True
    OpenGL.ARRAY_SIZE_CHECKING = False
    import OpenGL.GL
    print("OpenGL initialized without accelerate")
except Exception as e:
    print(f"OpenGL init warning: {e}")
finally:
    builtins.__import__ = _original_import

# Now import everything else
import cv2
import glob
import argparse
import numpy as np
import torch
import struct
from scipy.spatial.transform import Rotation
from PIL import Image
import open3d as o3d
from pathlib import Path
from torchvision import transforms as TF
from einops import rearrange, repeat
import struct
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import KDTree
import torch.nn.functional as F
import pandas as pd
from tqdm.auto import tqdm
from datetime import datetime as dt
import json
from pathlib import Path
from rel_pose_runtime_dataset import build_runtime_pair_loader
from bop_toolkit_lib.misc import *
from bop_toolkit_lib.renderer_vispy import RendererVispy
from bop_toolkit_lib.pose_error import my_mssd, my_mspd, vsd
from utils.pcd import get_diameter
from utils.runtime_eval_log import write_runtime_summary
from typing import Optional, Tuple
from plyfile import PlyData

sys.path.append("opt/")

from opt.models.opt import OPT
from opt.utils.load_fn import load_and_preprocess_images
from opt.utils.pose_enc import pose_encoding_to_extri_intri
from opt.utils.geometry import unproject_depth_map_to_point_map, depth_to_cam_coords_points
from opt.utils.geometry import closed_form_inverse_se3

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def compute_relative_pose(pose_src: np.ndarray, pose_dst: np.ndarray) -> np.ndarray:
    """Compute the homogeneous transform that maps coordinates from src frame to dst frame."""
    pose_src = np.asarray(pose_src, dtype=np.float64)
    pose_dst = np.asarray(pose_dst, dtype=np.float64)

    pose_src_inv = np.linalg.inv(pose_src)
    relative_pose = pose_dst @ pose_src_inv

    return relative_pose

def load_model(device=None, checkpoint_path=None):
    """Load VGGT model with relative pose head enabled."""
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required but not available.")
        device = "cuda"
    print(f"Using device: {device}")

    if checkpoint_path is None:
        raise ValueError("checkpoint_path is required; checkpoints are not shipped with this release.")
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Resolve state_dict
    if isinstance(ckpt, dict):
        state = ckpt.get('state_dict', None) or ckpt.get('model', None) or ckpt.get('model_state_dict', None) or ckpt
    else:
        state = ckpt

    # Strip common prefixes
    def strip_prefix(sdict, prefix):
        return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sdict.items()}

    candidate_states = [state]
    for pref in ['module.', 'model.', 'vggt.']:
        candidate_states.append(strip_prefix(state, pref))

    # Initialize model with relative pose head enabled
    model = OPT(
        enable_pose=True,
        enable_nocs=True,
        enable_rel_scale=True,
    )
    
    loaded = False
    for idx, st in enumerate(candidate_states):
        try:
            missing, unexpected = model.load_state_dict(st, strict=False)
            print(f"Loaded with variant {idx}: missing={len(missing)}, unexpected={len(unexpected)}")
            loaded = True
            break
        except Exception:
            continue
    if not loaded:
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded with raw state: missing={len(missing)}, unexpected={len(unexpected)}")

    model.eval()
    model = model.to(device)

    return model, device

def read_image(image_file):
    img = Image.open(image_file)
    # img = imageio.v2.imread(image_file)
    return np.array(img)

def read_seg(seg_file):
    seg = cv2.imread(seg_file, cv2.IMREAD_GRAYSCALE)
    seg = seg.astype(np.float32) / 255.0
    seg = seg.astype(np.uint8)
    return seg
        
def read_depth_png(depth_file):
    """Read 16-bit PNG depth image"""
    depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
    return depth.astype(np.float32)

def read_intrinsics(intrinsics_file):
    """Read 3x3 intrinsics matrix from txt file"""
    K = np.loadtxt(intrinsics_file)
    return K

def read_pose(pose_file):
    """Read 4x4 pose matrix from txt file"""
    pose = np.loadtxt(pose_file)
    return pose

def read_ply_points(ply_file):
    """Read 3D points from PLY file (supports both ASCII and binary formats)"""
    points = []
    colors = []
    
    if not os.path.exists(ply_file):
        print(f"PLY file not found: {ply_file}")
        return np.array([]), np.array([])
    
    try:
        with open(ply_file, 'rb') as f:
            # Read header
            header_lines = []
            while True:
                line = f.readline().decode('ascii').strip()
                header_lines.append(line)
                if line == 'end_header':
                    break
            
            # Parse header
            is_binary = False
            vertex_count = 0
            properties = []
            
            for line in header_lines:
                if line.startswith('format'):
                    if 'binary' in line:
                        is_binary = True
                elif line.startswith('element vertex'):
                    vertex_count = int(line.split()[-1])
                elif line.startswith('property'):
                    properties.append(line.split())
            
            print(f"Reading PLY: {vertex_count} vertices, binary={is_binary}")
            
            if is_binary:
                # Read binary data
                # Determine format string for struct
                fmt_chars = []
                for prop in properties:
                    if len(prop) >= 2:
                        prop_type = prop[1]
                        if prop_type == 'double':
                            fmt_chars.append('d')
                        elif prop_type == 'float':
                            fmt_chars.append('f')
                        elif prop_type == 'uchar':
                            fmt_chars.append('B')
                        elif prop_type == 'int':
                            fmt_chars.append('i')
                
                fmt_str = '<' + ''.join(fmt_chars)  # little endian
                struct_size = struct.calcsize(fmt_str)
                
                for i in range(vertex_count):
                    data = f.read(struct_size)
                    if len(data) < struct_size:
                        break
                    
                    values = struct.unpack(fmt_str, data)
                    
                    # Extract x, y, z (first 3 values)
                    x, y, z = values[:3]
                    points.append([x, y, z])
                    
                    # Extract colors if available (typically last 3 values)
                    if len(values) >= 6:
                        r, g, b = values[-3:]
                        colors.append([int(r), int(g), int(b)])
                    else:
                        colors.append([128, 128, 128])  # Default gray
                        
            else:
                # ASCII format (original code)
                for line in f:
                    line = line.decode('ascii').strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 6:  # x, y, z, r, g, b
                            x, y, z = map(float, parts[:3])
                            r, g, b = map(int, parts[3:6])
                            points.append([x, y, z])
                            colors.append([r, g, b])
                        elif len(parts) >= 3:  # x, y, z only
                            x, y, z = map(float, parts[:3])
                            points.append([x, y, z])
                            colors.append([128, 128, 128])  # Default gray
        
        points_array = np.array(points)
        colors_array = np.array(colors)
        print(f"Successfully loaded {len(points_array)} points from PLY file")
        return points_array, colors_array
        
    except Exception as e:
        print(f"Error reading PLY file {ply_file}: {e}")
        return np.array([]), np.array([])

def parse_colmap_pose(qw, qx, qy, qz, tx, ty, tz):
    """
    Parse COLMAP pose format (quaternion + translation) to 4x4 transformation matrix
    COLMAP uses quaternion as qw, qx, qy, qz and translation as tx, ty, tz
    """
    # Convert quaternion to rotation matrix
    R = np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
    ])
    
    # Create 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    
    return T

def load_poses(pose_file):
    """Load poses from COLMAP format images.txt"""
    poses = {}
    
    with open(pose_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            
            parts = line.split()
            image_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            camera_id = int(parts[8])
            image_name = parts[9]
            
            # Convert to camera-to-world transformation matrix
            T_cam_world = parse_colmap_pose(qw, qx, qy, qz, tx, ty, tz)
            # COLMAP uses world-to-camera, so we need to invert
            T_world_cam = np.linalg.inv(T_cam_world)
            
            poses[image_id] = {
                'pose': T_world_cam,
                'camera_id': camera_id,
                'image_name': image_name
            }
    
    return poses

def load_cameras(cameras_file):
    """Load camera intrinsics from COLMAP cameras.txt format"""
    cameras = {}
    
    with open(cameras_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]  # Should be PINHOLE
            width = int(parts[2])
            height = int(parts[3])
            
            if model == 'PINHOLE':
                # PINHOLE: fx, fy, cx, cy
                fx, fy, cx, cy = map(float, parts[4:8])
                K = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ])
                cameras[camera_id] = {
                    'K': K,
                    'width': width,
                    'height': height,
                    'model': model
                }
    
    return cameras

def sample_depth_at_pixels(depth_map, pixels, interpolation='bilinear'):
    """
    Sample depth values from depth map at given pixel locations.
    
    Args:
        depth_map: (H, W) depth image
        pixels: (N, 2) pixel coordinates (x, y)
        interpolation: 'nearest' or 'bilinear'
    
    Returns:
        depths: (N,) sampled depth values
        valid_mask: (N,) boolean mask for valid samples
    """
    H, W = depth_map.shape
    depths = np.zeros(len(pixels), dtype=np.float64)
    valid_mask = np.zeros(len(pixels), dtype=bool)
    
    # Filter pixels within image bounds
    in_bounds = ((pixels[:, 0] >= 0) & (pixels[:, 0] < W) & 
                 (pixels[:, 1] >= 0) & (pixels[:, 1] < H))
    
    if not np.any(in_bounds):
        return depths, valid_mask
    
    pixels_valid = pixels[in_bounds]
    
    if interpolation == 'nearest':
        # Round to nearest integer pixel
        x_int = np.round(pixels_valid[:, 0]).astype(int)
        y_int = np.round(pixels_valid[:, 1]).astype(int)
        
        # Clamp to image bounds
        x_int = np.clip(x_int, 0, W - 1)
        y_int = np.clip(y_int, 0, H - 1)
        
        sampled_depths = depth_map[y_int, x_int].astype(np.float64)
        
    elif interpolation == 'bilinear':
        # Bilinear interpolation
        x = pixels_valid[:, 0].astype(np.float64)
        y = pixels_valid[:, 1].astype(np.float64)
        
        x0 = np.floor(x).astype(int)
        y0 = np.floor(y).astype(int)
        x1 = np.minimum(x0 + 1, W - 1)
        y1 = np.minimum(y0 + 1, H - 1)
        
        # Clamp to bounds
        x0 = np.clip(x0, 0, W - 1)
        y0 = np.clip(y0, 0, H - 1)
        
        # Get the four corner values and convert to float64
        I00 = depth_map[y0, x0].astype(np.float64)
        I01 = depth_map[y1, x0].astype(np.float64)
        I10 = depth_map[y0, x1].astype(np.float64)
        I11 = depth_map[y1, x1].astype(np.float64)
        
        # Calculate interpolation weights
        wx = x - x0.astype(np.float64)
        wy = y - y0.astype(np.float64)
        
        # Bilinear interpolation
        sampled_depths = (I00 * (1 - wx) * (1 - wy) + 
                         I10 * wx * (1 - wy) + 
                         I01 * (1 - wx) * wy + 
                         I11 * wx * wy)
    
    # Filter out invalid depth values
    depth_valid = (sampled_depths > 0) & (sampled_depths < 10000)  # Reasonable depth range
    
    depths[in_bounds] = sampled_depths
    valid_mask[in_bounds] = depth_valid
    
    return depths, valid_mask

def project_cam_points_to_pixels(points_cam: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project camera-frame 3D points to image pixels.

    Returns:
        pixels: (N, 2) projected pixel coordinates
        valid_mask: (N,) points with finite coordinates and positive depth
    """
    points_cam = np.asarray(points_cam, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)

    pixels = np.zeros((points_cam.shape[0], 2), dtype=np.float64)
    valid_mask = np.isfinite(points_cam).all(axis=1) & (points_cam[:, 2] > 1e-8)
    if not np.any(valid_mask):
        return pixels, valid_mask

    points_valid = points_cam[valid_mask]
    proj = (K @ points_valid.T).T
    pixels[valid_mask] = proj[:, :2] / proj[:, [2]]
    return pixels, valid_mask

def subsample_points(points: np.ndarray, max_points: Optional[int]) -> np.ndarray:
    """Deterministically subsample a point set to at most max_points points."""
    if max_points is None or max_points <= 0 or points.shape[0] <= max_points:
        return points

    idx = np.linspace(0, points.shape[0] - 1, num=max_points, dtype=np.int64)
    return points[idx]

def make_open3d_point_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    """Create an Open3D point cloud from an (N, 3) array."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return pcd

def refine_query_pose_with_icp(
    pred_pose: np.ndarray,
    model_points_obj: np.ndarray,
    depth_map_query: np.ndarray,
    obj_mask_query: np.ndarray,
    K_query: np.ndarray,
    *,
    enabled: bool = True,
    voxel_size: float = 0.005,
    max_correspondence_distance: float = 0.02,
    depth_gate: float = 0.02,
    max_iterations: int = 50,
    min_points: int = 128,
    max_points: int = 6000,
):
    """
    Refine an object->query-camera pose with ICP against the query depth cloud.

    The initial pose is used to transform the object model into query-camera space.
    ICP estimates a residual rigid transform in camera space, which is then composed
    back onto pred_pose.
    """
    info = {
        "applied": False,
        "source_points": 0,
        "target_points": 0,
        "fitness": np.nan,
        "inlier_rmse": np.nan,
    }

    pred_pose = np.asarray(pred_pose, dtype=np.float64)
    model_points_obj = np.asarray(model_points_obj, dtype=np.float64)
    depth_map_query = np.asarray(depth_map_query, dtype=np.float64)
    obj_mask_query = np.asarray(obj_mask_query).astype(bool)
    K_query = np.asarray(K_query, dtype=np.float64)

    if not enabled or model_points_obj.size == 0:
        return pred_pose, info

    target_points = depth_to_cam_coords_points(depth_map_query, K_query)[obj_mask_query]
    target_valid = np.isfinite(target_points).all(axis=1) & (target_points[:, 2] > 1e-8)
    target_points = target_points[target_valid]
    if target_points.shape[0] < min_points:
        return pred_pose, info

    rot = pred_pose[:3, :3]
    trans = pred_pose[:3, 3]
    model_points_cam = (rot @ model_points_obj.T).T + trans

    pixels, proj_valid = project_cam_points_to_pixels(model_points_cam, K_query)
    H, W = obj_mask_query.shape
    rounded_pixels = np.round(pixels).astype(np.int64)
    in_image = (
        proj_valid
        & (rounded_pixels[:, 0] >= 0)
        & (rounded_pixels[:, 0] < W)
        & (rounded_pixels[:, 1] >= 0)
        & (rounded_pixels[:, 1] < H)
    )
    if not np.any(in_image):
        return pred_pose, info

    mask_hits = np.zeros(model_points_cam.shape[0], dtype=bool)
    mask_hits[in_image] = obj_mask_query[rounded_pixels[in_image, 1], rounded_pixels[in_image, 0]]

    depth_samples = np.zeros(model_points_cam.shape[0], dtype=np.float64)
    depth_valid = np.zeros(model_points_cam.shape[0], dtype=bool)
    if np.any(mask_hits):
        depth_samples_masked, depth_valid_masked = sample_depth_at_pixels(
            depth_map_query, pixels[mask_hits], interpolation="bilinear"
        )
        depth_samples[mask_hits] = depth_samples_masked
        depth_valid[mask_hits] = depth_valid_masked

    depth_consistent = np.zeros(model_points_cam.shape[0], dtype=bool)
    depth_consistent[mask_hits] = np.abs(depth_samples[mask_hits] - model_points_cam[mask_hits, 2]) <= depth_gate

    source_valid = mask_hits & depth_valid & depth_consistent
    source_points = model_points_cam[source_valid]
    if source_points.shape[0] < min_points:
        source_points = model_points_cam[mask_hits]
    source_points = source_points[np.isfinite(source_points).all(axis=1) & (source_points[:, 2] > 1e-8)]
    if source_points.shape[0] < min_points:
        return pred_pose, info

    source_points = subsample_points(source_points, max_points)
    target_points = subsample_points(target_points, max_points)

    source_pcd = make_open3d_point_cloud(source_points)
    target_pcd = make_open3d_point_cloud(target_points)
    if voxel_size > 0:
        source_pcd = source_pcd.voxel_down_sample(voxel_size)
        target_pcd = target_pcd.voxel_down_sample(voxel_size)

    source_size = len(source_pcd.points)
    target_size = len(target_pcd.points)
    info["source_points"] = source_size
    info["target_points"] = target_size
    if source_size < min_points or target_size < min_points:
        return pred_pose, info

    estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)

    residual = np.eye(4, dtype=np.float64)
    fine_distance = max(voxel_size, max_correspondence_distance * 0.5)
    try:
        for corr_dist in (max_correspondence_distance, fine_distance):
            result = o3d.pipelines.registration.registration_icp(
                source_pcd,
                target_pcd,
                corr_dist,
                residual,
                estimation,
                criteria,
            )
            residual = result.transformation
    except Exception as exc:
        print(f"ICP refinement warning: {exc}")
        return pred_pose, info

    refined_pose = residual @ pred_pose
    info["applied"] = True
    info["fitness"] = float(result.fitness)
    info["inlier_rmse"] = float(result.inlier_rmse)
    return refined_pose, info

def np_transform_pcd(pcd: np.ndarray, r : np.ndarray, t: np.ndarray) -> np.ndarray:
    pcd = pcd.astype(np.float16)
    r = r.astype(np.float16)
    t = t.astype(np.float16)
    rot_pcd = np.dot(np.asarray(pcd), r.T) + t
    return rot_pcd 

def compute_add(pcd : np.ndarray, pred_pose : np.ndarray, gt_pose : np.ndarray) -> np.ndarray:
    pred_r, pred_t = pred_pose[:3,:3], pred_pose[:3,3]
    gt_r, gt_t = gt_pose[:3,:3], gt_pose[:3,3]

    model_pred = np_transform_pcd(pcd, pred_r, pred_t)
    model_gt = np_transform_pcd(pcd, gt_r, gt_t)

    # ADD computation
    add = np.mean(np.linalg.norm(model_pred - model_gt, axis=1))
    return add

def compute_adds(pcd : np.ndarray, pred_pose : np.ndarray, gt_pose : np.ndarray) -> np.ndarray:
    pred_r, pred_t = pred_pose[:3,:3], pred_pose[:3,3]
    gt_r, gt_t = gt_pose[:3,:3], gt_pose[:3,3]

    model_pred = np_transform_pcd(pcd, pred_r, pred_t)
    model_gt = np_transform_pcd(pcd, gt_r, gt_t)
    
    # ADD-S computation
    kdt = KDTree(model_gt, metric='euclidean')
    distance, _ = kdt.query(model_pred, k=1)
    adds = np.mean(distance)
    
    return adds

def compute_RT_distances(pose1 : np.ndarray, pose2 : np.ndarray):
    '''
    :param RT_1: [B, 4, 4]. homogeneous affine transformation
    :param RT_2: [B, 4, 4]. homogeneous affine transformation
    :return: theta: angle difference of R in degree, shift: l2 difference of T in centimeter
    Works in batched or unbatched manner. NB: assumes that translations are in Meters
    '''

    if pose1 is None or pose2 is None:
        return -1

    if len(pose1.shape) == 2:
        pose1 = np.expand_dims(pose1,axis=0)
        pose2 = np.expand_dims(pose2,axis=0)

    try:
        assert np.array_equal(pose1[:, 3, :], pose2[:, 3, :])
        assert np.array_equal(pose1[0, 3, :], np.array([0, 0, 0, 1]))
    except AssertionError:
        print(pose1[:, 3, :], pose2[:, 3, :])


    BS = pose1.shape[0]

    R1 = pose1[:, :3, :3] / np.cbrt(np.linalg.det(pose1[:, :3, :3]))[:,None,None]
    T1 = pose1[:, :3, 3]

    R2 = pose2[:, :3, :3] / np.cbrt(np.linalg.det(pose2[:, :3, :3]))[:,None,None]
    T2 = pose2[:, :3, 3]
    
    R = np.matmul(R1,R2.transpose(0,2,1))
    arccos_arg = (np.trace(R,axis1=1, axis2=2) - 1)/2
    arccos_arg = np.clip(arccos_arg, -1+1e-12, 1-1e-12)
    theta = np.arccos(arccos_arg) * 180/np.pi
    theta[np.isnan(theta)] = 180.
    shift = np.linalg.norm(T1-T2,axis=-1) * 100

    return theta, shift

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

def sim3_alignment_pipeline(point_map_anchor_gt, point_map_anchor, point_map_query, cam_map_query_gt, cam_map_query, obj_masks, point_conf):
    # Select valid points using object masks
    mask_a = obj_masks[0].astype(bool)
    mask_q = obj_masks[1].astype(bool)

    pm_a = point_map_anchor[mask_a]
    pm_a_gt = point_map_anchor_gt[mask_a]

    # extract confidences for masked points if provided
    conf_a = None
    conf_q = None
    try:
        if point_conf is not None:
            # point_conf expected shape (S, H, W) or (S, N) - handle common cases
            conf_a = np.asarray(point_conf[0])[mask_a]
            conf_q = np.asarray(point_conf[1])[mask_q]
    except Exception:
        conf_a = None
        conf_q = None

    # Estimate similarity transform for anchor -> anchor_gt (allow scale)
    R_a, t_a, s_a, sim3_a = umeyama_similarity_transform(pm_a, pm_a_gt, point_conf=conf_a, ransac=True, ransac_iters=300, inlier_thresh=0.02, fix_scale=False)
    se3_a = np.vstack([np.column_stack([R_a, t_a]), [0, 0, 0, 1]])  
    # print(f"Anchor sim3: scale={s_a:.4f}, translation norm={np.linalg.norm(t_a):.4f}")

    # Apply sim3 transform to predicted anchor points and predicted query points
    pm_a_homo = np.hstack([pm_a, np.ones((pm_a.shape[0], 1))])
    pm_a = (sim3_a @ pm_a_homo.T).T[:, :3]

    pm_q = point_map_query[mask_q]
    pm_q_homo = np.hstack([pm_q, np.ones((pm_q.shape[0], 1))])
    pm_q = (sim3_a @ pm_q_homo.T).T[:, :3]

    # After anchor Sim(3) calibration, query alignment should stay rigid.
    cm_q_gt = cam_map_query_gt[mask_q]
    cm_q = cam_map_query[mask_q]
    R_q, t_q, s_q, se3_q = umeyama_similarity_transform(pm_q, cm_q_gt, point_conf=conf_q, ransac=True, ransac_iters=300, inlier_thresh=0.02, fix_scale=True)

    return se3_a, se3_q

def process_images(images, device=None):
    to_tensor = TF.ToTensor()

    images = [Image.fromarray(image) for image in images]
    images_tensor = [to_tensor(image) for image in images]
    images_tensor = torch.stack(images_tensor, dim=0)
    
    if device is not None:
        images_tensor = images_tensor.to(device)
    
    return images_tensor

def run_OPT(model, images, dtype):
    # images: [B, 3, H, W]
    assert len(images.shape) == 4
    assert images.shape[1] == 3

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)

        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        # extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
        point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)


    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    point_map = point_map.squeeze(0).cpu().numpy()
    point_conf = point_conf.squeeze(0).cpu().numpy()
    return depth_map, depth_conf, point_map, point_conf

def process_pair_sample(sample, model, device, dtype, obj_models, obj_diams, obj_symms, renderer, vsd_delta, vsd_taus, vsd_rec, mssd_rec, mspd_rec, icp_cfg=None):
    """Process one runtime-loaded pair sample and return a dict with results."""
    out = {
        'scene': sample['scene'],
        'r_deg': np.nan,
        't_cm': np.nan,
        'ADD(S)-0.1d': np.nan,
        'MSSD': np.nan,
        'MSPD': np.nan,
        'VSD': np.nan,
        'AR': np.nan,
        'icp_applied': 0.0,
        'icp_fitness': np.nan,
        'icp_rmse': np.nan,
    }
    icp_cfg = icp_cfg or {}

    images = sample['images']
    masks = sample['masks']
    depths = sample['depths']
    depths_full = sample['depths_full']
    Ks = sample['Ks']
    obj_poses_gt = sample['poses']
    T_orig_crop = sample['T_orig_crop']
    obj_id = sample['obj_id']

    obj_masks = np.stack(masks, axis=0)  # (S, H, W)
    depth_maps_gt = np.stack(depths, axis=0)[..., None] / 1000.0  # (S, H, W, 1)
    # depth_maps_gt = depths[..., None] / 1000.0  # (S, H, W, 1)
    intrinsic_gt = np.stack(Ks, axis=0)  # (S, 3, 3)
    obj_poses_gt = np.stack(obj_poses_gt, axis=0)  # (S, 4, 4)
    T_orig_crop = np.stack(T_orig_crop, axis=0)  # (S, 4, 4), crop-cam -> orig-cam

    images_tensor = process_images(images, device)

    depth_map, depth_conf, point_map, point_conf = run_OPT(model, images_tensor, dtype)

    depth_conf = depth_conf * obj_masks
    point_conf = point_conf * obj_masks

    obj_rel_pose_gt = compute_relative_pose(obj_poses_gt[0], obj_poses_gt[1])
    # print(f"Ground truth relative object pose (anchor -> query):\n{obj_rel_pose_gt}")

    obj_rel_pose_gt = np.stack([np.eye(4), obj_rel_pose_gt], axis=0)
    # print(f"Object poses (anchor -> query):\n{obj_rel_pose_gt}")
    
    point_map_gt = unproject_depth_map_to_point_map(depth_maps_gt, obj_rel_pose_gt[:, :3, :], intrinsic_gt)

    point_map_anchor_gt = point_map_gt[0]
    point_map_query_gt = point_map_gt[1]

    cam_map_anchor_gt = depth_to_cam_coords_points(depth_maps_gt[0, ..., 0], intrinsic_gt[0])
    cam_map_query_gt = depth_to_cam_coords_points(depth_maps_gt[1, ..., 0], intrinsic_gt[1])
    cam_map_query = depth_to_cam_coords_points(depth_map[1, ..., 0], intrinsic_gt[1])

    point_map_anchor = point_map[0]
    point_map_query = point_map[1]

    se3_a, se3_q = sim3_alignment_pipeline(
        point_map_anchor_gt,
        point_map_anchor,
        point_map_query,
        cam_map_query_gt,
        cam_map_query,
        obj_masks,
        point_conf,
    )

    obj_model, obj_diam, obj_sym = get_obj_info(obj_id, obj_models, obj_diams, obj_symms)
    pred_pose_crop = se3_q @ obj_poses_gt[0]
    T_orig_crop_q = T_orig_crop[1]
    pred_pose = T_orig_crop_q @ pred_pose_crop
    # pred_pose, icp_info = refine_query_pose_with_icp(
    #     pred_pose,
    #     obj_model['pts'] / 1000.0,
    #     depth_maps_gt[1, ..., 0],
    #     obj_masks[1],
    #     intrinsic_gt[1],
    #     enabled=icp_cfg.get('enabled', True),
    #     voxel_size=icp_cfg.get('voxel_size', 0.005),
    #     max_correspondence_distance=icp_cfg.get('max_correspondence_distance', 0.02),
    #     depth_gate=icp_cfg.get('depth_gate', 0.02),
    #     max_iterations=icp_cfg.get('max_iterations', 50),
    #     min_points=icp_cfg.get('min_points', 128),
    #     max_points=icp_cfg.get('max_points', 6000),
    # )
    # out['icp_applied'] = float(icp_info['applied'])
    # out['icp_fitness'] = icp_info['fitness']
    # out['icp_rmse'] = icp_info['inlier_rmse']

    # print(f"Predicted pose: \n{pred_pose}")
    gt_pose = T_orig_crop_q @ obj_poses_gt[1]
    # add diam is different from bop diam
    add_diam = get_diameter(obj_model['pts']) / 1000.

    if obj_sym.shape[0] > 1:
        adds = compute_adds(obj_model['pts'] / 1000., pred_pose, gt_pose)
    else:
        adds = compute_add(obj_model['pts'] / 1000., pred_pose, gt_pose)

    #o3d_viz(obj_model, pred_pose, gt_pose)
    out['ADD(S)-0.1d'] = float(adds <= add_diam*0.1)
            
    pred_pose, gt_pose = pred_pose.astype(np.float16), gt_pose.astype(np.float16)
            
    pred_r, pred_t = pred_pose[:3,:3], np.expand_dims(pred_pose[:3,3],axis=1) * 1000
    gt_r, gt_t = gt_pose[:3,:3], np.expand_dims(gt_pose[:3,3],axis=1) * 1000

    # Use actual scene intrinsics (query view) for BOP metrics
    K_query = intrinsic_gt[1]
    
    # compute BOP metrics
    mspd_err = my_mspd(pred_r, pred_t, gt_r, gt_t, k_nocs.reshape(3,3), obj_model['pts'], obj_sym)
    mssd_err = my_mssd(pred_r, pred_t, gt_r, gt_t, obj_model['pts'], obj_sym)

    # MSSD recalls depends on object diameters
    # MSPD instead is fixed, as it depends on the image size
    mssd_cur_rec = mssd_rec * obj_diam
    mean_mssd = (mssd_err < mssd_cur_rec).mean()
    mean_mspd = (mspd_err < mspd_rec).mean()
    out['MSSD'] = mean_mssd
    out['MSPD'] = mean_mspd
    # VSD is special because of multiple recalls
    vsd_errs = vsd(pred_r, pred_t, gt_r, gt_t, depths_full[1], k_nocs.reshape(3,3), vsd_delta, vsd_taus, True, obj_diam, renderer, obj_id)
    vsd_errs = np.asarray(vsd_errs)
    all_vsd_recs = np.stack([vsd_errs < rec_i  for rec_i in vsd_rec],axis=1)
    mean_vsd = all_vsd_recs.mean()
    out['VSD'] = mean_vsd
    out['AR'] = (mean_mssd + mean_mspd + mean_vsd)/3.

    r_error, t_error = compute_RT_distances(pred_pose.astype(np.float64), gt_pose.astype(np.float64))
    out['r_deg'] = float(r_error[0]) if hasattr(r_error, '__len__') else float(r_error)
    out['t_cm'] = float(t_error[0]) if hasattr(t_error, '__len__') else float(t_error)
    out['pred_pose'] = pred_pose
    out['gt_pose'] = gt_pose
    return out

def get_obj_rendering(root: str, obj_id: int) -> dict:
    '''
    returns object usable for vispy rendering
    argument obj_model is expected to have the following fields:
      - pts : (N,3) xyz points in mm
      - normals: (N,3) normals
      - faces: (M,3) polygon faces needed for rendering 
    '''

    pcd = PlyData.read(os.path.join(root,'models_bop','obj_{:06d}.ply'.format(obj_id)))
    # these are already in mm    
    xs = pcd['vertex']['x']
    ys = pcd['vertex']['y']
    zs = pcd['vertex']['z']
    nxs = pcd['vertex']['nx']
    nys = pcd['vertex']['ny']
    nzs = pcd['vertex']['nz']

    raw_vertexs = np.asarray(pcd['face']['vertex_indices'])
    faces = np.stack([vert for vert in raw_vertexs],axis=0)

    xyz = np.stack((xs,ys,zs), axis=1)
    normals = np.stack((nxs,nys,nzs), axis=1)
    
    return {
        'pts': xyz,
        'normals': normals,
        'faces': faces
    }

def get_obj_data(root: str) -> Tuple[dict, dict, dict]:
    '''
    Returns complete information about a dataset point clouds
    '''
    obj_files = [file for file in os.listdir(os.path.join(root, 'models_bop')) if '.ply' in file]
    obj_models, obj_diams, obj_symm = dict(), dict(), dict()

    with open(os.path.join(root, 'models_bop', 'models_info.json')) as f:
        models_info = json.load(f)

    for obj_file in obj_files:
        obj_id = int(os.path.splitext(obj_file[4:])[0])
        model_info = models_info[str(obj_id)]
        
        obj_models[obj_id] = get_obj_rendering(root, obj_id)
        obj_diams[obj_id] = model_info['diameter']
        from bop_toolkit_lib.misc import get_symmetry_transformations
        obj_symm[obj_id] = get_symmetry_transformations(model_info, max_sym_disc_step=0.05)
        
    return (obj_models, obj_diams, obj_symm)

def get_obj_info(obj_id, obj_models, obj_diams, obj_symms):
    '''
    Get object ID, object model, object diameter, and object symmetry transformations
    Args:
        obj_id: Object ID
        obj_models: Object models
        obj_diams: Object diameters
        obj_symms: Object symmetry transformations
    Returns:
        Object model, object diameter, and object symmetry transformations
    '''
    return obj_models[obj_id], obj_diams[obj_id], obj_symms[obj_id]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, required=True, help='Path to the Oryon TOYL dataset root')
    parser.add_argument('--scene', type=str, default=None, help='Optional single scene name to process')
    parser.add_argument('--output', type=str, default='outputs/toyl_rel',
                        help='Output directory, or an explicit .xlsm/.xlsx filename.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the relative-pose checkpoint')
    parser.add_argument('--disable_icp_refine', action='store_true',
                        help='Disable query-pose ICP refinement against the query depth cloud.')
    parser.add_argument('--icp_voxel_size', type=float, default=0.005,
                        help='Voxel size in meters used to downsample ICP point clouds.')
    parser.add_argument('--icp_max_corr', type=float, default=0.02,
                        help='Maximum correspondence distance in meters for ICP.')
    parser.add_argument('--icp_depth_gate', type=float, default=0.02,
                        help='Depth consistency gate in meters used to select visible model points for ICP.')
    parser.add_argument('--icp_max_iter', type=int, default=50,
                        help='Maximum number of ICP iterations per stage.')
    parser.add_argument('--icp_min_points', type=int, default=128,
                        help='Minimum number of source/target points required to run ICP.')
    parser.add_argument('--icp_max_points', type=int, default=6000,
                        help='Maximum number of source/target points kept before Open3D downsampling.')
    parser.add_argument('--crop_size', type=int, default=518,
                        help='Runtime crop size for the anchor/query object views.')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of DataLoader workers used for runtime pair construction.')
    args = parser.parse_args()
    k_nocs_path = os.path.join(args.base_dir, 'camera.json')
    k_nocs_data = json.load(open(k_nocs_path))
    k_nocs = np.eye(3)
    k_nocs[0, 0] = k_nocs_data['fx']
    k_nocs[1, 1] = k_nocs_data['fy']
    k_nocs[0, 2] = k_nocs_data['cx']
    k_nocs[1, 2] = k_nocs_data['cy']

    obj_models, obj_diams, obj_symms = get_obj_data(args.base_dir)
    obj_symms = {k: format_sym_set(sym_set) for k, sym_set in obj_symms.items()}
    renderer = RendererVispy(640, 480, mode='depth')
    vsd_taus = list(np.arange(0.05, 0.51, 0.05))
    vsd_rec = np.arange(0.05, 0.51, 0.05)
    vsd_delta = 15.0
    mssd_rec = np.arange(0.05, 0.51, 0.05)
    mspd_rec = np.arange(5, 51, 5)
    vsd_taus = list(np.arange(0.05, 0.51, 0.05))
    vsd_rec = np.arange(0.05, 0.51, 0.05)
    vsd_delta = 15.
    for obj_id, obj in obj_models.items():
        renderer.my_add_object(obj, obj_id)

    ts = dt.now().strftime('%Y%m%d_%H%M%S')
    if os.path.splitext(args.output)[1].lower() in {'.xlsm', '.xlsx'}:
        output = args.output
        output_dir = os.path.dirname(output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    else:
        os.makedirs(args.output, exist_ok=True)
        output = os.path.join(args.output, f'scene_rt_errors_toyl_{ts}.xlsm')

    # Load model once
    model, device = load_model(checkpoint_path=args.checkpoint)
    if torch.cuda.is_available():
        dev_cap = torch.cuda.get_device_capability()[0]
        dtype = torch.bfloat16 if dev_cap >= 8 else torch.float16
    else:
        dtype = torch.float32

    dataset, loader = build_runtime_pair_loader(
        dataset_name='toyl',
        base_dir=args.base_dir,
        split='test',
        scene=args.scene,
        target_image_shape=(args.crop_size, args.crop_size),
        num_workers=args.num_workers,
    )
    if len(dataset) == 0:
        raise RuntimeError(f'No runtime pairs found under {args.base_dir}')

    results = []
    icp_cfg = {
        'enabled': not args.disable_icp_refine,
        'voxel_size': args.icp_voxel_size,
        'max_correspondence_distance': args.icp_max_corr,
        'depth_gate': args.icp_depth_gate,
        'max_iterations': args.icp_max_iter,
        'min_points': args.icp_min_points,
        'max_points': args.icp_max_points,
    }
    iter_pairs = tqdm(loader, total=len(dataset), desc="Processing pairs") if len(dataset) > 1 else loader
    for sample in iter_pairs:
        print(f"Processing pair: {sample['scene']}")
        res = process_pair_sample(sample, model, device, dtype, obj_models, obj_diams, obj_symms,
                           renderer, vsd_delta, vsd_taus, vsd_rec, mssd_rec, mspd_rec, icp_cfg=icp_cfg)
        print(f"  -> r_deg={res['r_deg']}, t_cm={res['t_cm']}, ADD(S)-0.1d={res['ADD(S)-0.1d']}, MSSD={res['MSSD']}, MSPD={res['MSPD']}, VSD={res['VSD']}, AR={res['AR']}")
        results.append(res)

    # Save to Excel
    df = pd.DataFrame(results)
    df.to_excel(output, index=False)
    print(f"Wrote results to {output}")
    csv_out = os.path.splitext(output)[0] + '.csv'
    df.to_csv(csv_out, index=False)
    print(f"Wrote results to {csv_out}")

    summary_text, summary_txt, summary_log = write_runtime_summary(
        output,
        exp_tag=os.path.splitext(os.path.basename(__file__))[0],
        df=df,
    )
    print("\nEvaluation summary:")
    print(summary_text, end="")
    print(f"Wrote summary to {summary_txt}")
    print(f"Wrote summary to {summary_log}")

    print('Done.')
