# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Project-owned portions of this file are licensed under CC-BY-NC-SA-4.0.
# See LICENSE and NOTICE for details. Third-party notices remain applicable.

import json
import os
import pickle
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from training.data.datasets import misc
from training.data.datasets.structs import CameraModel


def _first_item_collate(batch: List[Dict]) -> Dict:
    return batch[0]


def _read_rgb(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _read_gray(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"))


def _read_depth(path: str) -> np.ndarray:
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Failed to read depth image: {path}")
    return depth.astype(np.float32)


def _ensure_rigid_transform(matrix: np.ndarray) -> np.ndarray:
    mat4 = np.asarray(matrix, dtype=np.float64)
    if mat4.shape == (3, 4):
        tmp = np.eye(4, dtype=np.float64)
        tmp[:3, :] = mat4
        mat4 = tmp
    elif mat4.shape != (4, 4):
        raise ValueError(f"Expected a 3x4 or 4x4 pose matrix, got {mat4.shape}")

    u, _, vt = np.linalg.svd(mat4[:3, :3])
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1
        rot = u @ vt

    rigid = np.eye(4, dtype=np.float64)
    rigid[:3, :3] = rot
    rigid[:3, 3] = mat4[:3, 3]
    return rigid.astype(np.float32)


def _camera_from_json(camera_json_path: str, fallback: np.ndarray) -> np.ndarray:
    if not os.path.isfile(camera_json_path):
        return fallback.astype(np.float32)

    with open(camera_json_path, "r") as f:
        data = json.load(f)

    if "cam_K" in data:
        return np.asarray(data["cam_K"], dtype=np.float32).reshape(3, 3)

    if {"fx", "fy", "cx", "cy"}.issubset(data.keys()):
        return np.array(
            [
                [data["fx"], 0.0, data["cx"]],
                [0.0, data["fy"], data["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    return fallback.astype(np.float32)


def _crop_view(
    image: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    obj_to_cam: np.ndarray,
    target_image_shape: Tuple[int, int],
    crop_rel_pad: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obj_to_cam = _ensure_rigid_transform(obj_to_cam)
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    full_depth_masked = np.asarray(depth).copy()
    full_depth_masked[mask_u8 == 0] = 0

    orig_camera_c2w = CameraModel(
        width=image.shape[1],
        height=image.shape[0],
        f=(float(intrinsics[0, 0]), float(intrinsics[1, 1])),
        c=(float(intrinsics[0, 2]), float(intrinsics[1, 2])),
        T_world_from_eye=np.linalg.inv(obj_to_cam),
    )

    crop_box = misc.calc_crop_box(
        box=misc.get_bbox(mask_u8),
        make_square=True,
    )
    crop_camera_model_c2w = misc.construct_crop_camera(
        box=crop_box,
        camera_model_c2w=orig_camera_c2w,
        viewport_size=target_image_shape,
        viewport_rel_pad=crop_rel_pad,
    )

    image_crop = misc.warp_image(
        src_camera=orig_camera_c2w,
        dst_camera=crop_camera_model_c2w,
        src_image=image,
        interpolation=cv2.INTER_LINEAR,
    )
    mask_crop = misc.warp_image(
        src_camera=orig_camera_c2w,
        dst_camera=crop_camera_model_c2w,
        src_image=mask_u8,
        interpolation=cv2.INTER_NEAREST,
    )
    depth_crop = misc.warp_depth_image(
        src_camera=orig_camera_c2w,
        dst_camera=crop_camera_model_c2w,
        src_depth_image=depth,
    )

    mask_crop = (mask_crop > 0).astype(np.uint8)
    image_crop_masked = image_crop.copy()
    image_crop_masked[mask_crop == 0] = 0

    pose_crop = np.linalg.inv(crop_camera_model_c2w.T_world_from_eye).astype(np.float32)
    # Transform points from crop-camera frame to original-camera frame.
    T_orig_crop = (obj_to_cam @ np.linalg.inv(pose_crop)).astype(np.float32)
    fx, fy = crop_camera_model_c2w.f
    cx, cy = crop_camera_model_c2w.c
    intrinsics_crop = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    return (
        image_crop_masked.astype(np.uint8),
        mask_crop,
        np.asarray(depth_crop, dtype=np.float32),
        np.asarray(full_depth_masked, dtype=np.float32),
        intrinsics_crop,
        pose_crop,
        T_orig_crop,
    )


class _BaseRuntimePairDataset(Dataset):
    def __init__(
        self,
        base_dir: str,
        split: str,
        target_image_shape: Tuple[int, int],
        scene: Optional[str] = None,
    ):
        self.base_dir = os.path.abspath(base_dir)
        self.split = split
        self.target_image_shape = tuple(int(v) for v in target_image_shape)
        self.pairs = self._load_pairs()
        self.scene_names = [self._instance_id(pair) for pair in self.pairs]

        if scene is not None:
            filtered_pairs = [
                pair for pair in self.pairs if self._instance_id(pair) == scene
            ]
            if not filtered_pairs:
                raise ValueError(f"Pair '{scene}' was not found under {self.base_dir}")
            self.pairs = filtered_pairs
            self.scene_names = [scene]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Dict:
        return self._load_pair_sample(self.pairs[index])

    def _load_pairs(self) -> List[Tuple]:
        raise NotImplementedError

    def _instance_id(self, pair: Sequence) -> str:
        raise NotImplementedError

    def _load_pair_sample(self, pair: Sequence) -> Dict:
        raise NotImplementedError


class NOCSRuntimePairDataset(_BaseRuntimePairDataset):
    def __init__(
        self,
        base_dir: str,
        split: str = "test",
        target_image_shape: Tuple[int, int] = (518, 518),
        scene: Optional[str] = None,
    ):
        self.gt_pose_paths = self._index_nocs_gt_pose_paths(base_dir)
        fallback_K = np.array(
            [
                [591.0125, 0.0, 322.525],
                [0.0, 590.16775, 244.11084],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.K_original = _camera_from_json(
            os.path.join(base_dir, "camera.json"),
            fallback=fallback_K,
        )
        super().__init__(
            base_dir=base_dir,
            split=split,
            target_image_shape=target_image_shape,
            scene=scene,
        )

    @staticmethod
    def _index_nocs_gt_pose_paths(base_dir: str) -> Dict[Tuple[int, int], str]:
        gt_dir = os.path.join(base_dir, "gts", "real_test")
        pose_paths: Dict[Tuple[int, int], str] = {}
        if not os.path.isdir(gt_dir):
            return pose_paths

        for filename in os.listdir(gt_dir):
            stem, ext = os.path.splitext(filename)
            if ext.lower() != ".pkl":
                continue
            parts = stem.split("_")
            if len(parts) < 2:
                continue
            try:
                scene_id = int(parts[-2])
                img_id = int(parts[-1])
            except ValueError:
                continue
            pose_paths[(scene_id, img_id)] = os.path.join(gt_dir, filename)

        return pose_paths

    def _load_pairs(self) -> List[Tuple]:
        # path_split = os.path.join(self.base_dir, "fixed_split", self.split, "instance_list.txt")
        path_split = os.path.join(self.base_dir, "fixed_split", "cross_scene_test", "instance_list.txt")
        pairs: List[Tuple[int, int, int, int, str]] = []
        with open(path_split, "r") as f:
            for line in f:
                _, idx_a, idx_q, cat = line.split(",")
                _, obj_name = cat.strip().split(" ")
                scene_a, img_a = [int(v) for v in idx_a.split(" ") if v]
                scene_q, img_q = [int(v) for v in idx_q.split(" ") if v]
                pairs.append((scene_a, img_a, scene_q, img_q, obj_name))
        return pairs

    def _instance_id(self, pair: Sequence) -> str:
        scene_a, img_a, scene_q, img_q, obj_name = pair
        return f"{scene_a}_{img_a}_{scene_q}_{img_q}_{obj_name}"

    def _load_nocs_pose_and_mask_id(
        self,
        scene_id: int,
        img_id: int,
        obj_name: str,
    ) -> Tuple[np.ndarray, int]:
        meta_path = os.path.join(
            self.base_dir,
            "real_test",
            f"scene_{scene_id}",
            f"{img_id:04d}_meta.txt",
        )
        pose_path = self.gt_pose_paths.get((scene_id, img_id))
        if pose_path is None:
            raise FileNotFoundError(f"Missing GT pose pickle for NOCS frame {(scene_id, img_id)}")

        with open(pose_path, "rb") as f:
            pose_data = pickle.load(f)["gt_RTs"]

        with open(meta_path, "r") as f:
            meta_lines = [line.strip().split() for line in f if line.strip()]

        obj_index = None
        mask_id = None
        for idx, parts in enumerate(meta_lines):
            if len(parts) >= 3 and parts[2] == obj_name:
                obj_index = idx
                mask_id = int(parts[0])
                break

        if obj_index is None or mask_id is None:
            raise ValueError(f"Object '{obj_name}' not found in {meta_path}")

        pose = np.asarray(pose_data[obj_index], dtype=np.float32).copy()
        pose[:3, :3] = pose[:3, :3] / np.linalg.norm(pose[:3, :3], axis=1, keepdims=True)
        return pose, mask_id

    def _load_view(self, scene_id: int, img_id: int, obj_name: str) -> Tuple[np.ndarray, ...]:
        base_path = os.path.join(
            self.base_dir,
            "real_test",
            f"scene_{scene_id}",
            f"{img_id:04d}",
        )
        pose, mask_id = self._load_nocs_pose_and_mask_id(scene_id, img_id, obj_name)
        image = _read_rgb(base_path + "_color.png")
        depth = _read_depth(base_path + "_depth.png")
        mask_full = _read_gray(base_path + "_mask.png")
        mask = (mask_full == mask_id).astype(np.uint8)
        return _crop_view(
            image=image,
            depth=depth,
            mask=mask,
            intrinsics=self.K_original,
            obj_to_cam=pose,
            target_image_shape=self.target_image_shape,
        )

    def _load_pair_sample(self, pair: Sequence) -> Dict:
        scene_a, img_a, scene_q, img_q, obj_name = pair
        (
            image_a,
            mask_a,
            depth_a,
            depth_full_a,
            K_a,
            pose_a,
            T_orig_crop_a,
        ) = self._load_view(scene_a, img_a, obj_name)
        (
            image_q,
            mask_q,
            depth_q,
            depth_full_q,
            K_q,
            pose_q,
            T_orig_crop_q,
        ) = self._load_view(scene_q, img_q, obj_name)

        return {
            "scene": self._instance_id(pair),
            "obj_id": obj_name,
            "images": [image_a, image_q],
            "masks": [mask_a, mask_q],
            "depths": [depth_a, depth_q],
            "depths_full": [depth_full_a, depth_full_q],
            "Ks": [K_a, K_q],
            "poses": [pose_a, pose_q],
            "T_orig_crop": [T_orig_crop_a, T_orig_crop_q],
        }


class TOYLRuntimePairDataset(_BaseRuntimePairDataset):
    def __init__(
        self,
        base_dir: str,
        split: str = "test",
        target_image_shape: Tuple[int, int] = (518, 518),
        scene: Optional[str] = None,
    ):
        fallback_K = _camera_from_json(
            os.path.join(base_dir, "camera.json"),
            fallback=np.array(
                [
                    [572.4114, 0.0, 325.2611],
                    [0.0, 573.5704, 242.0489],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
        )
        self.fallback_K = fallback_K
        self.scene_gt_cache: Dict[int, Dict] = {}
        self.scene_camera_cache: Dict[int, Dict] = {}
        super().__init__(
            base_dir=base_dir,
            split=split,
            target_image_shape=target_image_shape,
            scene=scene,
        )

    def _load_pairs(self) -> List[Tuple]:
        path_split = os.path.join(self.base_dir, "fixed_split", "cross_scene_test", "instance_list.txt")
        pairs: List[Tuple[int, int, int, int, int]] = []
        with open(path_split, "r") as f:
            for line in f:
                _, idx_a, idx_q, obj_id = line.strip("\n").split(",")
                scene_a, img_a = idx_a.strip().split(" ")
                scene_q, img_q = idx_q.strip().split(" ")
                pairs.append((int(scene_a), int(img_a), int(scene_q), int(img_q), int(obj_id.strip())))
        return pairs

    def _instance_id(self, pair: Sequence) -> str:
        scene_a, img_a, scene_q, img_q, cls_id = pair
        return f"{scene_a}_{img_a}_{scene_q}_{img_q}_{cls_id}"

    def _get_scene_gt(self, scene_id: int) -> Dict:
        if scene_id not in self.scene_gt_cache:
            scene_gt_path = os.path.join(
                self.base_dir,
                "split",
                "test",
                f"{scene_id:06d}",
                "scene_gt.json",
            )
            with open(scene_gt_path, "r") as f:
                self.scene_gt_cache[scene_id] = json.load(f)
        return self.scene_gt_cache[scene_id]

    def _get_scene_camera(self, scene_id: int) -> Dict:
        if scene_id not in self.scene_camera_cache:
            scene_camera_path = os.path.join(
                self.base_dir,
                "split",
                "test",
                f"{scene_id:06d}",
                "scene_camera.json",
            )
            if os.path.isfile(scene_camera_path):
                with open(scene_camera_path, "r") as f:
                    self.scene_camera_cache[scene_id] = json.load(f)
            else:
                self.scene_camera_cache[scene_id] = {}
        return self.scene_camera_cache[scene_id]

    def _get_frame_intrinsics(self, scene_id: int, img_id: int) -> np.ndarray:
        scene_camera = self._get_scene_camera(scene_id)
        cam_entry = scene_camera.get(str(int(img_id)), {})
        cam_K = cam_entry.get("cam_K")
        if cam_K is None:
            return self.fallback_K.copy()
        return np.asarray(cam_K, dtype=np.float32).reshape(3, 3)

    def _load_toyl_pose_and_mask_id(
        self,
        scene_id: int,
        img_id: int,
        cls_id: int,
    ) -> Tuple[np.ndarray, int]:
        frame_entries = self._get_scene_gt(scene_id).get(str(int(img_id)), [])
        obj_index = None
        pose = np.eye(4, dtype=np.float32)
        for idx, entry in enumerate(frame_entries):
            if int(entry.get("obj_id", -1)) != int(cls_id):
                continue
            pose[:3, :3] = np.asarray(entry["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
            pose[:3, 3] = np.asarray(entry["cam_t_m2c"], dtype=np.float32).reshape(3) / 1000.0
            obj_index = idx
            break

        if obj_index is None:
            raise ValueError(f"Object id {cls_id} not found in scene {scene_id}, frame {img_id}")

        return pose, obj_index + 1

    def _load_view(self, scene_id: int, img_id: int, cls_id: int) -> Tuple[np.ndarray, ...]:
        scene_dir = os.path.join(
            self.base_dir,
            "split",
            "test",
            f"{scene_id:06d}",
        )
        pose, mask_id = self._load_toyl_pose_and_mask_id(scene_id, img_id, cls_id)
        image = _read_rgb(os.path.join(scene_dir, "rgb", f"{img_id:06d}.png"))
        depth = _read_depth(os.path.join(scene_dir, "depth", f"{img_id:06d}.png"))
        mask_full = _read_gray(os.path.join(scene_dir, "mask_visib", f"{img_id:06d}.png"))
        mask = (mask_full == mask_id).astype(np.uint8)
        K_original = self._get_frame_intrinsics(scene_id, img_id)
        return _crop_view(
            image=image,
            depth=depth,
            mask=mask,
            intrinsics=K_original,
            obj_to_cam=pose,
            target_image_shape=self.target_image_shape,
        )

    def _load_pair_sample(self, pair: Sequence) -> Dict:
        scene_a, img_a, scene_q, img_q, cls_id = pair
        (
            image_a,
            mask_a,
            depth_a,
            depth_full_a,
            K_a,
            pose_a,
            T_orig_crop_a,
        ) = self._load_view(scene_a, img_a, cls_id)
        (
            image_q,
            mask_q,
            depth_q,
            depth_full_q,
            K_q,
            pose_q,
            T_orig_crop_q,
        ) = self._load_view(scene_q, img_q, cls_id)

        return {
            "scene": self._instance_id(pair),
            "obj_id": int(cls_id),
            "images": [image_a, image_q],
            "masks": [mask_a, mask_q],
            "depths": [depth_a, depth_q],
            "depths_full": [depth_full_a, depth_full_q],
            "Ks": [K_a, K_q],
            "poses": [pose_a, pose_q],
            "T_orig_crop": [T_orig_crop_a, T_orig_crop_q],
        }


def build_runtime_pair_loader(
    dataset_name: str,
    base_dir: str,
    split: str = "test",
    scene: Optional[str] = None,
    target_image_shape: Tuple[int, int] = (518, 518),
    num_workers: int = 0,
) -> Tuple[Dataset, DataLoader]:
    dataset_key = dataset_name.strip().lower()
    if dataset_key == "nocs":
        dataset = NOCSRuntimePairDataset(
            base_dir=base_dir,
            split=split,
            target_image_shape=target_image_shape,
            scene=scene,
        )
    elif dataset_key == "toyl":
        dataset = TOYLRuntimePairDataset(
            base_dir=base_dir,
            split=split,
            target_image_shape=target_image_shape,
            scene=scene,
        )
    else:
        raise ValueError(f"Unsupported runtime pair dataset: {dataset_name}")

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=_first_item_collate,
    )
    return dataset, loader
