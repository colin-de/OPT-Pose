# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Project-owned portions of this file are licensed under CC-BY-NC-SA-4.0.
# See LICENSE and NOTICE for details. Third-party notices remain applicable.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC
import json
import os

from hydra.utils import instantiate
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
import bisect
from .dataset_util import *
from .augmentation import get_image_augmentation


class ComposedDataset(Dataset, ABC):
    """
    Composes multiple base datasets and applies common configurations.

    This dataset provides a flexible way to combine multiple base datasets while
    applying shared augmentations, and other processing steps.
    It handles image normalization, tensor conversion, and other preparations
    needed for training computer vision models with sequences of images.
    """
    def __init__(self, dataset_configs: dict, common_config: dict, **kwargs):
        """
        Initializes the ComposedDataset.

        Args:
            dataset_configs (dict): List of Hydra configurations for base datasets.
            common_config (dict): Shared configurations (augs, mode, etc.).
            **kwargs: Additional arguments (unused).
        """
        base_dataset_list = []

        # Instantiate each base dataset with common configuration
        for baseset_dict in dataset_configs:
            baseset = instantiate(baseset_dict, common_conf=common_config)
            base_dataset_list.append(baseset)

        # Use custom concatenation class that supports tuple indexing and optional per-dataset weights
        self.base_dataset = TupleConcatDataset(base_dataset_list, common_config)

        # --- Augmentation Settings (safe access with defaults) ---
        augs = getattr(common_config, "augs", None)
        # Controls whether to apply identical color jittering across all frames in a sequence
        self.cojitter = getattr(augs, "cojitter", False)
        # Probability of using shared jitter vs. frame-specific jitter
        self.cojitter_ratio = getattr(augs, "cojitter_ratio", 0.5)
        # Initialize image augmentations (color jitter, grayscale, gaussian blur)
        self.image_aug = get_image_augmentation(
            color_jitter=getattr(augs, "color_jitter", 0.0),
            gray_scale=getattr(augs, "gray_scale", 0.0),
            gau_blur=getattr(augs, "gau_blur", 0.0),
        )

        # --- Optional Fixed Settings (useful for debugging) ---
        # Force each sequence to have exactly this many images (if > 0)
        self.fixed_num_images = getattr(common_config, "fix_img_num", 0)
        # Force a specific aspect ratio for all images
        self.fixed_aspect_ratio = getattr(common_config, "fix_aspect_ratio", 1.0)

        # --- Mode Settings ---
        # Whether the dataset is being used for training (affects augmentations)
        self.training = getattr(common_config, "training", True)
        self.common_config = common_config

        self.total_samples = len(self.base_dataset)
        
        # Load category name to ID mapping for NOCS/HouseCat datasets
        # This file maps category names to integer IDs used by the model
        cat_mapping_path = os.path.join(os.path.dirname(__file__), 'datasets', 'nocs_hc_cat_name_2_id.json')
        if os.path.exists(cat_mapping_path):
            with open(cat_mapping_path, 'r') as f:
                self.cat_name_2_id = json.load(f)
        else:
            # Default mapping if file not found
            self.cat_name_2_id = {
                "bottle": 0, "box": 1, "can": 2, "cup": 3, "remote": 4, 
                "teapot": 5, "cutlery": 6, "glass": 7, "tube": 8, "shoe": 9,
                "bowl": 10, "camera": 11, "laptop": 12, "mug": 3  # mug maps to same as cup
            }

    def __len__(self):
        """Returns the total number of sequences in the dataset."""
        return self.total_samples


    def __getitem__(self, idx_tuple):
        """
        Retrieves a data sample (sequence) from the dataset.

        Loads raw data, converts to PyTorch tensors, applies augmentations,
        and prepares other data if enabled.

        Args:
            idx_tuple (tuple): a tuple of (seq_idx, num_images, aspect_ratio)

        Returns:
            dict: A dictionary containing the sequence data (images, poses, etc.).
        """
        # If fixed settings are provided, override the tuple values
        if self.fixed_num_images > 0:
            seq_idx = idx_tuple[0] if isinstance(idx_tuple, tuple) else idx_tuple
            idx_tuple = (seq_idx, self.fixed_num_images, self.fixed_aspect_ratio)

        # Retrieve the raw data batch from the appropriate base dataset
        batch = self.base_dataset[idx_tuple]
        seq_name = batch["seq_name"]
        seq_id = batch["seq_id"]
        
        # Get category ID if category name is provided
        cat_name = batch.get("cat_name", None)
        cat_id = None
        if cat_name is not None and cat_name in self.cat_name_2_id:
            cat_id = self.cat_name_2_id[cat_name]
        elif cat_name is not None:
            # If cat_name not in mapping, use a default ID or log warning
            # print(f"Warning: Unknown category '{cat_name}', using default ID 0")
            cat_id = 0

        # Check for empty batch and handle it
        if len(batch["images"]) == 0:
            # Return None to indicate this sample should be skipped
            return None

        # --- Data Conversion and Preparation ---
        # Convert numpy arrays to tensors
        images = torch.from_numpy(np.stack(batch["images"]).astype(np.float32)).contiguous()
        # Normalize images from [0, 255] to [0, 1]
        images = images.permute(0,3,1,2).to(torch.get_default_dtype()).div(255)

        # Convert other data to tensors with appropriate types
        depths = torch.from_numpy(np.stack(batch["depths"]).astype(np.float32))
        depths_sensor = torch.from_numpy(np.stack(batch["depths_sensor"]).astype(np.float32))
        extrinsics = torch.from_numpy(np.stack(batch["extrinsics_sym"]).astype(np.float32))
        extrinsics_sym = torch.from_numpy(np.stack(batch["extrinsics_sym"]).astype(np.float32))
        intrinsics = torch.from_numpy(np.stack(batch["intrinsics"]).astype(np.float32))
        cam_points = torch.from_numpy(np.stack(batch["cam_points"]).astype(np.float32))
        world_points = torch.from_numpy(np.stack(batch["world_points"]).astype(np.float32))
        point_masks = torch.from_numpy(np.stack(batch["point_masks"])) # Mask indicating valid depths / world points / cam points per frame
        ids = torch.from_numpy(batch["ids"])    # Frame indices sampled from the original sequence
        sizes = torch.from_numpy(np.stack(batch["sizes"]).astype(np.float32))  # Object sizes


        # --- Apply Color Augmentation (training mode only) ---
        if self.training and self.image_aug is not None:
            if self.cojitter and random.random() > self.cojitter_ratio:
                # Apply the same color jittering transformation to all frames
                images = self.image_aug(images)
            else:
                # Apply different color jittering to each frame individually
                for aug_img_idx in range(len(images)):
                    images[aug_img_idx] = self.image_aug(images[aug_img_idx])


        # --- Prepare Final Sample Dictionary ---
        sample = {
            "seq_name": seq_name,
            "seq_id": seq_id,
            "ids": ids,
            "images": images,
            "depths": depths,
            "depths_sensor": depths_sensor,
            "extrinsics": extrinsics,
            "extrinsics_sym": extrinsics_sym,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "sizes": sizes,
        }
        
        if cat_name is not None:
            sample["category_name"] = cat_name

        if cat_id is not None:
            # Convert to tensor with same batch size as images
            B = len(images)  # Number of frames
            cat_id_tensor = torch.full((B,), cat_id, dtype=torch.long)
            sample["cat_id"] = cat_id_tensor
        if "nocs" in batch:
            nocs = torch.from_numpy(np.stack(batch["nocs"]).astype(np.float32))
            sample["nocs"] = nocs
        if "inst_masks" in batch:
            inst_masks = torch.from_numpy(np.stack(batch["inst_masks"]).astype(np.uint8)).bool()
            sample["inst_masks"] = inst_masks
        if "crop_boxes" in batch:
            crop_boxes = torch.from_numpy(np.stack(batch["crop_boxes"]).astype(np.float32))
            sample["crop_boxes"] = crop_boxes
        if "choose_indices" in batch:
            choose_indices = torch.from_numpy(np.stack(batch["choose_indices"]).astype(np.int64))
            sample["choose_indices"] = choose_indices
        if "sym_codes" in batch:
            sym_codes = batch["sym_codes"]
            if isinstance(sym_codes, (list, tuple)) and len(sym_codes) > 0:
                sample["sym_codes"] = torch.from_numpy(np.stack(sym_codes).astype(np.int64))
            else:
                sample["sym_codes"] = torch.as_tensor(np.asarray(sym_codes), dtype=torch.long)
        else:
            B = len(images)
            sample["sym_codes"] = torch.zeros((B, 3), dtype=torch.long)

        return sample


class TupleConcatDataset(ConcatDataset):
    """
    A custom ConcatDataset that supports indexing with a tuple.

    Standard PyTorch ConcatDataset only accepts an integer index. This class extends
    that functionality to allow passing a tuple like (sample_idx, num_images, aspect_ratio),
    where the first element is used to determine which sample to fetch, and the full
    tuple is passed down to the selected dataset's __getitem__ method.

    It also supports an option to randomly sample across all datasets, ignoring the
    provided index. This is useful during training when shuffling the entire dataset
    might cause memory issues due to duplicating dictionaries. If doing this, you can
    set pytorch's dataloader shuffle to False.
    """
    def __init__(self, datasets, common_config):
        """
        Initialize the TupleConcatDataset.

        Args:
            datasets (iterable): An iterable of PyTorch Dataset objects to concatenate.
            common_config (dict): Common configuration dict, used to check for random sampling.
        """
        super().__init__(datasets)
        # If True, ignores the input index and samples randomly across datasets
        # This provides an alternative to dataloader shuffling for large datasets
        self.inside_random = getattr(common_config, "inside_random", False)
        # Optional per-dataset sampling weights (len == len(datasets)); will be
        # normalized to sum to 1.0. When provided and inside_random is True, we
        # sample dataset indices with these probabilities to reduce dominance of
        # very large datasets.
        self.dataset_weights = getattr(common_config, "dataset_weights", None)
        if self.dataset_weights is not None:
            try:
                import numpy as _np
                w = _np.asarray(self.dataset_weights, dtype=_np.float64)
                assert w.ndim == 1 and len(w) == len(self.datasets), (
                    f"dataset_weights length {len(w)} must match number of datasets {len(self.datasets)}"
                )
                if (w < 0).any():
                    raise ValueError("dataset_weights must be non-negative")
                s = float(w.sum())
                if s <= 0:
                    # fallback to uniform
                    w = _np.ones_like(w) / len(w)
                else:
                    w = w / s
                self.dataset_weights = w.tolist()
            except Exception:
                # On any failure, ignore weights gracefully
                self.dataset_weights = None

    def __getitem__(self, idx):
        """
        Retrieves an item using either an integer index or a tuple index.

        Args:
            idx (int or tuple): The index. If tuple, the first element is the sequence
                               index across the concatenated datasets, and the rest are
                               passed down. If int, it's treated as the sequence index.

        Returns:
            The item returned by the underlying dataset's __getitem__ method.

        Raises:
            ValueError: If the index is out of range or the tuple doesn't have exactly 3 elements.
        """
        idx_tuple = None
        if isinstance(idx, tuple):
            idx_tuple = idx
            idx = idx_tuple[0]  # Extract the sequence index

        # Override index with randomized sampling when enabled
        if self.inside_random:
            if self.dataset_weights is not None:
                # Sample a dataset by weight, then sample a valid index within it
                ds_idx = random.choices(range(len(self.datasets)), weights=self.dataset_weights, k=1)[0]
                ds_len = len(self.datasets[ds_idx])
                if ds_len < 1:
                    raise ValueError(f"Dataset {ds_idx} has zero length")
                sample_idx = random.randint(0, ds_len - 1)
                dataset_idx = ds_idx
            else:
                # Uniform over all samples across datasets
                total_len = self.cumulative_sizes[-1]
                idx = random.randint(0, total_len - 1)
                dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
                if dataset_idx == 0:
                    sample_idx = idx
                else:
                    sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
            # Create the tuple to pass to the underlying dataset using provided params
            if isinstance(idx_tuple, tuple) and len(idx_tuple) == 3:
                idx_tuple = (sample_idx,) + idx_tuple[1:]
                return self.datasets[dataset_idx][idx_tuple]
            else:
                raise ValueError("Tuple index must have exactly three elements")

        # Handle negative indices
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx

        # Find which dataset the index belongs to
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        # Create the tuple to pass to the underlying dataset
        if len(idx_tuple) == 3:
            idx_tuple = (sample_idx,) + idx_tuple[1:]
        else:
            raise ValueError("Tuple index must have exactly three elements")

        # Pass the modified tuple to the appropriate dataset
        return self.datasets[dataset_idx][idx_tuple]


def collate_fn_filter_none(batch):
    """
    Custom collate function that filters out None values from the batch.
    This is needed when the dataset returns None for empty samples.
    """
    # Filter out None values
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        # If all items are None, return None to indicate empty batch
        return None
    
    # # Ensure all tensors are contiguous before collation
    # def make_contiguous(item):
    #     if isinstance(item, dict):
    #         return {key: make_contiguous(value) for key, value in item.items()}
    #     elif isinstance(item, torch.Tensor):
    #         return item.contiguous()
    #     else:
    #         return item
    
    # # Make all tensors contiguous
    # batch = [make_contiguous(item) for item in batch]
    
    # Use default collate for the remaining items
    from torch.utils.data.dataloader import default_collate
    return default_collate(batch)
