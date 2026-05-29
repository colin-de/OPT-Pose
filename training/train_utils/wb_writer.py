# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Project-owned portions of this file are licensed under CC-BY-NC-SA-4.0.
# See LICENSE and NOTICE for details. Third-party notices remain applicable.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, Optional
from dataclasses import is_dataclass, asdict
import inspect

from .distributed import get_machine_local_and_dist_rank


def _make_serializable(obj: Any) -> Any:
    """Convert complex objects to serializable values for wandb config."""
    if obj is None:
        return None
    elif isinstance(obj, (int, float, str, bool)):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: _make_serializable(value) for key, value in obj.items()}
    elif is_dataclass(obj):
        try:
            # Convert dataclass to dict, but handle non-serializable fields
            result = {}
            for field_name, field_value in asdict(obj).items():
                try:
                    result[field_name] = _make_serializable(field_value)
                except Exception:
                    # If conversion fails, try to get a string representation
                    result[field_name] = str(field_value)
            return result
        except Exception:
            return str(obj)
    elif hasattr(obj, '__dict__'):
        # For other objects, try to get their attributes
        try:
            result = {}
            for attr_name in dir(obj):
                if not attr_name.startswith('_'):
                    try:
                        attr_value = getattr(obj, attr_name)
                        if not callable(attr_value):
                            result[attr_name] = _make_serializable(attr_value)
                    except Exception:
                        pass
            return result
        except Exception:
            return str(obj)
    else:
        # For anything else, try to convert to string
        return str(obj)


class WandbLogger:
    """A minimal Weights & Biases logger with distributed training support.

    Only writes from rank 0 to avoid duplicate logs across processes.
    API: log, log_dict, log_visuals.
    """

    def __init__(
        self,
        path: str,
        *args: Any,
        project: str = "nocs3r",
        run_name: str = None,
        entity: str = None,
        config: Optional[Dict[str, Any]] = None,
        mode: str = None,
        **kwargs: Any,
    ) -> None:
        self._writer = None
        _, self._rank = get_machine_local_and_dist_rank()
        self._path = path
        if self._rank == 0:
            try:
                import wandb
            except Exception as e:
                logging.error(f"wandb import failed: {e}. Falling back to no-op logger.")
                self._writer = None
                return

            init_kwargs: Dict[str, Any] = {
                "project": project,
                "dir": path,
            }
            if run_name is not None:
                init_kwargs["name"] = run_name
            if entity is not None:
                init_kwargs["entity"] = entity
            if config is not None:
                # Convert config to serializable values for wandb
                init_kwargs["config"] = _make_serializable(config)
            if mode is not None:
                init_kwargs["mode"] = mode

            logging.info(
                f"Weights & Biases run initialized. Project: {project}, dir: {path}"
            )
            self._writer = wandb.init(**init_kwargs)
            
            # Log system information and setup
            if self._writer is not None:
                try:
                    import torch
                    import platform
                    system_info = {
                        "python_version": platform.python_version(),
                        "torch_version": torch.__version__,
                        "cuda_available": torch.cuda.is_available(),
                        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                        "platform": platform.platform(),
                    }
                    wandb.config.update({"system_info": system_info})
                except Exception as e:
                    logging.debug(f"Failed to log system info: {e}")

    @property
    def writer(self):
        return self._writer

    @property
    def path(self) -> str:
        return self._path

    def flush(self) -> None:
        pass

    def close(self) -> None:
        if self._writer is not None:
            try:
                import wandb
                wandb.finish(quiet=True)
            except Exception:
                pass
            self._writer = None

    def log_dict(self, payload: Dict[str, Any], step: int) -> None:
        if self._writer is None:
            return
        try:
            import wandb
            wandb.log(payload, step=step)
        except Exception as e:
            logging.debug(f"wandb.log failed: {e}")

    def log(self, name: str, data: Any, step: int) -> None:
        if self._writer is None:
            return
        try:
            import wandb
            if hasattr(data, "item") and not isinstance(data, (int, float)):
                data = data.item()
            wandb.log({name: data}, step=step)
        except Exception as e:
            logging.debug(f"wandb.log scalar failed: {e}")

    def log_visuals(
        self,
        name: str,
        data: Any,
        step: int,
        fps: int = 4,
    ) -> None:
        if self._writer is None:
            return
        try:
            import wandb
            import numpy as np
            import torch as _torch

            if isinstance(data, _torch.Tensor):
                data_np = data.detach().cpu().numpy()
            else:
                data_np = data

            if data_np.ndim == 3:
                # Handle single image: (C, H, W) or (H, W, C)
                if data_np.shape[0] <= 4:  # Assume channels first if first dim <= 4
                    data_np = data_np.transpose(1, 2, 0)
                # Ensure values are in [0, 1] or [0, 255] range
                if data_np.max() <= 1.0:
                    data_np = (data_np * 255).astype(np.uint8)
                wandb.log({name: wandb.Image(data_np)}, step=step)
            elif data_np.ndim == 4:
                # Handle batch of images: (B, C, H, W)
                if data_np.shape[1] <= 4:  # Assume channels are second dim
                    data_np = data_np.transpose(0, 2, 3, 1)
                # Log only first image from batch
                if data_np.max() <= 1.0:
                    data_np = (data_np * 255).astype(np.uint8)
                wandb.log({name: wandb.Image(data_np[0])}, step=step)
            elif data_np.ndim == 5:
                # Handle video: (T, B, C, H, W)
                T, B, C, H, W = data_np.shape
                video = data_np.transpose(0, 1, 3, 4, 2)
                if video.max() <= 1.0:
                    video = (video * 255).astype(np.uint8)
                wandb.log({name: wandb.Video(video[:, 0], fps=fps, format="mp4")}, step=step)
            else:
                logging.debug(f"Unsupported visuals ndim for wandb: {data_np.ndim}")
        except Exception as e:
            logging.debug(f"wandb.log visuals failed: {e}")

    def log_model_summary(self, model: Any) -> None:
        """Log model architecture summary to wandb."""
        if self._writer is None:
            return
        try:
            import wandb
            import torch
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            model_info = {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
            }
            
            wandb.config.update({"model_info": model_info})
            logging.info(f"Logged model info to wandb: {model_info}")
            
        except Exception as e:
            logging.debug(f"Failed to log model summary: {e}")


