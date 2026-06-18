import os
import threading
from collections import OrderedDict

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import ListConfig
from PIL import Image
from torch.utils.data import Dataset


class McDataset(Dataset):
    """
    Optimized Dataset for MC's HDF5 format with trajectories stored as groups.
    Images are stored as individual PNG files in 'obs/{traj_idx}/{frame_idx}.png'.
    """

    IGNORE_INDEX = -100
    POSE_CENTER_X = 1.0
    POSE_CENTER_Y = 1.0

    def __init__(
        self,
        data_dir,
        max_length=64,
        cache_size=0,
        scale=1.5,
        size=64,
        full_trajectory=False,
        preload_valid_keys=True,
        stage="train",
        pose_center=1.0,
        load_images=True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.img_dir = os.path.join(data_dir, "obs")
        self.info_path = os.path.join(data_dir, "info.hdf5")

        self.max_length = max_length
        if isinstance(scale, (list, tuple, ListConfig)) and len(scale) == 2:
            self.scale_x = float(scale[0])
            self.scale_y = float(scale[1])
        else:
            self.scale_x = float(scale)
            self.scale_y = float(scale)
        if isinstance(pose_center, (list, tuple, ListConfig)) and len(pose_center) == 2:
            self.pose_center_x = float(pose_center[0])
            self.pose_center_y = float(pose_center[1])
        else:
            self.pose_center_x = float(pose_center)
            self.pose_center_y = float(pose_center)
        self.cache_size = cache_size
        self.size = size
        self.full_trajectory = full_trajectory
        self.stage = stage
        self.load_images = bool(load_images)
        self.img_dir_exists = os.path.isdir(self.img_dir)

        self.cache = OrderedDict()
        self._cache_lock = threading.RLock()

        if not os.path.exists(self.info_path):
            raise FileNotFoundError(f"HDF5 info file not found: {self.info_path}")
        if self.load_images and not self.img_dir_exists:
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        self._initialize_valid_trajectories(preload_valid_keys)

        print(f"McDataset initialized with {len(self.valid_traj_keys)} valid trajectories")

    def _initialize_valid_trajectories(self, preload_valid_keys=True):
        """Initialize valid trajectory keys with optimized file existence checking."""
        with h5py.File(self.info_path, "r") as h5_file:
            all_traj_keys = sorted(h5_file["trajectories"].keys())

        if not self.img_dir_exists:
            self.valid_traj_keys = list(all_traj_keys)
            self.traj_key_to_idx = {key: int(key.split("_")[1]) for key in self.valid_traj_keys}
            print(
                "Image directory not found. Using all trajectories from HDF5 metadata only."
            )
            print(
                f"Found {len(self.valid_traj_keys)} valid trajectories out of {len(all_traj_keys)} total trajectories"
            )
            return

        if preload_valid_keys:
            try:
                existing_indices = set()
                with os.scandir(self.img_dir) as it:
                    for entry in it:
                        if entry.is_dir() and entry.name.isdigit():
                            existing_indices.add(int(entry.name))
            except OSError as e:
                print(f"Warning: Could not read image directory {self.img_dir}: {e}")
                existing_indices = set()

            self.valid_traj_keys = []
            for key in all_traj_keys:
                try:
                    idx = int(key.split("_")[1])
                    if idx in existing_indices:
                        self.valid_traj_keys.append(key)
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse index from HDF5 key '{key}'. Skipping.")
        else:
            self.valid_traj_keys = []
            for key in all_traj_keys:
                try:
                    idx = int(key.split("_")[1])
                    traj_dir = os.path.join(self.img_dir, str(idx))
                    if os.path.isdir(traj_dir):
                        self.valid_traj_keys.append(key)
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse index from HDF5 key '{key}'. Skipping.")

        self.traj_key_to_idx = {key: int(key.split("_")[1]) for key in self.valid_traj_keys}

        total_keys = len(all_traj_keys)
        valid_keys = len(self.valid_traj_keys)
        print(f"Found {valid_keys} valid trajectories out of {total_keys} total trajectories")

    def _load_trajectory_data(self, traj_key):
        """Thread-safe trajectory data loading with on-demand HDF5 access."""
        try:
            with h5py.File(self.info_path, "r") as h5_file:
                traj_data = h5_file["trajectories"][traj_key]["data"][:]

            return {
                "x": traj_data["x"].astype(np.float32),
                "y": traj_data["y"].astype(np.float32),
                "theta": traj_data["theta"].astype(np.float32),
                "action": traj_data["action"].astype(np.float32),
            }
        except Exception as e:
            raise RuntimeError(f"Failed to load trajectory data for {traj_key}: {e}")

    def _load_images_sequence(self, traj_idx, start_idx, end_idx):
        """Load a sequence of images from PNG files."""
        images = []
        traj_dir = os.path.join(self.img_dir, str(traj_idx))

        for i in range(start_idx, end_idx):
            img_path = os.path.join(traj_dir, f"{i:04d}.png")
            try:
                with Image.open(img_path) as img:
                    img_np = np.array(img.convert("RGB"))
                    images.append(img_np)
            except Exception as e:
                raise RuntimeError(f"Failed to load image {img_path}: {e}")

        if not images:
            return np.zeros((0, self.size, self.size, 3), dtype=np.uint8)

        return np.stack(images)

    def _process_images_vectorized(self, img_raw):
        """Vectorized image preprocessing using PyTorch operations."""
        img_processed = img_raw.astype(np.float32) / 127.5 - 1.0

        if img_processed.shape[1:3] != (self.size, self.size):
            img_tensor = torch.from_numpy(img_processed).permute(0, 3, 1, 2).contiguous()
            img_resized = F.interpolate(
                img_tensor,
                size=(self.size, self.size),
                mode="bilinear",
                align_corners=False,
            )
            img_processed = img_resized.numpy()
        else:
            img_processed = np.transpose(img_processed, (0, 3, 1, 2))

        return img_processed

    def _get_from_cache(self, cache_key):
        """Thread-safe cache retrieval."""
        if self.cache_size <= 0:
            return None

        with self._cache_lock:
            if cache_key in self.cache:
                value = self.cache.pop(cache_key)
                self.cache[cache_key] = value
                return value
        return None

    def _put_to_cache(self, cache_key, value):
        """Thread-safe cache storage with LRU eviction."""
        if self.cache_size <= 0:
            return

        with self._cache_lock:
            if cache_key in self.cache:
                del self.cache[cache_key]

            while len(self.cache) >= self.cache_size:
                self.cache.popitem(last=False)

            self.cache[cache_key] = value

    def __len__(self):
        return len(self.valid_traj_keys)

    def __getitem__(self, idx):
        """Optimized data loading with improved error handling."""
        if idx >= len(self.valid_traj_keys):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self.valid_traj_keys)}"
            )

        traj_key = self.valid_traj_keys[idx]
        actual_idx = self.traj_key_to_idx[traj_key]

        cached_result = self._get_from_cache(actual_idx)
        if cached_result is not None:
            return cached_result

        traj_data = self._load_trajectory_data(traj_key)

        pose_raw_np = np.column_stack([
            traj_data["x"],
            traj_data["y"],
            traj_data["theta"],
        ])
        actions = traj_data["action"]

        if self.load_images:
            traj_dir = os.path.join(self.img_dir, str(actual_idx))
            if not os.path.exists(traj_dir):
                num_images = 0
            else:
                num_images = 0
                with os.scandir(traj_dir) as it:
                    for entry in it:
                        if entry.is_file() and entry.name.endswith(".png"):
                            num_images += 1

            current_raw_length = min(len(pose_raw_np), num_images)
        else:
            num_images = None
            current_raw_length = len(pose_raw_np)

        pose_raw_np = pose_raw_np[:current_raw_length]
        actions = actions[:current_raw_length]

        if current_raw_length == 0:
            image_info = num_images if num_images is not None else "skipped"
            raise RuntimeError(
                f"Trajectory {traj_key} has 0 valid frames (pose={len(traj_data['x'])}, imgs={image_info})"
            )

        if self.full_trajectory:
            start_crop, end_crop = 0, current_raw_length
        elif current_raw_length > self.max_length:
            if self.stage == "train":
                start_crop = np.random.randint(0, current_raw_length - self.max_length + 1)
                end_crop = start_crop + self.max_length
            else:
                start_crop = 0
                end_crop = self.max_length
        else:
            start_crop, end_crop = 0, current_raw_length

        pose_selected = pose_raw_np[start_crop:end_crop]
        action_selected = actions[start_crop:end_crop]

        action_np = action_selected - 1
        if len(action_np) > 1:
            action_np[:-1] = action_np[1:]
        action_np[-1] = self.IGNORE_INDEX

        new_pose_np = np.zeros((pose_selected.shape[0], 4), dtype=np.float32)
        if pose_selected.shape[0] > 0:
            theta_rad = pose_selected[:, 2] * np.pi / 180.0
            new_pose_np[:, 0] = pose_selected[:, 0] / self.scale_x - self.pose_center_x
            new_pose_np[:, 1] = pose_selected[:, 1] / self.scale_y - self.pose_center_y
            new_pose_np[:, 2] = np.cos(theta_rad)
            new_pose_np[:, 3] = np.sin(theta_rad)

        result = {
            "pose": torch.tensor(new_pose_np, dtype=torch.float32),
            "action": torch.tensor(action_np, dtype=torch.float32),
        }
        if self.load_images:
            img_selected = self._load_images_sequence(actual_idx, start_crop, end_crop)
            img_processed = self._process_images_vectorized(img_selected)
            result["image"] = torch.tensor(img_processed, dtype=torch.float32)

        self._put_to_cache(actual_idx, result)

        return result

    def clear_cache(self):
        """Clear the internal cache manually."""
        with self._cache_lock:
            self.cache.clear()
        print("Dataset cache cleared")

    def get_cache_info(self):
        """Get cache statistics."""
        with self._cache_lock:
            return {
                "cache_size": len(self.cache),
                "cache_limit": self.cache_size,
                "cached_keys": list(self.cache.keys()),
            }

    def __repr__(self):
        return (
            f"McDataset(data_dir='{self.data_dir}', "
            f"num_trajectories={len(self.valid_traj_keys)}, "
            f"max_length={self.max_length}, "
            f"cache_size={self.cache_size}, "
            f"image_size={self.size}, "
            f"load_images={self.load_images})"
        )
