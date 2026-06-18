import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import ListConfig
from torch.utils.data import Dataset


class MazeDataset(Dataset):
    IGNORE_INDEX = -100
    MAZE_CENTER_RAW = 4.5

    def __init__(
        self,
        data_dir,
        max_length=64,
        cache_size=0,
        scale=1.5,
        stage="train",
        full_trajectory=False,
        size=32,
        load_images=True,
        pose_center=None,
    ):
        super().__init__()
        self.img_dir = os.path.join(data_dir, "obs")
        self.cache = {}
        self.max_length = max_length
        if isinstance(scale, (list, tuple, ListConfig)) and len(scale) == 2:
            self.scale_x = float(scale[0])
            self.scale_y = float(scale[1])
        else:
            self.scale_x = float(scale)
            self.scale_y = float(scale)
        self.cache_size = cache_size
        self.cached_keys_order = []
        self.stage = stage
        self.full_trajectory = full_trajectory
        self.size = size
        self.load_images = bool(load_images)

        info_path = os.path.join(data_dir, "info.hdf5")
        self.h5_file = h5py.File(info_path, "r")
        self.trajectory_indices = self.h5_file["trajectory_indices"][:]
        self.steps = self.h5_file["steps"]
        if pose_center is None:
            self.pose_center_x = self.MAZE_CENTER_RAW / self.scale_x
            self.pose_center_y = self.MAZE_CENTER_RAW / self.scale_y
        elif isinstance(pose_center, (list, tuple, ListConfig)) and len(pose_center) == 2:
            self.pose_center_x = float(pose_center[0])
            self.pose_center_y = float(pose_center[1])
        else:
            self.pose_center_x = float(pose_center)
            self.pose_center_y = float(pose_center)

        self.valid_indices = []
        total_trajectories = len(self.trajectory_indices) - 1
        for idx in range(total_trajectories):
            img_path = os.path.join(self.img_dir, f"{idx}.npy")
            if (not self.load_images) or os.path.exists(img_path):
                self.valid_indices.append(idx)

        print(
            "Found {valid} valid trajectories (with corresponding image files) out of {total} total trajectories".format(
                valid=len(self.valid_indices),
                total=total_trajectories,
            )
        )

    def close(self):
        """Closes the HDF5 file resource. Should be called when done with the dataset."""
        if hasattr(self, "h5_file") and self.h5_file:
            self.h5_file.close()
            self.h5_file = None
            print("HDF5 file closed.")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]

        if self.cache_size > 0 and actual_idx in self.cache:
            self.cached_keys_order.remove(actual_idx)
            self.cached_keys_order.append(actual_idx)
            return self.cache[actual_idx]

        start_idx = self.trajectory_indices[actual_idx]
        end_idx = self.trajectory_indices[actual_idx + 1]
        item_info_raw = self.steps[start_idx:end_idx]
        img_raw = None
        if self.load_images:
            img_path = os.path.join(self.img_dir, f"{actual_idx}.npy")
            try:
                img_raw = np.load(img_path, mmap_mode="r")
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                return None

        current_raw_length = len(item_info_raw)

        if self.full_trajectory:
            info_selected = item_info_raw
            if self.load_images:
                img_selected = img_raw[:]
        elif current_raw_length > self.max_length:
            if self.stage == "train":
                start_idx_crop = np.random.randint(0, current_raw_length - self.max_length + 1)
                end_idx_crop = start_idx_crop + self.max_length
            else:
                start_idx_crop = 0
                end_idx_crop = self.max_length
            info_selected = item_info_raw[start_idx_crop:end_idx_crop]
            if self.load_images:
                img_selected = img_raw[start_idx_crop:end_idx_crop]
        else:
            info_selected = item_info_raw
            if self.load_images:
                img_selected = img_raw[:]

        info_np = np.array(info_selected, dtype=np.float32)
        pose_raw_np = info_np[:, :3]
        action_np = info_np[:, 3:] - 1

        action_np[:-1] = action_np[1:]
        action_np[-1] = self.IGNORE_INDEX

        new_pose_np = np.zeros((pose_raw_np.shape[0], 4), dtype=np.float32)
        if pose_raw_np.shape[0] > 0:
            theta_rad = pose_raw_np[:, 2] * np.pi / 180.0
            new_pose_np[:, 0] = pose_raw_np[:, 0] / self.scale_x - self.pose_center_x
            new_pose_np[:, 1] = pose_raw_np[:, 1] / self.scale_y - self.pose_center_y
            new_pose_np[:, 2] = np.cos(theta_rad)
            new_pose_np[:, 3] = np.sin(theta_rad)
        pose_processed_np = new_pose_np

        if self.load_images:
            img_processed_np = img_selected.astype(np.float32) / 127.5 - 1.0
            if self.size == 32 and img_processed_np.shape[-2:] == (64, 64):
                img_processed_np = img_processed_np[:, :, ::2, ::2]
            elif img_processed_np.shape[-2:] != (self.size, self.size):
                img_tensor = torch.from_numpy(np.ascontiguousarray(img_processed_np))
                img_processed_np = F.interpolate(
                    img_tensor,
                    size=(self.size, self.size),
                    mode="bilinear",
                    align_corners=False,
                ).numpy()

        pose = torch.tensor(pose_processed_np, dtype=torch.float32)
        action = torch.tensor(action_np, dtype=torch.float32)
        result = dict(pose=pose, action=action)
        if self.load_images:
            img = torch.tensor(img_processed_np, dtype=torch.float32)
            result["image"] = img

        if self.cache_size > 0:
            if len(self.cache) >= self.cache_size:
                key_to_evict = self.cached_keys_order.pop(0)
                if key_to_evict in self.cache:
                    del self.cache[key_to_evict]

            self.cache[actual_idx] = result
            self.cached_keys_order.append(actual_idx)

        return result
