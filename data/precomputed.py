import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


class PrecomputedDataset(Dataset):
    def __init__(self, data_dir, image_dir=None, max_length=None, size=64):
        self.file_paths = glob.glob(os.path.join(data_dir, "*.pt"))
        if not self.file_paths:
            raise FileNotFoundError(f"No .pt files found in {data_dir}")
        self.max_length = max_length
        self.image_dir = image_dir
        self.size = size

        if self.image_dir:
            if os.path.exists(os.path.join(image_dir, "obs")):
                self.img_root = os.path.join(image_dir, "obs")
            else:
                self.img_root = image_dir

    def _load_images_sequence(self, traj_idx, start_idx, end_idx):
        """Load a sequence of images from PNG files."""
        images = []
        traj_dir = os.path.join(self.img_root, str(traj_idx + 8000))

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
        """Vectorized image preprocessing."""
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

    def __len__(self):
        return len(self.file_paths)

    def _resolve_image_window(self, file_path, data, start_idx, end_idx):
        if "traj_idx" in data:
            traj_idx = int(data["traj_idx"])
            if "start_idx" in data:
                start_idx = int(data["start_idx"])
            if "end_idx" in data:
                end_idx = int(data["end_idx"])
            return traj_idx, start_idx, end_idx

        traj_idx = int(os.path.basename(file_path).split(".")[0])
        return traj_idx, start_idx, end_idx

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = torch.load(file_path, map_location="cpu")

        if "latents" in data:
            seq_len = data["latents"].shape[0]
        else:
            seq_len = list(data.values())[0].shape[0]

        start_idx = 0
        end_idx = seq_len

        if self.max_length is not None:
            if seq_len > self.max_length:
                start_idx = torch.randint(0, seq_len - self.max_length + 1, (1,)).item()
                end_idx = start_idx + self.max_length

                for key in data:
                    if (
                        isinstance(data[key], torch.Tensor)
                        and data[key].dim() > 0
                        and data[key].shape[0] == seq_len
                    ):
                        data[key] = data[key][start_idx:end_idx]

        if self.image_dir:
            traj_idx, start_idx, end_idx = self._resolve_image_window(
                file_path, data, start_idx, end_idx
            )
            img_raw = self._load_images_sequence(traj_idx, start_idx, end_idx)
            img_processed = self._process_images_vectorized(img_raw)
            data["image"] = torch.tensor(img_processed, dtype=torch.float32)

        return data
