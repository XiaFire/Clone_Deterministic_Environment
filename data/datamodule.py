try:
    import pytorch_lightning as pl
except Exception:
    class _LightningDataModule:
        pass

    class _PLNamespace:
        LightningDataModule = _LightningDataModule

    pl = _PLNamespace()
import torch
from torch.utils.data import DataLoader, Subset

from .collate import collate_fn_maze, collate_fn_precomputed
from .maze import MazeDataset
from .mc import McDataset
from .precomputed import PrecomputedDataset


class TrajectoryDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        print(config)
        self.config = config
        self.dataset_name = config.dataset

    def setup(self, stage=None):
        if self.config.data.get("precomputed", False):
            print("Using precomputed dataset.")
            self.train_dataset = PrecomputedDataset(
                self.config.data.train_dir,
                image_dir=self.config.data.get("train_image_dir", None),
                max_length=self.config.data.get("max_length", 100),
                size=self.config.data.get("size", 64),
            )
            self.val_dataset = PrecomputedDataset(
                self.config.data.val_dir,
                image_dir=self.config.data.get("val_image_dir", None),
                max_length=self.config.data.get("max_length", 100),
                size=self.config.data.get("size", 64),
            )
            return

        if self.dataset_name == "mc":
            print("Loading MC dataset with grouped HDF5 format.")
            train_cache = self.config.data.get("cache_size", 10000)
            val_cache = self.config.data.get(
                "val_cache_size", self.config.data.get("cache_size", 1000)
            )
            if stage == "train":
                self.train_dataset = McDataset(
                    data_dir=self.config.data.train_dir,
                    cache_size=train_cache,
                    scale=self.config.data.get("scale", 1),
                    size=self.config.data.get("size", 64),
                    max_length=self.config.data.get("max_length", 5),
                    full_trajectory=self.config.data.get("full_trajectory", False),
                    load_images=self.config.data.get("load_images", True),
                    pose_center=self.config.data.get("pose_center", 1.0),
                    stage="train",
                )
            else:
                self.train_dataset = McDataset(
                    data_dir=self.config.data.train_dir,
                    cache_size=train_cache,
                    scale=self.config.data.get("scale", 1),
                    size=self.config.data.get("size", 64),
                    max_length=self.config.data.get("max_length", 5),
                    full_trajectory=self.config.data.get("full_trajectory", False),
                    load_images=self.config.data.get("load_images", True),
                    pose_center=self.config.data.get("pose_center", 1.0),
                    stage="train",
                )
                self.val_dataset = McDataset(
                    data_dir=self.config.data.val_dir,
                    cache_size=val_cache,
                    scale=self.config.data.get("scale", 1),
                    size=self.config.data.get("size", 64),
                    max_length=self.config.data.get("max_length", 5),
                    full_trajectory=self.config.data.get("full_trajectory", False),
                    load_images=self.config.data.get("load_images", True),
                    pose_center=self.config.data.get("pose_center", 1.0),
                    stage="val",
                )
        elif self.dataset_name == "maze":
            print("Loading maze dataset.")
            train_cache = self.config.data.get("cache_size", 10000)
            val_cache = self.config.data.get(
                "val_cache_size", self.config.data.get("cache_size", 1000)
            )
            if stage == "train":
                self.train_dataset = MazeDataset(
                    data_dir=self.config.data.train_dir,
                    cache_size=train_cache,
                    scale=self.config.data.get("scale", 1.5),
                    stage="train",
                    max_length=self.config.data.get("max_length", 64),
                    full_trajectory=self.config.data.get("full_trajectory", False),
                    size=self.config.data.get("size", 32),
                    load_images=self.config.data.get("load_images", True),
                    pose_center=self.config.data.get("pose_center", None),
                )
            else:
                self.train_dataset = MazeDataset(
                    data_dir=self.config.data.train_dir,
                    cache_size=train_cache,
                    scale=self.config.data.get("scale", 1.5),
                    stage="train",
                    max_length=self.config.data.get("max_length", 64),
                    full_trajectory=self.config.data.get("full_trajectory", False),
                    size=self.config.data.get("size", 32),
                    load_images=self.config.data.get("load_images", True),
                    pose_center=self.config.data.get("pose_center", None),
                )
                self.val_dataset = MazeDataset(
                    data_dir=self.config.data.val_dir,
                    cache_size=val_cache,
                    scale=self.config.data.get("scale", 1.5),
                    stage="val",
                    max_length=self.config.data.get("max_length", 64),
                    full_trajectory=self.config.data.get("full_trajectory", False),
                    size=self.config.data.get("size", 32),
                    load_images=self.config.data.get("load_images", True),
                    pose_center=self.config.data.get("pose_center", None),
                )

        self._maybe_apply_dataset_subsets(stage)

    def _make_subset(self, dataset, subset_size, seed=0, shuffle=True):
        if dataset is None or subset_size is None:
            return dataset

        subset_size = int(subset_size)
        if subset_size <= 0 or subset_size >= len(dataset):
            return dataset

        if shuffle:
            generator = torch.Generator().manual_seed(int(seed))
            indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
        else:
            indices = list(range(subset_size))
        return Subset(dataset, indices)

    def _maybe_apply_dataset_subsets(self, stage=None):
        seed = self.config.data.get("subset_seed", 0)
        shuffle = bool(self.config.data.get("subset_shuffle", True))

        train_subset_size = self.config.data.get("train_subset_size", None)
        if hasattr(self, "train_dataset"):
            original_len = len(self.train_dataset)
            self.train_dataset = self._make_subset(
                self.train_dataset,
                train_subset_size,
                seed=seed,
                shuffle=shuffle,
            )
            if len(self.train_dataset) != original_len:
                print(f"Using train subset: {len(self.train_dataset)}/{original_len}")

        val_subset_size = self.config.data.get("val_subset_size", None)
        if stage != "train" and hasattr(self, "val_dataset"):
            original_len = len(self.val_dataset)
            self.val_dataset = self._make_subset(
                self.val_dataset,
                val_subset_size,
                seed=int(seed) + 1,
                shuffle=shuffle,
            )
            if len(self.val_dataset) != original_len:
                print(f"Using val subset: {len(self.val_dataset)}/{original_len}")

    def train_dataloader(self):
        return self._create_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._create_dataloader(self.val_dataset)

    def test_dataloader(self):
        return self._create_dataloader(self.val_dataset)

    def _create_dataloader(self, dataset, shuffle=False):
        collate_fn_map = {
            "maze": collate_fn_maze,
            "precomputed": collate_fn_precomputed,
        }
        collate_fn_name = "maze"
        if isinstance(dataset, PrecomputedDataset):
            collate_fn_name = "precomputed"

        num_workers = int(self.config.data.num_workers)
        pin_memory = bool(self.config.data.get("pin_memory", True))
        persistent_workers = bool(
            self.config.data.get("persistent_workers", num_workers > 0)
        )
        drop_last = bool(self.config.data.get("drop_last_train", False)) and shuffle

        dataloader_kwargs = {}
        prefetch_factor = self.config.data.get("prefetch_factor", None)
        if num_workers > 0 and prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = int(prefetch_factor)

        pin_memory_device = self.config.data.get("pin_memory_device", None)
        if pin_memory and pin_memory_device:
            dataloader_kwargs["pin_memory_device"] = pin_memory_device

        return DataLoader(
            dataset,
            batch_size=self.config.data.batch_size,
            collate_fn=collate_fn_map[collate_fn_name],
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers and num_workers > 0,
            drop_last=drop_last,
            **dataloader_kwargs,
        )
