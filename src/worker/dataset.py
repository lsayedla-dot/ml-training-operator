"""NuScenes classification dataset for AV object detection training."""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset

# Class labels for AV object classification
CLASSES = ["car", "pedestrian", "cyclist", "other"]
NUM_CLASSES = len(CLASSES)
IMAGE_SIZE = 224


class NuScenesClassificationDataset(Dataset):
    """Dataset for classifying AV objects from nuScenes camera images.

    Supports two modes:
    - Real: loads actual nuScenes mini split data using nuscenes-devkit
    - Synthetic: generates random tensors with correct shapes (for CI/testing)
    """

    def __init__(
        self,
        data_dir: str = "data/nuscenes-mini",
        split: str = "train",
        synthetic: bool = False,
        num_synthetic_samples: int = 200,
    ) -> None:
        self.data_dir = data_dir
        self.split = split
        self.synthetic = synthetic
        self.samples: list[tuple[torch.Tensor, int]] = []

        if synthetic:
            self._generate_synthetic(num_synthetic_samples)
        else:
            self._load_nuscenes()

    def _generate_synthetic(self, num_samples: int) -> None:
        """Generate synthetic data with correct shapes for testing."""
        rng = np.random.RandomState(42 if self.split == "train" else 99)
        for _ in range(num_samples):
            image = torch.randn(3, IMAGE_SIZE, IMAGE_SIZE)
            label = rng.randint(0, NUM_CLASSES)
            self.samples.append((image, label))

    def _load_nuscenes(self) -> None:
        """Load real nuScenes data using nuscenes-devkit."""
        try:
            from nuscenes.nuscenes import NuScenes
            from PIL import Image
            from torchvision import transforms
        except ImportError:
            raise ImportError(
                "nuscenes-devkit, Pillow, and torchvision are required. "
                "Install with: pip install -e '.[worker]'"
            )

        transform = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        nusc = NuScenes(version="v1.0-mini", dataroot=self.data_dir, verbose=False)

        # Split scenes: 0-7 train, 8-9 val
        all_scenes = sorted(nusc.scene, key=lambda s: s["name"])
        if self.split == "train":
            scenes = all_scenes[:8]
        else:
            scenes = all_scenes[8:]

        category_map = {
            "vehicle.car": 0,
            "vehicle.truck": 0,
            "vehicle.bus": 0,
            "human.pedestrian": 1,
            "vehicle.bicycle": 2,
            "vehicle.motorcycle": 2,
        }

        for scene in scenes:
            sample_token = scene["first_sample_token"]
            while sample_token:
                sample = nusc.get("sample", sample_token)
                cam_token = sample["data"]["CAM_FRONT"]
                cam_data = nusc.get("sample_data", cam_token)
                img_path = os.path.join(nusc.dataroot, cam_data["filename"])

                if os.path.exists(img_path):
                    img = Image.open(img_path).convert("RGB")

                    # Get annotations for this sample
                    for ann_token in sample["anns"]:
                        ann = nusc.get("sample_annotation", ann_token)
                        cat = ann["category_name"]
                        label = 3  # default: other
                        for prefix, lbl in category_map.items():
                            if cat.startswith(prefix):
                                label = lbl
                                break

                        # Use full image with label (simplified — production would crop)
                        tensor = transform(img)
                        self.samples.append((tensor, label))

                sample_token = sample["next"] if sample["next"] else None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.samples[idx]
