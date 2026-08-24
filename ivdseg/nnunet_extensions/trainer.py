"""Project-specific nnU-Net trainer variants used by B2 only."""

from __future__ import annotations

import os
import random

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import numpy as np
import torch


class nnUNetTrainerB2NoMirroring(nnUNetTrainer):
    """Keep nnU-Net's self-configured 3D baseline while prohibiting axis flips."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        seed = int(os.environ.get("IVDSEG_B2_SEED", "17"))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        super().__init__(plans, configuration, fold, dataset_json, device)

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation, dummy_2d, initial_patch_size, _mirror_axes = super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        self.inference_allowed_mirroring_axes = None
        return rotation, dummy_2d, initial_patch_size, None
