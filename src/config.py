from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple


@dataclass
class TRDNConfig:
    """Central configuration for TRDN Colab and script runs."""

    dataset_root: str = "/content/drive/MyDrive/REVIDE"
    project_root: str = "/content/drive/MyDrive/TRDN_REVIDE"

    image_size: int = 256
    crop_size: int = 256
    seq_len: int = 10
    batch_size: int = 1
    num_workers: int = 2
    mixed_precision: str = "fp16"
    seed: int = 1234

    train_split: str = "train"
    val_split: str = "val"
    image_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    sd_model_id: str = "runwayml/stable-diffusion-inpainting"
    use_raft_alignment: bool = True
    freeze_raft: bool = True
    train_unet: bool = True
    train_temporal_modules: bool = True
    enable_unet_gradient_checkpointing: bool = True
    enable_xformers_if_available: bool = True

    learning_rate: float = 1e-5
    temporal_learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    max_train_steps: int = 1000
    num_epochs: int = 1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    log_every: int = 10
    validate_every: int = 100
    checkpoint_every: int = 250
    num_inference_steps: int = 30

    w_diffusion: float = 1.0
    w_l1: float = 0.25
    w_lpips: float = 0.05
    w_temporal: float = 0.05
    w_flow: float = 0.05
    w_reference: float = 0.05

    resume_from_checkpoint: str = ""

    def paths(self) -> dict:
        root = Path(self.project_root)
        return {
            "project": root,
            "checkpoints": root / "checkpoints",
            "logs": root / "logs",
            "outputs": root / "outputs",
            "visualizations": root / "visualizations",
        }

    def ensure_dirs(self) -> dict:
        paths = self.paths()
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        return paths

    def to_dict(self) -> dict:
        return asdict(self)
