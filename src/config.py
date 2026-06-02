from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal, Tuple


@dataclass
class TRDNConfig:
    """Central configuration for TRDN Colab and script runs."""

    train_root: str = "/content/drive/MyDrive/REVIDE_sequences/Train"
    test_root: str = "/content/drive/MyDrive/REVIDE_sequences/Test"
    train_hazy: str = "/content/drive/MyDrive/REVIDE_sequences/Train/hazy"
    test_hazy: str = "/content/drive/MyDrive/REVIDE_sequences/Test/hazy"
    flow_train: str = "/content/drive/MyDrive/video_dehaze_flows/train"
    flow_val: str = "/content/drive/MyDrive/video_dehaze_flows/val"
    dataset_root: str = "/content/drive/MyDrive/REVIDE_sequences"
    project_root: str = "/content/drive/MyDrive/TRDN_REVIDE"

    image_size: int = 256
    crop_size: int = 256
    seq_len: int = 10
    batch_size: int = 1
    num_workers: int = 2
    mixed_precision: str = "fp16"
    seed: int = 1234
    train_mode: Literal["dehaze", "reconstruct"] = "reconstruct"

    train_split: str = "train"
    val_split: str = "val"
    image_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    sd_model_id: str = "runwayml/stable-diffusion-inpainting"
    use_raft_alignment: bool = True
    freeze_raft: bool = True
    use_temporal_transformer: bool = True
    transformer_num_layers: int = 4
    transformer_num_heads: int = 8
    transformer_token_dim: int = 256
    transformer_pool_size: int = 8
    train_unet: bool = True
    train_temporal_modules: bool = True
    enable_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    enable_unet_gradient_checkpointing: bool = True
    enable_xformers_if_available: bool = True
    enable_torch_compile: bool = False

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
            "debug_outputs": root / "debug_outputs",
        }

    def ensure_dirs(self) -> dict:
        paths = self.paths()
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        return paths

    def to_dict(self) -> dict:
        return asdict(self)

    def root_for_split(self, split: str) -> str:
        normalized = split.lower()
        if normalized in {"train", "training"}:
            return self.train_root
        if normalized in {"val", "valid", "validation", "test", "testing"}:
            return self.test_root
        return self.dataset_root

    def flow_root_for_split(self, split: str) -> str:
        normalized = split.lower()
        return self.flow_train if normalized in {"train", "training"} else self.flow_val
