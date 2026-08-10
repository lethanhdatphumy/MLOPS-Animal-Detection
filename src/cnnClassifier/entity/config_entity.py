from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class DataIngestionConfig:
    """
    Data Ingestion Configurations
    """
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    """
    Prepare Base Model Configurations
    """
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int


@dataclass(frozen=True)
class PrepareCallbacksConfig:
    """
    Prepare Callbacks Configurations
    """
    root_dir: Path
    checkpoint_model_filepath: Path
    tensorboard_log_dir: Path


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list

@dataclass(frozen=True)
class EvaluationConfig:
    root_dir: Path
    trained_model_path: Path
    data_root: Path
    params_image_size: List[int]