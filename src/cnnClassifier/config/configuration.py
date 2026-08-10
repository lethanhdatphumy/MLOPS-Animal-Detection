from src.cnnClassifier.constants import PARAMS_FILE_PATH, CONFIG_FILE_PATH
from src.cnnClassifier.utils.common import create_directories, read_yaml
from cnnClassifier.entity.config_entity import DataIngestionConfig, PrepareBaseModelConfig, PrepareCallbacksConfig, \
    TrainingConfig, EvaluationConfig
from pathlib import Path


class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        """Load config.yaml and params.yaml, then create the top-level artifacts directory."""
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """Return a DataIngestionConfig built from config.yaml paths."""
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir)
        )
        return data_ingestion_config

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        """Return a PrepareBaseModelConfig combining artifact paths and model hyperparameters."""
        config = self.config.prepare_base_model
        params = self.params

        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=params.IMAGE_SIZE,
            params_include_top=params.INCLUDE_TOP,
            params_classes=params.CLASSES,
            params_weights=params.WEIGHTS,
            params_learning_rate=params.LEARNING_RATE,
        )

        return prepare_base_model_config

    def get_prepare_callbacks_config(self) -> PrepareCallbacksConfig:
        """Return a PrepareCallbacksConfig with TensorBoard log dir and checkpoint filepath template."""
        config = self.config.prepare_callbacks
        create_directories([config.root_dir])
        return PrepareCallbacksConfig(
            root_dir=Path(config.root_dir),
            checkpoint_model_filepath=Path(config.checkpoint_model_filepath),
            tensorboard_log_dir=Path(config.tensorboard_root_log_dir),
        )

    def get_training_config(self) -> TrainingConfig:
        """Return a TrainingConfig with data path, model path, and training hyperparameters."""
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        # Use the data_ingestion root as ImageFolder root (expects class subfolders)
        training_data = Path(self.config.data_ingestion.unzip_dir)
        create_directories([
            Path(training.root_dir)
        ])
        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE
        )

        return training_config

    def get_evaluation_config(self) -> EvaluationConfig:
        """Return an EvaluationConfig pointing at the trained model and dataset root."""
        cfg = self.config
        params = self.params
        return EvaluationConfig(
            root_dir=Path(cfg.training.root_dir),
            trained_model_path=Path(cfg.training.trained_model_path),
            data_root=Path(cfg.data_ingestion.unzip_dir),
            params_image_size=params.IMAGE_SIZE,
        )
