from pathlib import Path

from cnnClassifier.entity.config_entity import PrepareCallbacksConfig
from cnnClassifier.utils.common import create_directories
import torch


class CheckpointCallback:
    """Saves model state dict to disk after each epoch."""

    def __init__(self, filepath: Path):
        """Store the filepath template (supports {epoch} formatting)."""
        self.filepath = filepath

    def save(self, model, epoch: int):
        """Write the model's state dict for the given epoch number."""
        torch.save(model.state_dict(), str(self.filepath).format(epoch=epoch))


class PrepareCallbacks:
    def __init__(self, config: PrepareCallbacksConfig):
        """Create required directories and store callback config."""
        self.config = config
        create_directories([self.config.root_dir,
                            self.config.tensorboard_log_dir.parent,
                            self.config.checkpoint_model_filepath.parent])

    def get_tensorboard_callback(self):
        """Build a TensorBoard SummaryWriter pointed at the configured log directory."""
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(self.config.tensorboard_log_dir))
        return writer

    def get_checkpoint_callback(self):
        """Build a CheckpointCallback that saves weights to the configured filepath template."""
        checkpoint_callback = CheckpointCallback(filepath=self.config.checkpoint_model_filepath)
        return checkpoint_callback
