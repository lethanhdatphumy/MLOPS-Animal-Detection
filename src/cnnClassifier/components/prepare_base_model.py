import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path

from src.cnnClassifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        """Store model preparation config (paths, hyperparameters)."""
        self.config = config

    def get_base_model(self):
        """Download/load pretrained VGG16 weights and save the state dict to disk."""
        weights = models.VGG16_Weights.IMAGENET1K_V1 if self.config.params_weights == "imagenet" else None
        base_model = models.vgg16(weights=weights)

        self.save_model(model=base_model, model_path=self.config.base_model_path)
        return base_model

    @staticmethod
    def _prepare_full_model(base_model, config: PrepareBaseModelConfig):
        """Freeze the feature extractor and replace the final classifier layer with a custom head."""
        # Freeze all parameters in the feature extractor
        for param in base_model.features.parameters():
            param.requires_grad = False

        # Only the final layer is replaced so ImageNet-pretrained classifier weights are kept
        num_features = base_model.classifier[6].in_features
        base_model.classifier[6] = nn.Linear(num_features, config.params_classes)

        return base_model

    def update_base_model(self):
        """Load the saved base model, attach the custom classifier head, and save the result."""
        # Allow-list VGG for safe deserialization when using weights_only=True
        torch.serialization.add_safe_globals([models.vgg.VGG])

        weights = models.VGG16_Weights.IMAGENET1K_V1 if self.config.params_weights == "imagenet" else None
        base_model = models.vgg16(weights=weights)
        state_dict = torch.load(self.config.base_model_path, weights_only=True)
        base_model.load_state_dict(state_dict)

        full_model = self._prepare_full_model(base_model=base_model, config=self.config)
        self.save_model(model=full_model, model_path=self.config.updated_base_model_path)
        return full_model

    @staticmethod
    def save_model(model, model_path: Path):
        """Persist the model's state dict to the given path."""
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)