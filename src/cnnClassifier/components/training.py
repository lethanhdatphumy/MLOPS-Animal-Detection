from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import os
from cnnClassifier.entity.config_entity import TrainingConfig
from cnnClassifier.components.prepare_callbacks import CheckpointCallback


class Training:
    def __init__(self, config: TrainingConfig,
                 tensorboard_callback: SummaryWriter,
                 checkpoint_callback: CheckpointCallback):
        """Set up device, model, loss, optimizer, and data loader from the given config."""
        self.config = config
        self.tensorboard_callback = tensorboard_callback
        self.checkpoint_callback = checkpoint_callback

        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.model = self._load_model().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        # Only optimize parameters that are not frozen
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.0005)
        self.train_loader = self._get_data_loader()

    def _load_model(self):
        """Build VGG16 with a custom head, unfreeze the last 3 classifier layers, and load saved weights if available."""
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        # Infer number of classes from the data directory structure
        try:
            class_dirs = [d for d in os.listdir(self.config.training_data) if (self.config.training_data / d).is_dir()]
            num_classes = len(class_dirs) if len(class_dirs) > 0 else 2
        except Exception:
            num_classes = 2
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
        # Freeze entire classifier, then selectively unfreeze deeper layers
        for p in model.classifier.parameters():
            p.requires_grad = False
        for idx in [3, 5, 6]:
            for p in model.classifier[idx].parameters():
                p.requires_grad = True
        # Load checkpoint if it exists; fall back to unsafe load on failure
        path = self.config.updated_base_model_path
        if Path(path).exists():
            try:
                from torch.serialization import add_safe_globals
                add_safe_globals([models.vgg.VGG])
            except Exception:
                pass
            try:
                state_obj = torch.load(path, weights_only=True, map_location=self.device)
                sd = state_obj.state_dict() if hasattr(state_obj, 'state_dict') else state_obj
                model.load_state_dict(sd, strict=False)
            except Exception:
                try:
                    state_obj = torch.load(path, weights_only=False, map_location=self.device)
                    sd = state_obj.state_dict() if hasattr(state_obj, 'state_dict') else state_obj
                    model.load_state_dict(sd, strict=False)
                except Exception:
                    pass
        return model

    def _get_data_loader(self):
        """Build the training DataLoader with optional augmentation and ImageNet normalization."""
        transform_list = [transforms.Resize((self.config.params_image_size[0], self.config.params_image_size[1])),
                          transforms.ToTensor()]
        if self.config.params_is_augmentation:
            transform_list.insert(0, transforms.RandomHorizontalFlip())
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        transform = transforms.Compose(transform_list)

        dataset = datasets.ImageFolder(root=str(self.config.training_data), transform=transform)
        dataloader = DataLoader(dataset, batch_size=self.config.params_batch_size, shuffle=True)
        return dataloader

    @staticmethod
    def save_model(model, path: Path):
        """Persist the model's state dict to disk."""
        torch.save(model.state_dict(), str(path))

    def train(self):
        """Run the training loop for the configured number of epochs, logging and checkpointing each epoch."""
        for epoch in range(self.config.params_epochs):
            epoch_loss = 0.0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * inputs.size(0)

            epoch_loss /= len(self.train_loader.dataset)
            self.tensorboard_callback.add_scalar('Loss/train', epoch_loss, epoch)
            print(f'Epoch {epoch + 1}/{self.config.params_epochs}, Loss: {epoch_loss:.4f}')

            self.checkpoint_callback.save(self.model, epoch + 1)

        # Persist the final model after all epochs complete
        self.save_model(self.model, self.config.trained_model_path)
