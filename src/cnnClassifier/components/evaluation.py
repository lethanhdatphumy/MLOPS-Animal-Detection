from pathlib import Path
import os
import json
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from cnnClassifier.entity.config_entity import EvaluationConfig


class Evaluator:
    def __init__(self, config: EvaluationConfig):
        """Set up device, load the trained model, and build the eval transform."""
        self.config = config
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = self._load_model().to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((self.config.params_image_size[0], self.config.params_image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.criterion = nn.CrossEntropyLoss()

    def _load_model(self):
        """Build VGG16 with a matching custom head and load the trained weights from disk."""
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        in_features = model.classifier[6].in_features
        try:
            class_dirs = [d for d in os.listdir(self.config.data_root) if (self.config.data_root / d).is_dir()]
            num_classes = len(class_dirs) if len(class_dirs) > 0 else 2
        except Exception:
            num_classes = 2
        model.classifier[6] = nn.Linear(in_features, num_classes)
        path = self.config.trained_model_path
        if path.exists():
            try:
                sd = torch.load(path, weights_only=True, map_location=self.device)
                if hasattr(sd, 'state_dict'):
                    sd = sd.state_dict()
                model.load_state_dict(sd, strict=False)
            except Exception:
                try:
                    sd = torch.load(path, weights_only=False, map_location=self.device)
                    if hasattr(sd, 'state_dict'):
                        sd = sd.state_dict()
                    model.load_state_dict(sd, strict=False)
                except Exception:
                    pass
        model.eval()
        return model

    def build_dataset(self):
        """Create an ImageFolder dataset and DataLoader from the configured data root."""
        dataset = datasets.ImageFolder(root=str(self.config.data_root), transform=self.transform)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        return dataset, loader

    def evaluate(self):
        """Run inference over the full dataset and return average loss and accuracy."""
        _, loader = self.build_dataset()
        total_loss = 0.0
        total_items = 0
        total_correct = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += float(loss.item()) * inputs.size(0)
                total_items += inputs.size(0)
                preds = outputs.argmax(dim=1)
                total_correct += int((preds == labels).sum().item())
        avg_loss = total_loss / max(total_items, 1)
        acc = total_correct / max(total_items, 1)
        return avg_loss, acc

    def write_scores(self, loss: float, accuracy: float, path: Path):
        """Write loss and accuracy as JSON to the repo root scores.json file."""
        root_scores = Path("scores.json")
        with open(root_scores, "w") as f:
            json.dump({"loss": float(loss), "accuracy": float(accuracy)}, f)
