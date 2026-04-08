from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


CLASS_NAMES = [
    "Bacterial Pneumonia",
    "Corona Virus Disease",
    "Normal",
    "Tuberculosis",
    "Viral Pneumonia",
]

IMG_SIZE = 224


def build_lung_model(num_classes: int = len(CLASS_NAMES)) -> nn.Module:
    """Recreate the EfficientNet-B0 architecture used in lung_short.ipynb."""
    # Use weights=None for deployment loading so we don't need network/cache access.
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Linear(in_features, num_classes))
    return model


def get_lung_inference_transforms() -> transforms.Compose:
    """Validation/test transforms used in the notebook."""
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )


def load_lung_weights(weights_path: str, device: str | torch.device = "cpu") -> nn.Module:
    """Load trained weights into the notebook's EfficientNet-B0 model."""
    model = build_lung_model()
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
