import timm
import torch
import torch.nn as nn


class BirdCLEFModel(nn.Module):
    def __init__(self, num_classes: int, model_name: str = "efficientnet_b0", pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=1,  # mel-spectrogram is single channel
            num_classes=0,  # remove classifier head
        )
        in_features = self.backbone.num_features
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)
