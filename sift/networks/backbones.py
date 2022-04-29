import torch
import torch.nn as nn
import timm


class ResNet50(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = timm.create_model(
            'resnet50',
            num_classes=0,
            pretrained=True
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.float())


def resnet50() -> nn.Module:
    return ResNet50()
