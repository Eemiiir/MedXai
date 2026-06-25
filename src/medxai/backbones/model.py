"""ResNet-50 multi-label backbone with a layer3 feature-export path.

The manual forward lets us grab the layer3 feature map (the GNN arm pools region
features over it). At 320x320 input, layer3 is 20x20x1024 — a finer spatial grid
than layer4's 10x10, which is why we pool nodes from layer3 (better superpixel
alignment), exactly as recorded in conf/frozen.yaml.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class ResNet50MultiLabel(nn.Module):
    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        self.feature_channels = 1024  # layer3 output channels
        self.backbone.fc = nn.Linear(2048, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        b = self.backbone
        x = b.conv1(x)
        x = b.bn1(x)
        x = b.relu(x)
        x = b.maxpool(x)
        x = b.layer1(x)
        x = b.layer2(x)
        f3 = b.layer3(x)            # (B, 1024, H/16, W/16)  -> 20x20 at 320
        x = b.layer4(f3)
        x = b.avgpool(x)
        x = torch.flatten(x, 1)
        logits = b.fc(x)
        if return_features:
            return logits, f3
        return logits
