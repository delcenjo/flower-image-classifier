import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes, freeze_backbone=True, pretrained=True):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
