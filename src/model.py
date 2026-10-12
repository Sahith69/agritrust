import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import config
import torch
import torch.nn as nn
from torchvision import models

def unfreeze_last_block(model, backbone_name):
    if backbone_name in ['resnet18', 'resnet34']:
        for param in model.layer4.parameters():
            param.requires_grad = True
    elif backbone_name == 'mobilenet_v2':
        for param in model.features[14:].parameters():
            param.requires_grad = True

class CropSeverityModel(nn.Module):
    def __init__(self, backbone='resnet18', num_classes=4, ordinal=True, pretrained=True):
        super(CropSeverityModel, self).__init__()
        self.backbone_name = backbone
        self.ordinal = ordinal
        
        if backbone == 'resnet18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            num_ftrs = self.backbone.fc.in_features
        elif backbone == 'resnet34':
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet34(weights=weights)
            num_ftrs = self.backbone.fc.in_features
        elif backbone == 'mobilenet_v2':
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.mobilenet_v2(weights=weights)
            num_ftrs = self.backbone.classifier[1].in_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
            
        # Freeze all layers
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        out_features = num_classes - 1 if ordinal else num_classes
        if backbone in ['resnet18', 'resnet34']:
            self.backbone.fc = nn.Linear(num_ftrs, out_features)
        elif backbone == 'mobilenet_v2':
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(num_ftrs, out_features)
            )
            
        unfreeze_last_block(self.backbone, backbone)
        
    def forward(self, x):
        return self.backbone(x)
