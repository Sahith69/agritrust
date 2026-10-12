import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import config
import torch
import torch.nn as nn
import torch.nn.functional as F

class OrdinalCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes=4):
        super(OrdinalCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        
    def forward(self, logits, target):
        # logits: (batch_size, num_classes)
        # target: (batch_size,) integer
        
        pred = torch.argmax(logits, dim=1)
        penalty = torch.abs(pred - target).float()
        
        # Weight each sample by (1 + |pred - true|)
        weights = 1.0 + penalty
        
        ce_loss = F.cross_entropy(logits, target, reduction='none')
        weighted_loss = ce_loss * weights
        
        return weighted_loss.mean()
