import sys; sys.path.insert(0, '/home/sahith/Projects/Hackathon'); import config
import torch
import torch.nn as nn
import torch.nn.functional as F

class CornLoss(nn.Module):
    def __init__(self, num_classes=4):
        super(CornLoss, self).__init__()
        self.num_classes = num_classes
        self.num_tasks = num_classes - 1

    def forward(self, logits, target):
        # logits: (batch_size, num_tasks)
        # target: (batch_size,) integer grades 0 to K-1
        loss = 0.0
        for task_idx in range(self.num_tasks):
            # For task k (k=0,1,2 for num_classes=4)
            # Only samples with target >= k contribute
            mask = (target >= task_idx)
            if not mask.any():
                continue
            
            task_logits = logits[mask, task_idx]
            task_targets = (target[mask] > task_idx).float()
            
            loss += F.binary_cross_entropy_with_logits(task_logits, task_targets, reduction='mean')
            
        return loss / self.num_tasks

def corn_label_from_grade(grade, num_classes=4):
    # Convert single grade to K-1 binary targets
    tasks = []
    for k in range(num_classes - 1):
        if grade > k:
            tasks.append(1)
        else:
            tasks.append(0)
    return tasks

def corn_probs_to_grade(probs):
    # probs: (batch_size, num_tasks) containing sigmoid outputs
    # Outputs predicted grade + per-grade probabilities
    is_1d = probs.dim() == 1
    if is_1d:
        probs = probs.unsqueeze(0)
        
    batch_size = probs.size(0)
    num_tasks = probs.size(1)
    num_classes = num_tasks + 1
    
    p_gt_k = torch.zeros(batch_size, num_tasks, device=probs.device)
    p_gt_k[:, 0] = probs[:, 0]
    for k in range(1, num_tasks):
        p_gt_k[:, k] = p_gt_k[:, k-1] * probs[:, k]
        
    p_k = torch.zeros(batch_size, num_classes, device=probs.device)
    
    # P(Y=0) = 1 - P(Y>0)
    p_k[:, 0] = 1.0 - p_gt_k[:, 0]
    for k in range(1, num_tasks):
        # P(Y=k) = P(Y>k-1) - P(Y>k)
        p_k[:, k] = p_gt_k[:, k-1] - p_gt_k[:, k]
    # P(Y=K-1) = P(Y>K-2)
    p_k[:, num_classes-1] = p_gt_k[:, num_tasks-1]
    
    pred_grade = torch.argmax(p_k, dim=1)
    
    if is_1d:
        return pred_grade[0], p_k[0]
    return pred_grade, p_k
