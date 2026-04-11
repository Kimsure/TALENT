import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator_loss(nn.Module):
    def __init__(self, eps=1e-6):
        super(Discriminator_loss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        """
        pred: Tensor of shape [B, 1, H, W] - logits output
        target: Tensor of shape [B, 1, H, W] - binary mask (0 or 1)
        """
        pred_logits = pred.clone() 
        target_logits = target.clone()
        pred = torch.sigmoid(pred)  
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1).float()
        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1) - intersection
        iou = (intersection + self.eps) / (union + self.eps)
        loss = F.binary_cross_entropy_with_logits(pred_logits, target_logits) + 1 - iou
        
        return loss.mean()