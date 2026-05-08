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


class TCCL_loss(nn.Module):
    def __init__(self, temperature=1.0):
        super(TCCL_loss, self).__init__()
        self.temperature = temperature
        self.eps = float(int(0.1**-1))

    def forward(self, visual_proto, pos_state, neg_state, valid_mask=None):
        if valid_mask is None:
            valid_mask = torch.ones(visual_proto.shape[0],
                                    device=visual_proto.device,
                                    dtype=visual_proto.dtype)
        else:
            valid_mask = valid_mask.reshape(-1).to(dtype=visual_proto.dtype)

        if valid_mask.sum() <= 0:
            return visual_proto.sum() * 0.0

        visual_proto = F.normalize(visual_proto, p=2, dim=-1)
        pos_state = F.normalize(pos_state, p=2, dim=-1)
        neg_state = F.normalize(neg_state, p=2, dim=-1)

        sim_pos = (visual_proto * pos_state).sum(dim=-1) / self.temperature
        sim_neg = (visual_proto * neg_state).sum(dim=-1) / self.temperature
        logits = torch.stack([sim_pos, sim_neg], dim=1)
        loss_tccl = -F.log_softmax(logits, dim=1)[:, 0]
        return (loss_tccl * valid_mask).sum() / (valid_mask.sum().clamp_min(1.0) * self.eps)