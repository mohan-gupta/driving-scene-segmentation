import torch
import torch.nn as nn
import torch.nn.functional as F
import config

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss

def save_model(epoch, model, optimizer, loss):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, config.MODEL_PATH)

def load_model():
    checkpoint = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
    model = checkpoint['model_state_dict']

    optimizer = checkpoint['optimizer_state_dict']
    
    epoch = checkpoint['epoch']

    return model, optimizer, epoch

def smp_dice_score(y_pred, y_true):
    dice_loss = DiceLoss(mode='multiclass')
    loss = dice_loss(y_pred, y_true)
    score = 1 - loss

    return score

def metrics(output, target):
    tp, fp, fn, tn = smp.metrics.get_stats(output, target, mode='multiclass', threshold=0.5)
    iou_score = smp.metrics.functional.iou_score(tp, fp, fn, tn, reduction='macro')
    accuracy = smp.metrics.functional.accuracy(tp, fp, fn, tn, reduction='macro')

    return {'IOU':iou_score, "Accuracy":accuracy}

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        self.nll = nn.NLLLoss(weight=weight)

    def forward(self, y_pred, y_true):
        if y_pred.ndim > 2:
            channels = y_pred.shape[1]
            y_pred = y_pred.permute(0, *range(2, y_pred.ndim), 1).reshape(-1, channels)
            
            y_true = y_true.view(-1)

        log_probs = nn.functional.log_softmax(y_pred, dim=-1)
        probs = log_probs.exp()

        cross_entropy = self.nll(log_probs, y_true)
        focal_term = (1-probs)**self.gamma

        loss = focal_term*cross_entropy

        if self.reduction=='mean':
            loss = loss.mean()
        elif self.reduction=='sum':
            loss = loss.sum()

        return loss