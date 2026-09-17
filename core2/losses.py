import torch
import torch.nn.functional as F
from torchmetrics.classification import JaccardIndex

def tversky(y_true, y_pred, alpha=0.6, beta=0.4, epsilon=1e-5):
    if y_true.size(1) == 2:
        weights = y_true[:, 1:2, :, :]
        y_true = y_true[:, 0:1, :, :]
    else:
        weights = torch.ones_like(y_true)

    y_true = y_true.float()
    y_pred = y_pred.float()

    p0 = y_pred
    p1 = 1.0 - y_pred
    g0 = y_true
    g1 = 1.0 - y_true

    tp = torch.sum(weights * p0 * g0)
    fp = alpha * torch.sum(weights * p0 * g1)
    fn = beta * torch.sum(weights * p1 * g0)
    score = tp / (tp + fp + fn + epsilon)
    return (1.0 - score).mean()

def accuracy(y_true, y_pred):
    if y_true.size(1) == 2:
        y_true = y_true[:, 0:1, :, :].float()
    y_pred = torch.round(y_pred.float())
    return torch.mean((y_true == y_pred).float())

def dice_coef(y_true, y_pred, smooth=1e-7):
    if y_true.size(1) == 2:
        y_true = y_true[:, 0:1, :, :]

    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)

    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred)

    dice = (2. * intersection + smooth) / (union + smooth)

    return dice

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def true_positives(y_true, y_pred):
    if y_true.size(1) == 2:
        y_true = y_true[:, 0:1, :, :]
    return torch.round(y_true * y_pred)

def false_positives(y_true, y_pred):
    if y_true.size(1) == 2:
        y_true = y_true[:, 0:1, :, :].float()
    return torch.round((1 - y_true) * y_pred)

def true_negatives(y_true, y_pred):
    if y_true.size(1) == 2:
        y_true = y_true[:, 0:1, :, :].float()
    return torch.round((1 - y_true) * (1 - y_pred))

def false_negatives(y_true, y_pred):
    if y_true.size(1) == 2:
        y_true = y_true[:, 0:1, :, :].float()
    return torch.round(y_true * (1 - y_pred))

def sensitivity(y_true, y_pred):
    if y_true.size(1) == 2:
        y_true = y_true[:, 0:1, :, :].float()
    tp = true_positives(y_true, y_pred)
    fn = false_negatives(y_true, y_pred)
    return tp.sum() / (tp.sum() + fn.sum() + 1e-7)

def specificity(y_true, y_pred):
    if y_true.size(1) == 2:
        y_true = y_true[:, 0:1, :, :].float()
    tn = true_negatives(y_true, y_pred)
    fp = false_positives(y_true, y_pred)
    return tn.sum() / (tn.sum() + fp.sum() + 1e-7)

def miou(y_true, y_pred, device):
    if y_true.size(1) == 2:
        y_true = y_true[:, 0:1, :, :].float()

    y_true = (y_true > 0.5).float()
    y_pred = (y_pred > 0.5).float()

    iou_metric = JaccardIndex(task="binary", num_classes=2).to(device)

    iou_score = iou_metric(y_pred, y_true).item()

    return iou_score

def weight_miou(y_true, y_pred):
    if y_true.size(1) == 2:
        weights = y_true[:, 1:2, :, :]
        y_true = y_true[:, 0:1, :, :]
    else:
        weights = torch.ones_like(y_true)

    y_pred = y_pred.float()

    intersection = (y_true * y_pred * weights).sum()
    union = ((y_true + y_pred) * weights).sum() - intersection
    iou = (intersection + 1e-7) / (union + 1e-7)

    return iou.item()
