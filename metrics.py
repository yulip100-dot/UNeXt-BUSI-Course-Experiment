import numpy as np
import torch
import torch.nn.functional as F


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()

    # 与预测掩膜保存时使用相同阈值
    prediction = output >= 0.5
    truth = target > 0.5

    # 每张图像分别计算，再对批次取平均
    axes = tuple(range(1, prediction.ndim))
    intersection = np.logical_and(prediction, truth).sum(axis=axes)
    union = np.logical_or(prediction, truth).sum(axis=axes)
    pred_area = prediction.sum(axis=axes)
    true_area = truth.sum(axis=axes)

    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * intersection + smooth) / (
        pred_area + true_area + smooth
    )

    return float(iou.mean()), float(dice.mean())


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)
