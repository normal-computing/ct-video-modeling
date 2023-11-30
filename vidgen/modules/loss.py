import torch.nn.functional as F
import torch
import math


def dice_coef(prediction, target, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|) = 2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    batch_size = prediction.size(0)

    prediction = prediction.reshape(batch_size, -1)
    target = target.reshape(batch_size, -1)

    intersection = torch.sum(target * prediction, -1)
    coef = (2.0 * intersection + smooth) / (
        torch.sum(torch.square(target), -1)
        + torch.sum(torch.square(prediction), -1)
        + smooth
    )
    return torch.mean(coef)


def compute_dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def compute_psnr(pred, target):
    return -10 * torch.log10(F.mse_loss(pred, target))


def compute_j_metric(pred, target):
    intersection = torch.logical_and(pred, target)
    union = torch.logical_or(pred, target)

    return torch.sum(intersection) / torch.sum(union)


def compute_f_metric(pred, target):
    true_positive = torch.sum(torch.logical_and(pred, target))
    false_positive = torch.sum(torch.logical_and(torch.logical_not(pred), target))
    false_negative = torch.sum(torch.logical_and(pred, torch.logical_not(target)))

    return true_positive / (true_positive + 0.5 * (false_positive + false_negative))


if __name__ == "__main__":
    # Example usage:
    # Replace 'predicted' and 'target' with your actual prediction and ground truth tensors
    predicted = torch.randn(10, 1, 16, 16)
    target = torch.randn(10, 1, 16, 16)

    loss = compute_dice_loss(predicted, target)
    print("Dice Loss:", loss.item())
