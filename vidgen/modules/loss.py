from einops import rearrange
import torch.nn.functional as F
import torch


class LapLoss(torch.nn.Module):
    @staticmethod
    def gauss_kernel(channels=3):
        kernel = torch.tensor(
            [
                [1.0, 4.0, 6.0, 4.0, 1],
                [4.0, 16.0, 24.0, 16.0, 4.0],
                [6.0, 24.0, 36.0, 24.0, 6.0],
                [4.0, 16.0, 24.0, 16.0, 4.0],
                [1.0, 4.0, 6.0, 4.0, 1.0],
            ]
        )
        kernel /= 256.0
        kernel = kernel.repeat(channels, 1, 1, 1)
        return kernel

    def __init__(self, max_levels=5, channels=3):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.gauss_kernel = LapLoss.gauss_kernel(channels=channels)

    def laplacian_pyramid(self, img, kernel, max_levels=3):
        def downsample(x):
            return x[:, :, ::2, ::2]

        def upsample(x):
            cc = torch.cat(
                [x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3]).to(x)],
                dim=3,
            )
            cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
            cc = cc.permute(0, 1, 3, 2)
            cc = torch.cat(
                [
                    cc,
                    torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2).to(
                        x
                    ),
                ],
                dim=3,
            )
            cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
            x_up = cc.permute(0, 1, 3, 2)
            return conv_gauss(x_up, 4 * LapLoss.gauss_kernel(channels=x.shape[1]))

        def conv_gauss(img, kernel):
            kernel = kernel.to(img)
            img = F.pad(img, (2, 2, 2, 2), mode="reflect")
            out = F.conv2d(img, kernel, groups=img.shape[1])
            return out

        current = img
        pyr = []
        for _ in range(max_levels):
            filtered = conv_gauss(current, kernel)
            down = downsample(filtered)
            up = upsample(down)
            diff = current - up
            pyr.append(diff)
            current = down
        return pyr

    def forward(self, input, target):
        if len(input.shape) == 4:
            input = input.unsqueeze(1)
            target = target.unsqueeze(1)
        input = rearrange(input, "b f c h w -> (b f) c h w")
        target = rearrange(target, "b f c h w -> (b f) c h w")

        pyr_input = self.laplacian_pyramid(
            img=input, kernel=self.gauss_kernel, max_levels=self.max_levels
        )
        pyr_target = self.laplacian_pyramid(
            img=target, kernel=self.gauss_kernel, max_levels=self.max_levels
        )
        return sum(F.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))


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
