from vidgen.utils import instantiate_from_config
from vidgen.modules.loss import compute_psnr

from typing import Any, Dict, Mapping

from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

import pytorch_lightning as pl
import torchvision
import random

from einops import rearrange
import torch.nn.functional as F
import torch.nn as nn
import torch

from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim as optim

from torchcde import natural_cubic_coeffs, CubicSpline


class VideoModel(pl.LightningModule):
    def __init__(
        self,
        cde_function,
        lr=1e-3,
        use_vgg_loss=False,
        ckpt_path=None,
    ):
        super().__init__()

        self.cde_model = instantiate_from_config(cde_function)

        self.vgg16_conv_4_3 = None
        if use_vgg_loss:
            vgg16 = torchvision.models.vgg16(pretrained=True)
            vgg16_conv_4_3 = nn.Sequential(*list(vgg16.children())[0][:22])
            for param in vgg16_conv_4_3.parameters():
                param.requires_grad = False
            self.vgg16_conv_4_3 = vgg16_conv_4_3

        self.lr = lr
        self.epoch_loss_metrics = {}
        self.ssim_metric = SSIM(data_range=1.0)

        if ckpt_path is not None:
            print(f"Instantiating weights from: {ckpt_path}")
            self.load_state_dict(torch.load(ckpt_path))

    def perceptual_loss(self, input, target):
        assert input.shape == target.shape, "Shapes mismatch..."
        if len(input.shape) == 4:
            input = input.unsqueeze(1)
            target = target.unsqueeze(1)
        input = rearrange(input, "b f c h w -> (b f) c h w")
        target = rearrange(target, "b f c h w -> (b f) c h w")
        return F.mse_loss(self.vgg16_conv_4_3(input), self.vgg16_conv_4_3(target))

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        return super().load_state_dict(state_dict, False)

    def solver(self, z, ts, spline_ts):
        return self.cde_model(z, ts, spline_ts)

    def midpoint_interpolation(self, x, depth=1):
        for _ in range(depth):
            n = x.size(1)
            spline_ts = torch.linspace(0, n - 1, n).to(x)
            solver_ts = torch.linspace(0, n - 1, 2 * n - 1).to(x)
            x = self.cde_model(x, solver_ts, spline_ts)

        return x

    def forward(self, x, factor=2):
        n = x.size(1) - 1
        spline_ts = torch.arange(0, n + 1).to(x)
        solver_ts = torch.linspace(0, n, factor * n + 1).to(x)
        return self.cde_model(x, solver_ts, spline_ts)

    def subsample_frames(self, batch):
        rn = random.uniform(0, 1)
        if batch.size(1) >= 9:
            if rn < 0.33:
                return batch[:, :9:4]
            if 0.33 < rn < 0.67:
                return batch[:, :9:2]
            return batch

        if batch.size(1) >= 5:
            if rn < 0.5:
                return batch[:, :5:2]
            return batch

        return batch

    def get_subsample_factor(self, n_frames):
        if n_frames == 7:
            return 2

        subsample_choices = []
        for i in range(1, 5):
            if 2**i + 1 > n_frames:
                break
            subsample_choices.append(2**i)
        return random.choice(subsample_choices)

    def add_to_epoch_metrics(self, key, value):
        if key not in self.epoch_loss_metrics:
            self.epoch_loss_metrics[key] = {"n": 0, "loss": 0.0}
        self.epoch_loss_metrics[key]["loss"] += value.item()
        self.epoch_loss_metrics[key]["n"] += 1
        self.log(
            f"{key}_epoch",
            self.epoch_loss_metrics[key]["loss"] / self.epoch_loss_metrics[key]["n"],
            prog_bar=True,
        )

    @torch.no_grad()
    def compute_eval_metrics(self, output, target, phase):
        output_for_metrics = (output.clamp(-1.0, 1.0) + 1.0) / 2.0
        batch_for_metrics = (target.clamp(-1.0, 1.0) + 1.0) / 2.0
        psnr_val = compute_psnr(output_for_metrics, batch_for_metrics)
        ssim_val = self.ssim_metric(
            rearrange(output_for_metrics, "b t c h w -> (b t) c h w"),
            rearrange(batch_for_metrics, "b t c h w -> (b t) c h w"),
        )
        self.add_to_epoch_metrics(f"{phase}_psnr", psnr_val)
        self.add_to_epoch_metrics(f"{phase}_ssim", ssim_val)

    def on_train_epoch_start(self) -> None:
        self.epoch_loss_metrics = {}
        return super().on_train_epoch_start()

    def on_validation_epoch_start(self) -> None:
        self.epoch_loss_metrics = {}
        return super().on_validation_epoch_start()

    def training_step(self, batch_dict):
        # batch, *_ = batch

        perceptual_loss = 0.0
        l1_reconstructions = 0.0
        for batch in batch_dict.values():
            assert batch.size(1) % 2 == 1, "Number of input frames must be odd"

            batch = self.subsample_frames(batch)
            # output = self.midpoint_interpolation(batch[:, ::2])
            interp_f = self.get_subsample_factor(batch.size(1))
            output = self(batch[:, ::interp_f], factor=interp_f)

            l1_reconstructions += F.l1_loss(output, batch)
            if self.vgg16_conv_4_3:
                perceptual_loss += self.perceptual_loss(output, batch)

        l1_reconstructions /= len(batch_dict)
        perceptual_loss /= len(batch_dict)

        self.log("train_rec", l1_reconstructions, prog_bar=True)
        self.add_to_epoch_metrics("train_rec", l1_reconstructions)
        self.compute_eval_metrics(output, batch, "train")

        if self.vgg16_conv_4_3:
            self.log("train_perceptual", perceptual_loss, prog_bar=True)
            self.add_to_epoch_metrics("train_perceptual", perceptual_loss)

            return 0.7 * l1_reconstructions + 0.3 * perceptual_loss

        return l1_reconstructions

    def validation_step(self, batch, batch_idx):
        assert batch.size(1) % 2 == 1, "Number of input frames must be odd"

        output = self.midpoint_interpolation(batch[:, ::2])
        l1_reconstructions = F.l1_loss(output, batch)
        self.log("valid_rec", l1_reconstructions, prog_bar=True)
        self.add_to_epoch_metrics("valid_rec", l1_reconstructions)
        self.compute_eval_metrics(output, batch, "valid")

        perceptual_loss = 0.0
        if self.vgg16_conv_4_3:
            perceptual_loss = self.perceptual_loss(output, batch)
            self.log("valid_perceptual", perceptual_loss, prog_bar=True)
            self.add_to_epoch_metrics("valid_perceptual", perceptual_loss)

            return 0.7 * l1_reconstructions + 0.3 * perceptual_loss

        return l1_reconstructions

    def get_parameters(self):
        for name, param in self.named_parameters():
            if "vgg16_conv_4_3" not in name:
                yield param

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.get_parameters(), lr=self.lr, amsgrad=False, weight_decay=1e-6
        )
        # lr_scheduler = CosineAnnealingLR(optimizer, T_max=5)
        return {"optimizer": optimizer}  # , s"lr_scheduler": lr_scheduler}

    def naive_interpolation(self, frames, factor=2):
        n_frames = frames.size(1) - 1
        solver_ts = torch.linspace(0, n_frames, n_frames * factor + 1).to(frames)
        spline_ts = torch.arange(0, frames.size(1)).to(frames)
        spline = CubicSpline(
            natural_cubic_coeffs(frames.flatten(start_dim=2), t=spline_ts),
            t=spline_ts,
        )
        return torch.stack([spline.evaluate(t) for t in solver_ts], dim=1).unflatten(
            2, frames.shape[-3:]
        )

    @torch.no_grad()
    def log_video_sample(self, batch, batch_idx, **kwargs):
        assert batch.size(1) % 2 == 1, "Number of input frames must be odd"

        sample_frames = batch
        depth = random.choice([1, 2, 3, 4])
        interp_f = 2**depth

        output_vid = torch.tensor([], dtype=batch.dtype)
        for idx in range(sample_frames.size(1) - 1):
            # curr_interp = self.midpoint_interpolation(
            #     sample_frames[:, idx : idx + 2], depth=depth
            # )
            curr_interp = self(sample_frames[:, idx : idx + 2], factor=interp_f)
            output_vid = torch.cat([output_vid, curr_interp[:, :-1].cpu()], dim=1)
        output_vid = torch.cat([output_vid, sample_frames[:, -1:].cpu()], dim=1)

        sample_frames = sample_frames.cpu()
        stretched_vid = sample_frames.repeat_interleave(interp_f, dim=1)
        stretched_vid = stretched_vid[:, : -interp_f + 1]

        return {
            f"video_{interp_f}": torch.cat([stretched_vid, output_vid], dim=-2),
        }

    def on_train_epoch_end(self) -> None:
        return super().on_train_epoch_end()
