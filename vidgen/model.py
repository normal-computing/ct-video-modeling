from vidgen.utils import instantiate_from_config
from vidgen.modules.loss import LapLoss, compute_psnr

from typing import Any, Dict, Mapping

from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

import pytorch_lightning as pl
import torchvision
import random

import numpy as np

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

        # self.lap_loss = LapLoss(max_levels=3)

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

    def midpoint_interpolation(self, x, depth=1):
        for _ in range(depth):
            n = x.size(1)
            spline_ts = torch.linspace(0, n - 1, n).to(x)
            solver_ts = torch.linspace(0, n - 1, 2 * n - 1).to(x)
            x = self.cde_model(x, solver_ts, spline_ts)

        return x

    # def inference(self, x, factor, midpoint=True):
    #     grab_idx = list(range(0, x.size(1)))
    #     if midpoint:
    #         for i in range(1, int(np.log2(factor)) + 1):
    #             n = x.size(1)
    #             spline_ts = torch.linspace(0, n - 1, n).to(x)
    #             solver_ts = torch.linspace(0, n - 1, 2 * n - 1).to(x)
    #             x_out = self.cde_model(x, solver_ts, spline_ts)

    #             replace_idx = [i for i in range(0, x_out.size(1), 2**i)]
    #             x_out[:, replace_idx] = x[:, grab_idx]
    #             grab_idx = replace_idx
    #             x = x_out
    #         return x

    #     x_out = self(x, factor)
    #     replace_idx = [i for i in range(0, x_out.size(1), factor)]
    #     x_out[:, replace_idx] = x[:, grab_idx]
    #     return x_out

    def forward(self, x, spline_ts, solver_ts):
        return self.cde_model(x, solver_ts, spline_ts)

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
        # spline_reg_loss = 0.0
        for batch in batch_dict.values():
            assert batch.size(1) % 2 == 1, "Number of input frames must be odd"

            n_frames = batch.size(1)
            # solver_ts = torch.arange(0, batch.size(1) - 1).to(batch)
            solver_ts = torch.arange(0, batch.size(1)).to(batch)
            if n_frames == 3:
                idx_list = [0, 2]
                spline_ts = torch.tensor(idx_list).to(batch)
            else:
                keep_frames_idx = list(range(1, n_frames - 1))
                subset_size = random.randint(0, len(keep_frames_idx) - 1)
                keep_frames_idx = sorted(random.sample(keep_frames_idx, subset_size))
                idx_list = [0] + keep_frames_idx + [n_frames - 1]
                spline_ts = torch.tensor(idx_list).to(batch)

            # matches = torch.isin(solver_ts, spline_ts)
            # generation_idx = torch.nonzero(~matches).squeeze()

            # output = self(batch[:, idx_list], spline_ts, solver_ts)[:, generation_idx]
            # batch = batch[:, generation_idx]

            # if batch.ndim == 4:
            #     batch, output = batch.unsqueeze(1), output.unsqueeze(1)

            output = self(batch[:, idx_list], spline_ts, solver_ts)

            # l1_loss = self.lap_loss(output, batch)
            l2_loss = ((output - batch) ** 2 + 1e-6) ** 0.5
            l1_reconstructions += l2_loss.mean()  # + l1_loss.mean()
            # for layer in self.cde_model.encoder:
            #     if hasattr(layer, "fit_spline"):
            #         spline_derivs = torch.stack(
            #             [layer.static.derivative(t) for t in solver_ts], dim=1
            #         )
            #         spline_reg_loss += spline_derivs.abs().mean()

            if self.vgg16_conv_4_3:
                perceptual_loss += self.perceptual_loss(output, batch)

            self.compute_eval_metrics(output, batch, "train")

        # self.log("n_frames", len(solver_ts) - len(spline_ts), prog_bar=True)
        self.log("train_rec", l1_reconstructions, prog_bar=True)
        # self.log("train_spline_reg", spline_reg_loss, prog_bar=True)
        # self.add_to_epoch_metrics("train_rec", l1_reconstructions)

        if self.vgg16_conv_4_3:
            self.log("train_perceptual", perceptual_loss, prog_bar=True)
            self.add_to_epoch_metrics("train_perceptual", perceptual_loss)

            return l1_reconstructions + 1e-2 * perceptual_loss

        return l1_reconstructions  # + 1e-2 * spline_reg_loss

    def validation_step(self, batch, batch_idx):
        assert batch.size(1) % 2 == 1, "Number of input frames must be odd"

        solver_ts = torch.arange(0, batch.size(1)).to(batch)
        spline_ts = torch.arange(0, batch.size(1), 2).to(batch)

        output = self(batch[:, ::2], spline_ts, solver_ts)[:, 1::2]
        batch = batch[:, 1::2]
        l1_reconstructions = F.l1_loss(output, batch)
        self.log("valid_rec", l1_reconstructions, prog_bar=True)
        self.add_to_epoch_metrics("valid_rec", l1_reconstructions)
        self.compute_eval_metrics(output, batch, "valid")

        perceptual_loss = 0.0
        if self.vgg16_conv_4_3:
            perceptual_loss = self.perceptual_loss(output, batch)
            self.log("valid_perceptual", perceptual_loss, prog_bar=True)
            self.add_to_epoch_metrics("valid_perceptual", perceptual_loss)

            return l1_reconstructions + 1e-2 * perceptual_loss

        return l1_reconstructions

    def get_parameters(self):
        for name, param in self.named_parameters():
            if "vgg16_conv_4_3" not in name:
                yield param

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.get_parameters(), lr=self.lr, amsgrad=False, weight_decay=1e-4
        )
        # lr_scheduler = CosineAnnealingLR(optimizer, T_max=5)
        return {"optimizer": optimizer}  # , "lr_scheduler": lr_scheduler}

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
            solver_ts = torch.arange(0, interp_f + 1).to(batch)
            spline_ts = torch.tensor([0, interp_f]).to(batch)
            curr_interp = self(sample_frames[:, idx : idx + 2], spline_ts, solver_ts)
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
