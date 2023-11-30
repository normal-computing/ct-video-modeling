from vidgen.modules.layers import memory_efficient_operation
from vidgen.utils import instantiate_from_config
from vidgen.modules.loss import compute_dice_loss, compute_j_metric, compute_f_metric

from typing import Any, Dict, Mapping

from einops import rearrange
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim as optim
import torch.nn.functional as F
import torch

import pytorch_lightning as pl


class VideoModel(pl.LightningModule):
    def __init__(
        self,
        cde_function,
        lr=1e-3,
    ):
        super().__init__()

        self.cde_solver = instantiate_from_config(cde_function)
        self.lr = lr
        self.epoch_loss_metrics = {}

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        return super().load_state_dict(state_dict, False)

    def forward(self, x, m0, ts, logits=True):
        assert len(m0.shape) == 5, "Mask must include a time dimension of size 1"
        assert m0.size(1) == 1, "Time dimension of mask can only be of size 1"

        y = self.cde_solver(x, m0[:, 0], ts)
        if logits:
            return torch.sigmoid(y)
        return y

    @torch.no_grad()
    def compute_eval_metrics(self, output, target, phase):
        output_for_metrics = output * 255.0
        batch_for_metrics = target * 255.0
        j_val = compute_j_metric(output_for_metrics, batch_for_metrics)
        f_val = compute_f_metric(output_for_metrics, batch_for_metrics)
        self.add_to_epoch_metrics(f"{phase}_j", j_val)
        self.add_to_epoch_metrics(f"{phase}_f", f_val)

    def add_to_epoch_metrics(self, key, value):
        if key not in self.epoch_loss_metrics:
            self.epoch_loss_metrics[key] = {"n": 0, "loss": 0.0}
        key_dict = self.epoch_loss_metrics[key]
        key_dict["loss"] += value.item()
        key_dict["n"] += 1
        self.log(f"{key}_epoch", key_dict["loss"] / key_dict["n"], prog_bar=True)

    def on_train_epoch_start(self) -> None:
        self.epoch_loss_metrics = {}
        return super().on_train_epoch_start()

    def on_validation_epoch_start(self) -> None:
        self.epoch_loss_metrics = {}
        return super().on_validation_epoch_start()

    def training_step(self, batch):
        batch, *_ = batch

        x, annotations = batch
        annotations = annotations[:, :, :1]

        ts = torch.arange(x.size(1)).to(x)
        masks = self(x, annotations[:, :1], ts, logits=False)

        bce_loss = F.binary_cross_entropy_with_logits(masks[:, 1:], annotations[:, 1:])
        self.log("train_bce", bce_loss, prog_bar=True)
        self.add_to_epoch_metrics("train_bce", bce_loss)

        masks_0_1 = torch.sigmoid(masks[:, 1:])
        l1_loss = F.l1_loss(masks_0_1, annotations[:, 1:])
        self.log("train_l1", l1_loss, prog_bar=True)
        self.add_to_epoch_metrics("train_l1", l1_loss)

        # boundary_loss = torch.exp(-5 * F.mse_loss(masks_0_1, 1 - masks_0_1))
        # self.log("train_bound", boundary_loss, prog_bar=True)

        self.compute_eval_metrics(masks_0_1, annotations[:, 1:], "train")

        return 0.8 * bce_loss + 0.2 * l1_loss  # + 0.1 * boundary_loss

    def validation_step(self, batch, batch_idx):
        batch, *_ = batch

        x, annotations = batch
        annotations = annotations[:, :, :1]

        ts = torch.arange(x.size(1)).to(x)
        masks = self(x, annotations[:, :1], ts, logits=False)

        bce_loss = F.binary_cross_entropy_with_logits(masks[:, 1:], annotations[:, 1:])
        self.log("valid_bce", bce_loss, prog_bar=True)
        self.add_to_epoch_metrics("valid_bce", bce_loss)

        masks_0_1 = torch.sigmoid(masks[:, 1:])
        l1_loss = F.l1_loss(masks_0_1, annotations[:, 1:])
        self.log("valid_l1", l1_loss, prog_bar=True)
        self.add_to_epoch_metrics("valid_l1", l1_loss)

        # boundary_loss = torch.exp(-5 * F.mse_loss(masks_0_1, 1 - masks_0_1))
        # self.log("valid_bound", boundary_loss, prog_bar=True)

        self.compute_eval_metrics(masks_0_1, annotations[:, 1:], "valid")

        return 0.8 * bce_loss + 0.2 * l1_loss  # + 0.1 * boundary_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr=self.lr, amsgrad=False, weight_decay=1e-6
        )
        # lr_scheduler = MultiStepLR(optimizer, milestones=[20, 40, 60], gamma=0.1)
        # lr_scheduler = CosineAnnealingLR(optimizer, T_max=10)
        return {"optimizer": optimizer}  # , "lr_scheduler": lr_scheduler}

    @torch.no_grad()
    def log_video_sample(self, x, annotations, **kwargs):
        ts = torch.arange(x.size(1)).to(x)
        m0 = annotations[:, :1, :1]
        output = self(x, m0, ts, logits=True).repeat(1, 1, 3, 1, 1)
        return {
            "samples": torch.cat([annotations[:, :1], output[:, 1:]], dim=1),
        }
