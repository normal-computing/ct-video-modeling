import torch.optim as optim
import pytorch_lightning as pl

import torch.nn.functional as F
import torch.nn as nn
import torch


class DebugModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.identity = nn.Parameter(torch.tensor([0.0]))

    def training_step(self, batch):
        y = self.identity * batch
        return F.mse_loss(y, batch)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr=1e-3, amsgrad=False, weight_decay=1e-6
        )
        return optimizer
