from argparse import ArgumentParser
from os import times

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from spikingjelly.clock_driven import functional
import torchmetrics

from project.utils.polygon2iou import polygon_iou
from chamferdist import ChamferDistance


class OxfordPetModule(pl.LightningModule):
    def __init__(self, learning_rate: float, loss: str, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=["loss"])

        self.model = None  # TODO

        # choice of loss function
        if loss == "mse":
            self.criterion = torch.nn.MSELoss()
        else:
            self.criterion = ChamferDistance()

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        image, class_id, mask, polygon = batch
        y_hat = self(image)
        loss = F.mse_loss(y_hat, polygon)

        iou = polygon_iou(y_hat, mask)

        self.log('train_loss', loss, on_epoch=True, prog_bar=False)
        self.log('train_iou', iou, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image, class_id, mask, polygon = batch
        y_hat = self(image)
        loss = F.mse_loss(y_hat, polygon)

        iou = polygon_iou(y_hat, mask)

        self.log('val_loss', loss, on_epoch=True, prog_bar=False)
        self.log('val_iou', iou, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        image, class_id, mask, polygon = batch
        y_hat = self(image)
        loss = F.mse_loss(y_hat, polygon)

        iou = polygon_iou(y_hat, mask)

        self.log('test_loss', loss, on_epoch=True, prog_bar=False)
        self.log('test_iou', iou, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        # Here, you add every arguments needed for your module
        # NOTE: they must appear as arguments in the __init___() function
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        parser.add_argument('--loss', type=str, choices=["mse", "chamfer"], default="mse")
        return parser
