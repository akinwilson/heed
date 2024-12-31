import numpy as np
import bisect
import tensorboard
import torch
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from wwv.eval import Metric

from torch.optim.lr_scheduler import ReduceLROnPlateau


class Routine(pl.LightningModule):
    def __init__(self, model, cfg_fitting, cfg_model, localization=False):
        super().__init__()
        self.model = model
        self.metric = Metric
        self.cfg_fitting = cfg_fitting
        self.cfg_model = cfg_model
        self.localization = localization
        self.lr = 1e-3
        self.validation_step_outputs = []
        self.training_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        if self.cfg_model.model_name == "HSTAT":
            mix_lambda = None
            y_hat = self.model(x, mix_lambda)
        else:
            y_hat = self.model(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x = batch["x"]
        y = batch["y"]
        y_hat = self(x)
        y_hat = y_hat.squeeze()
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        y_hat = (F.sigmoid(y_hat) > 0.5).float()

        metrics = self.metric(y_hat, y)()

        metrics_dict = {
            "loss": loss,
            "train_ttr": metrics.ttr,
            "train_ftr": metrics.ftr,
            "train_acc": metrics.acc,
        }
        self.training_step_outputs.append(metrics_dict)
        return metrics_dict

    def on_train_epoch_end(self):
        results = {
            "loss": torch.tensor(
                [x["loss"] for x in self.training_step_outputs]
            ).mean(),
            "ttr": torch.tensor(
                [x["train_ttr"] for x in self.training_step_outputs]
            ).mean(),
            "ftr": torch.tensor(
                [x["train_ftr"] for x in self.training_step_outputs]
            ).mean(),
            "acc": torch.tensor(
                [x["train_acc"] for x in self.training_step_outputs]
            ).mean(),
        }
        # self.log(f"LR",self.lr, on_epoch=True, prog_bar=True, logger=True)
        for (k, v) in results.items():
            self.log(
                f"train_{k}",
                v,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

    def validation_step(self, batch, batch_idx):
        x = batch["x"]
        y = batch["y"]
        y_hat = self(x)
        # (batch, num_classes)
        y_hat = y_hat.squeeze()
        # (batch,)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        pred = F.sigmoid(y_hat)
        y_hat = (pred > 0.5).float()
        metrics = self.metric(y_hat, y)()
        metrics_dict = {
            "val_loss": loss,
            "val_ttr": metrics.ttr,
            "val_ftr": metrics.ftr,
            "val_acc": metrics.acc,
        }
        self.validation_step_outputs.append(metrics_dict)
        return metrics_dict

    def on_validation_epoch_end(self):
        results = {
            "loss": torch.tensor(
                [x["val_loss"] for x in self.validation_step_outputs]
            ).mean(),
            "ttr": torch.tensor(
                [x["val_ttr"] for x in self.validation_step_outputs]
            ).mean(),
            "ftr": torch.tensor(
                [x["val_ftr"] for x in self.validation_step_outputs]
            ).mean(),
            "acc": torch.tensor(
                [x["val_acc"] for x in self.validation_step_outputs]
            ).mean(),
        }
        for (k, v) in results.items():
            self.log(
                f"val_{k}", v, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
            )
            # self.log(f"val_{k}", v, on_epoch=True, prog_bar=True) # , logger=True)

    def test_step(self, batch, batch_idx):
        x = batch["x"]
        y = batch["y"]
        y_hat = self(x)
        # (batch, num_classes)
        y_hat = y_hat.squeeze()
        # (batch,)
        pred = F.sigmoid(y_hat)
        # (batch_probabilities,)
        y_hat = (pred > 0.5).float()
        # (batch_labels,)
        metrics = self.metric(y_hat, y)()
        metrics_dict = {
            "test_ttr": metrics.ttr,
            "test_ftr": metrics.ftr,
            "test_acc": metrics.acc,
        }
        self.test_step_outputs.append(metrics_dict)
        return metrics_dict

    def on_test_epoch_end(self):
        results = {
            "ttr": torch.tensor([x["test_ttr"] for x in self.test_step_outputs]).mean(),
            "ftr": torch.tensor([x["test_ftr"] for x in self.test_step_outputs]).mean(),
            "acc": torch.tensor([x["test_acc"] for x in self.test_step_outputs]).mean(),
        }

        for (k, v) in results.items():
            self.log(
                f"test_{k}",
                v,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

    def configure_optimizers(self):

        # for normal models CNNs etc.
        if self.cfg_model.model_name != "HSTAT":
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.lr,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0.05,
            )
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.1,
                patience=10,
                threshold=0.0001,
                threshold_mode="rel",
                cooldown=0,
                min_lr=0,
                eps=1e-08,
                verbose=False,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_loss",
            }
        else:
            # special scheduler for transformers
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.cfg_fitting.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0.05,
            )

            def lr_scheduler_lambda_1(epoch):
                if epoch < 3:
                    # warm up lr
                    lr_scale = self.cfg_fitting.lr_rate[epoch]
                else:
                    # warmup schedule
                    lr_pos = int(
                        -1
                        - bisect.bisect_left(self.cfg_fitting.lr_scheduler_epoch, epoch)
                    )
                    if lr_pos < -3:
                        lr_scale = max(
                            self.cfg_fitting.lr_rate[0] * (0.98**epoch), 0.03
                        )
                    else:
                        lr_scale = self.cfg_fitting.lr_rate[lr_pos]
                return lr_scale

            scheduler_1 = optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lr_scheduler_lambda_1
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler_1,
                "monitor": "val_loss",
            }
