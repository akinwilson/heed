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

    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.metric = Metric
        self.cfg = cfg
        self.lr = 1e-3


    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x = batch['x']
        y = batch['y']
        y_hat = self.model(x)
        y_hat = y_hat.squeeze()
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        y_hat = (F.sigmoid(y_hat) > 0.5).float()

        metrics = self.metric(y_hat, y)()
        return {"loss":loss, "train_ttr": metrics.ttr, "train_ftr": metrics.ftr, "train_acc": metrics.acc}


    def training_epoch_end(self, training_step_outputs):
        results = {
            "loss": torch.tensor([x['loss'].float().item() for x in training_step_outputs]).mean(),
            "ttr": torch.tensor([x['train_ttr'].float().mean().item() for x in training_step_outputs]).mean(),
            "ftr": torch.tensor([x['train_ftr'].float().mean().item() for x in training_step_outputs]).mean(),
            "acc": torch.tensor([x['train_acc'].float().mean().item() for x in training_step_outputs]).mean()
            }
        # self.log(f"LR",self.lr, on_epoch=True, prog_bar=True, logger=True)
        for (k,v) in results.items():
            self.log(f"train_{k}", v, on_epoch=True, prog_bar=True, logger=True)    


    def validation_step(self, batch, batch_idx):
        x = batch['x']
        y = batch['y']
        y_hat = self.model(x)
        # (batch, num_classes)
        y_hat = y_hat.squeeze()
        # (batch,)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        pred = F.sigmoid(y_hat)
        y_hat = (pred > 0.5).float()
        metrics = self.metric(y_hat, y)()
        return {"val_loss": loss, "val_ttr": metrics.ttr, "val_ftr": metrics.ftr, "val_acc": metrics.acc}


    def validation_epoch_end(self, validation_step_outputs):
        results = {
            "loss": torch.tensor([x['val_loss'].float().mean().item() for x in validation_step_outputs]).mean(),
            "ttr": torch.tensor([x['val_ttr'].float().mean().item() for x in validation_step_outputs]).mean(),
            "ftr": torch.tensor([x['val_ftr'].float().mean().item() for x in validation_step_outputs]).mean(),
            "acc": torch.tensor([x['val_acc'].float().mean().item() for x in validation_step_outputs]).mean()
            }
        for (k,v) in results.items():
            self.log(f"val_{k}", v, on_epoch=True, prog_bar=True, logger=True)    


    def test_step(self, batch, batch_idx):
        x = batch['x']
        y = batch['y']
        y_hat = self.model(x)
        # (batch, num_classes)
        y_hat = y_hat.squeeze()
        # (batch,)
        pred = F.sigmoid(y_hat)
        # (batch_probabilities,)
        y_hat = (pred > 0.5).float()
        # (batch_labels,)
        metrics = self.metric(y_hat, y)()
        return {"test_ttr": metrics.ttr, "test_ftr": metrics.ftr, "test_acc": metrics.acc}


    def test_epoch_end(self, test_step_outputs):
        results = {
            "ttr": torch.tensor([x['test_ttr'].float().mean().item() for x in test_step_outputs]).mean(),
            "ftr": torch.tensor([x['test_ftr'].float().mean().item() for x in test_step_outputs]).mean(),
            "acc": torch.tensor([x['test_acc'].float().mean().item() for x in test_step_outputs]).mean()
            }

        for (k,v) in results.items():
            self.log(f"test_{k}", v, on_epoch=True, prog_bar=True, logger=True)    


    def configure_optimizers(self):
        
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr = self.lr, 
            betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0.05, 
        )

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
        
        return  {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"} 




class HTSwinRoutine(pl.LightningModule):
    def __init__(self, model, config, cfg_fitting):
        super().__init__()
        self.model = model
        self.config = config
        self.cfg_fitting = cfg_fitting
        self.loss_func = F.cross_entropy

    def forward(self, x, mix_lambda = None):
        output_dict = self.model(x, mix_lambda)
        return output_dict["clipwise_output"], output_dict["framewise_output"]

    def inference(self, x):
        self.eval()
        x = torch.from_numpy(x).float().to(self.device_type)
        output_dict = self.model(x, None, True)
        for key in output_dict.keys():
            output_dict[key] = output_dict[key].detach().cpu().numpy()
        return output_dict

    def training_step(self, batch, batch_idx):
        mix_lambda = None
        pred, _ = self(batch["waveform"], mix_lambda)
        loss = self.loss_func(pred, batch["target"])
        return loss

    def validation_step(self, batch, batch_idx):
        mix_lambda = None
        pred, _ = self(batch["waveform"], mix_lambda)
        loss = self.loss_func(pred, batch["target"])
        return [pred.detach(), batch["target"].detach(), loss.detach()]
    

    def validation_epoch_end(self, validation_step_outputs):
        pred = torch.cat([d[0] for d in validation_step_outputs], dim = 0)
        target = torch.cat([d[1] for d in validation_step_outputs], dim = 0)
        loss = torch.cat([d[2] for d in validation_step_outputs], dim = 0)

        gather_pred = pred.cpu().numpy()
        gather_target = target.cpu().numpy()
        gather_loss = loss.cpu().numpy()

        ###############################  ############################### ###############################
        avg_loss = gather_loss.mean().item()
        gather_pred = np.argmax(gather_pred, 1)
        metrics = Metric(gather_pred, gather_target)()
        metric_dict = {"ttr": metrics.ttr, "ftr": metrics.ftr, "acc": metrics.acc}
        ###############################  ############################### ###############################
        self.log("val_loss", avg_loss, on_epoch=True, prog_bar=True, logger=True)
        for (k,v) in metric_dict.items():
            self.log(f"val_{k}", v, on_epoch=True, prog_bar=True, logger=True) 

        
    def time_shifting(self, x, shift_len):
        shift_len = int(shift_len)
        new_sample = torch.cat([x[:, shift_len:], x[:, :shift_len]], axis = 1)
        return new_sample 


    def test_step(self, batch, batch_idx):
        preds = []
        # time shifting optimization
        if self.config.fl_local: 
            shift_num = 1 # framewise localization cannot allow the time shifting
        else:
            shift_num = 10 
        for i in range(shift_num):
            pred, pred_map = self(batch["waveform"])
            preds.append(pred.unsqueeze(0))
            batch["waveform"] = self.time_shifting(batch["waveform"], shift_len = 100 * (i + 1))
        preds = torch.cat(preds, dim=0)
        pred = preds.mean(dim = 0)
        if self.config.fl_local:
            return [
                pred.detach().cpu().numpy(), 
                pred_map.detach().cpu().numpy(),
                batch["audio_name"],
                batch["real_len"].cpu().numpy()
            ]
        else:
            return [pred.detach(), batch["target"].detach()]


    def test_epoch_end(self, test_step_outputs):
        y_hat = torch.cat([d[0] for d in test_step_outputs], dim = 0)
        y = torch.cat([d[1] for d in test_step_outputs], dim = 0)
        
        gather_pred = y_hat.cpu().numpy()
        gather_target = y.cpu().numpy()
    
        gather_pred = np.argmax(gather_pred, 1)
        metrics = Metric(gather_pred, gather_target)()
        metric_dict = {"ttr": metrics.ttr, "ftr": metrics.ftr, "acc": metrics.acc}
        for (k,v) in metric_dict.items():
            self.log(f"test_{k}", v, on_epoch=True, prog_bar=True, logger=True) 


    def configure_optimizers(self):


        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr = self.cfg_fitting.learning_rate, 
            betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0.05, 
        )
        
        def lr_scheduler_lambda_1(epoch):       
            if epoch < 3:
                # warm up lr
                lr_scale = self.cfg_fitting.lr_rate[epoch]
            else:
                # warmup schedule
                lr_pos = int(-1 - bisect.bisect_left(self.cfg_fitting.lr_scheduler_epoch, epoch))
                if lr_pos < -3:
                    lr_scale = max(self.cfg_fitting.lr_rate[0] * (0.98 ** epoch), 0.03 )
                else:

                    lr_scale = self.cfg_fitting.lr_rate[lr_pos]

            return lr_scale


        scheduler_1 = optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lr_scheduler_lambda_1)

        # scheduler_2 = {
        #     "scheduler":  ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False),
        #     "monitor": "val_loss"
        # }

        # scheduler = ChainedScheduler([scheduler_1, scheduler_2])


        # return  {"optimizer": [optimizer], "lr_scheduler": [lr_scheduler_1, lr_scheduler_2], "monitor": "val_loss"} 
        return [optimizer], [scheduler_1]



