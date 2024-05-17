# https://towardsdatascience.com/linear-regression-with-pytorch-eb6dedead817

import torch
import pytorch_lightning as pl 
import torch.optim as optim
import numpy as np

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor

    
class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, X):
        out = self.linear(X)
        return out
    

class linearModel(pl.LightningModule):
    def __init__(self, lr, input_size, output_size, warmup=100, max_iters=1000):
        super().__init__()
        self.save_hyperparameters()
        self.step = 0

        self.linear_regression = linearRegression(input_size, output_size)
        self.metric = torch.nn.MSELoss()

        
    def forward(self, inputs):
        x1, x2 = inputs 
        XInput = torch.cat((x1, x2),dim=1)
        return self.linear_regression.forward(XInput)
    
    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)
        
    def training_step(self, batch, batch_idx):
        self.step = self.step + 1
        
        inputs, target = batch
        pred = self.forward(inputs)
        loss = self.metric(pred, target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("lr", self.lr_scheduler.get_last_lr()[0], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        
        if self.step % 100 == 0:
            print("Step: ", self.step, "Lr: ", self.lr_scheduler.get_last_lr()[0])
            print(loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, target = batch

        preds = self.forward(inputs)
        loss = self.metric(preds, target)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
    
        return loss
    
    def configure_optimizers(self):
        #https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        
        self.lr_scheduler = CosineWarmupScheduler(optimizer, 
                                                  warmup=self.hparams.warmup, 
                                                  max_iters=self.hparams.max_iters)
        
        return {'optimizer': optimizer, "lr_scheduler": self.lr_scheduler}
    
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()  # Step per iteration
