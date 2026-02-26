import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from src.models.resnet1d import resnet1d18
from src.utils.config import config

class ECGSupervisedModule(pl.LightningModule):
    def __init__(self, lr=1e-3, weight_decay=1e-4, pos_weights=None):
        super().__init__()
        self.save_hyperparameters()
        
        # Model
        self.model = resnet1d18(
            num_leads=config.NUM_LEADS,
            num_classes=len(config.DIAGNOSTIC_SUPERCLASSES),
            kernel_size=15
        )
        
        # Loss
        # Handle class imbalance if weights are provided
        if pos_weights is not None:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weights))
        else:
            self.criterion = nn.BCEWithLogitsLoss()
            
        # Metrics
        self.train_auroc = torchmetrics.AUROC(task="multilabel", num_labels=5, average="macro")
        self.val_auroc = torchmetrics.AUROC(task="multilabel", num_labels=5, average="macro")
        self.test_auroc = torchmetrics.AUROC(task="multilabel", num_labels=5, average="macro")
        
        self.train_f1 = torchmetrics.F1Score(task="multilabel", num_labels=5, average="macro")
        self.val_f1 = torchmetrics.F1Score(task="multilabel", num_labels=5, average="macro")
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, target = batch
        # Handle dict or legacy tensor
        if isinstance(target, dict):
            y = target['disease']
        else:
            y = target
            
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.sigmoid(logits)
        self.train_auroc(preds, y.long())
        self.train_f1(preds, y.long())
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_auroc", self.train_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, target = batch
        if isinstance(target, dict):
            y = target['disease']
        else:
            y = target
            
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.sigmoid(logits)
        self.val_auroc(preds, y.long())
        self.val_f1(preds, y.long())
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_auroc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True)
        
        return loss
        
    def test_step(self, batch, batch_idx):
        x, target = batch
        if isinstance(target, dict):
            y = target['disease']
        else:
            y = target
            
        logits = self(x)
        
        preds = torch.sigmoid(logits)
        self.test_auroc(preds, y.long())
        
        self.log("test_auroc", self.test_auroc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.1, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_auroc",
                "interval": "epoch",
                "frequency": 1
            },
        }
