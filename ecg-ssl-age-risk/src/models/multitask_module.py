import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from src.models.resnet1d import resnet1d18
from src.utils.config import config

class ECGMultiTaskModule(pl.LightningModule):
    def __init__(self, lr=1e-3, weight_decay=1e-4, pos_weights=None):
        super().__init__()
        self.save_hyperparameters()
        
        # Shared Encoder
        self.encoder = resnet1d18(num_leads=config.NUM_LEADS, num_classes=5, kernel_size=15)
        # We replace the default fc layer (which outputs 5) with Identity to get 512 features
        self.encoder.fc = nn.Identity()
        
        # Task Heads
        # ResNet18 output dim is 512
        self.age_head = nn.Linear(512, 1)
        self.disease_head = nn.Linear(512, len(config.DIAGNOSTIC_SUPERCLASSES))
        
        # Loss Weighting (Uncertainty)
        # log_vars[0] for Age, log_vars[1] for Disease
        self.log_vars = nn.Parameter(torch.zeros(2))
        
        # Losses
        self.age_criterion = nn.L1Loss() # MAE
        if pos_weights is not None:
            self.disease_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weights))
        else:
            self.disease_criterion = nn.BCEWithLogitsLoss()
            
        # Metrics
        self.train_age_mae = torchmetrics.MeanAbsoluteError()
        self.val_age_mae = torchmetrics.MeanAbsoluteError()
        
        self.train_disease_auroc = torchmetrics.AUROC(task="multilabel", num_labels=5, average="macro")
        self.val_disease_auroc = torchmetrics.AUROC(task="multilabel", num_labels=5, average="macro")
        
    def forward(self, x):
        features = self.encoder(x) # (Batch, 512)
        
        age_pred = self.age_head(features) # (Batch, 1)
        disease_logits = self.disease_head(features) # (Batch, 5)
        
        return age_pred, disease_logits
    
    def training_step(self, batch, batch_idx):
        x, target = batch
        age_y = target['age']
        disease_y = target['disease']
        
        # Forward
        age_pred, disease_logits = self(x)
        
        # Calculate individual losses
        loss_age = self.age_criterion(age_pred.view(-1), age_y)
        loss_disease = self.disease_criterion(disease_logits, disease_y)
        
        # Uncertainty Weighting
        # L = L1 / (2*sigma1^2) + L2 / (2*sigma2^2) + log(sigma1) + log(sigma2)
        # log_var = log(sigma^2)
        # precision = 1 / sigma^2 = exp(-log_var)
        
        precision_age = torch.exp(-self.log_vars[0])
        precision_disease = torch.exp(-self.log_vars[1])
        
        loss = (precision_age * loss_age) + self.log_vars[0] + \
               (precision_disease * loss_disease) + self.log_vars[1]
        
        # Metrics
        self.train_age_mae(age_pred.view(-1), age_y)
        self.train_disease_auroc(torch.sigmoid(disease_logits), disease_y.long())
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loss_age", loss_age, on_step=False, on_epoch=True)
        self.log("train_loss_disease", loss_disease, on_step=False, on_epoch=True)
        self.log("sigma_age", torch.exp(0.5 * self.log_vars[0]), on_step=False, on_epoch=True)
        self.log("sigma_disease", torch.exp(0.5 * self.log_vars[1]), on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, target = batch
        age_y = target['age']
        disease_y = target['disease']
        
        age_pred, disease_logits = self(x)
        
        loss_age = self.age_criterion(age_pred.view(-1), age_y)
        loss_disease = self.disease_criterion(disease_logits, disease_y)
        
        self.val_age_mae(age_pred.view(-1), age_y)
        self.val_disease_auroc(torch.sigmoid(disease_logits), disease_y.long())
        
        self.log("val_loss_age", loss_age, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mae", self.val_age_mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_auroc", self.val_disease_auroc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss_age + loss_disease # Log sum just for monitoring

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss_age", # Or some combined metric
                "interval": "epoch",
                "frequency": 1
            },
        }
