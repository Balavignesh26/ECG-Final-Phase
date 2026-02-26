import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.resnet1d import resnet1d18
from src.models.decoder1d import Decoder1D
from src.utils.masking import mask_random_patches, mask_leads, mask_temporal_block
from src.utils.config import config

class ECGSSLModule(pl.LightningModule):
    def __init__(self, lr=1e-3, mask_ratio=0.5, masking_strategy='random'):
        super().__init__()
        self.save_hyperparameters()
        
        # Encoder
        self.encoder = resnet1d18(num_leads=config.NUM_LEADS, num_classes=5, kernel_size=15)
        # We don't need the fc layer for SSL, but keeping it logic intact is fine.
        # We will use encoder.forward_features(x)
        
        # Decoder
        # Encoder output channels = 512 for ResNet18
        self.decoder = Decoder1D(latent_dim=512, output_channels=config.NUM_LEADS, output_len=int(config.TARGET_SAMPLING_RATE * config.DURATION))
        
        self.masking_strategy = masking_strategy
        
    def forward(self, x):
        # For inference/embedding extraction
        return self.encoder.forward_features(x)
        
    def training_step(self, batch, batch_idx):
        x, _ = batch # Ignore labels for SSL
        
        # 1. Apply Masking
        if self.masking_strategy == 'random':
            masked_x, mask = mask_random_patches(x, mask_ratio=self.hparams.mask_ratio)
        elif self.masking_strategy == 'leads':
            masked_x, mask = mask_leads(x, num_leads_to_mask=2) # Parametrize?
        elif self.masking_strategy == 'block':
            masked_x, mask = mask_temporal_block(x, block_size=int(0.2 * x.shape[2]))
        else:
            masked_x, mask = mask_random_patches(x, mask_ratio=0.5)
            
        # 2. Encode
        features = self.encoder.forward_features(masked_x)
        
        # 3. Decode
        recon_x = self.decoder(features)
        
        # 4. Loss (MSE)
        # Calculate loss only on masked regions?
        # mask is (B, C, L), 1 = masked.
        loss = F.mse_loss(recon_x, x, reduction='none')
        
        # Mean over masked regions
        # Add epsilon to avoid div by zero if no mask (unlikely)
        mask_loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        
        # Also track global reconstruction loss for sanity
        global_loss = loss.mean()
        
        self.log("train_loss", mask_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_global_loss", global_loss, on_step=False, on_epoch=True)
        
        return mask_loss
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        
        # Validation with fixed masking (e.g., random)
        masked_x, mask = mask_random_patches(x, mask_ratio=0.5)
        features = self.encoder.forward_features(masked_x)
        recon_x = self.decoder(features)
        
        loss = F.mse_loss(recon_x, x, reduction='none')
        mask_loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        
        self.log("val_loss", mask_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return mask_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            },
        }
