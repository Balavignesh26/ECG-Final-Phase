import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from src.models.multitask_module import ECGMultiTaskModule
from src.models.ssl_module import ECGSSLModule # To load backbone
from src.data.datamodule import ECGDataModule
from src.utils.config import config
import torch
import argparse

def train(args):
    # Set seed
    pl.seed_everything(config.SEED)
    
    # Data
    print("Setting up DataModule...")
    dm = ECGDataModule(
        batch_size=config.BATCH_SIZE, 
        num_workers=config.NUM_WORKERS,
        data_fraction=args.data_fraction
    )
    
    # Model
    print("Setting up Multi-Task Model...")
    model = ECGMultiTaskModule(lr=1e-3)
    
    # Load Pretrained Weights if provided
    if args.ssl_ckpt:
        print(f"Loading backbone from SSL checkpoint: {args.ssl_ckpt}")
        ssl_model = ECGSSLModule.load_from_checkpoint(args.ssl_ckpt)
        # Copy encoder weights
        # Note: SSL Encoder has NO fc layer (replaced by Identity in implementation or kept but unused)
        # MultiTask encoder expects fc to be Identity
        
        # Load state dict
        backbone_state = ssl_model.encoder.state_dict()
        # Remove 'fc' keys if present to avoid mismatch if shapes differ (though ResNet18 fc is standard)
        # In our ResNet1D, fc is 512->5. In MultiTask, fc is Identity.
        # We should filter out 'fc' weights.
        backbone_state = {k: v for k, v in backbone_state.items() if 'fc' not in k}
        
        missing, unexpected = model.encoder.load_state_dict(backbone_state, strict=False)
        print(f"Weights loaded. Missing: {missing}, Unexpected: {unexpected}")
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.PROJECT_ROOT / "experiments" / "checkpoints" / "multitask",
        filename="multitask-{epoch:02d}-{val_loss_age:.4f}",
        save_top_k=3,
        monitor="val_loss_age", # Monitor Age MAE primarily? Or Combined?
        mode="min"
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed",  # Mixed precision for RTX 3050
        callbacks=[checkpoint_callback, lr_monitor],
        logger=True,
        log_every_n_steps=10,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches
    )
    
    # Train
    print("Starting Multi-Task training...")
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--ssl_ckpt", type=str, default=None, help="Path to SSL checkpoint")
    parser.add_argument("--limit_train_batches", type=float, default=1.0)
    parser.add_argument("--limit_val_batches", type=float, default=1.0)
    parser.add_argument("--data_fraction", type=float, default=1.0, help="Fraction of training data to use")
    args = parser.parse_args()
    
    train(args)
