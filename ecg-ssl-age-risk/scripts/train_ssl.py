import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from src.models.ssl_module import ECGSSLModule
from src.data.datamodule import ECGDataModule
from src.utils.config import config
import torch
import argparse

def train(args):
    # Set seed
    pl.seed_everything(config.SEED)
    
    # Data
    print("Setting up DataModule...")
    dm = ECGDataModule(batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)
    
    # Model
    print(f"Setting up SSL Model with strategy: {args.masking_strategy}")
    model = ECGSSLModule(
        lr=1e-3, 
        mask_ratio=args.mask_ratio,
        masking_strategy=args.masking_strategy
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.PROJECT_ROOT / "experiments" / "checkpoints" / "ssl",
        filename="resnet1d-ssl-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss",
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
    print("Starting SSL training...")
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--mask_ratio", type=float, default=0.5)
    parser.add_argument("--masking_strategy", type=str, default='random', choices=['random', 'leads', 'block'])
    parser.add_argument("--limit_train_batches", type=float, default=1.0)
    parser.add_argument("--limit_val_batches", type=float, default=1.0)
    args = parser.parse_args()
    
    train(args)
