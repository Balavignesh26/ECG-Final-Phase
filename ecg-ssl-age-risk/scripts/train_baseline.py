import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger
from src.models.supervised_module import ECGSupervisedModule
from src.data.datamodule import ECGDataModule
from src.utils.config import config
import torch

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
    print("Setting up Model...")
    # Calculate pos_weights for imbalance adjustment? 
    # For now, let's start without, or we can calculate it from the dataset if needed.
    # Given the imbalance, standard BCE might struggle with 'HYP', but let's establish baseline first.
    model = ECGSupervisedModule(lr=1e-3)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.PROJECT_ROOT / "experiments" / "checkpoints" / "baseline",
        filename="resnet1d-{epoch:02d}-{val_auroc:.4f}",
        save_top_k=3,
        monitor="val_auroc",
        mode="max"
    )
    
    early_stopping = EarlyStopping(
        monitor="val_auroc",
        patience=10,
        mode="max"
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed",  # Mixed precision for RTX 3050
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        logger=True,
        log_every_n_steps=10,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        limit_test_batches=args.limit_test_batches,
    )
    
    # Train
    print("Starting training...")
    trainer.fit(model, datamodule=dm)
    
    # Test
    print("Starting testing...")
    trainer.test(model, datamodule=dm, ckpt_path="best")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--limit_train_batches", type=float, default=1.0)
    parser.add_argument("--limit_val_batches", type=float, default=1.0)
    parser.add_argument("--limit_test_batches", type=float, default=1.0)
    parser.add_argument("--data_fraction", type=float, default=1.0, help="Fraction of training data to use")
    args = parser.parse_args()
    
    train(args)
