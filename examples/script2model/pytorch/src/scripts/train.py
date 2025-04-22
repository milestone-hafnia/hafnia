import numpy as np  # noqa E402 Set MKL_SERVICE_FORCE_INTEL to force it

import argparse
import os
from pathlib import Path

import torch
from train_utils import create_dataloaders, create_model, train_loop

from hafnia.experiment import HafniaLogger

DATA_DIR = os.getenv("MDI_DATASET_DIR", "/opt/ml/input/data/training")
ARTIFACT_DIR = os.getenv("MDI_ARTIFACT_DIR", "/opt/ml/output/data")
MODEL_DIR = os.getenv("MDI_MODEL_DIR", "/opt/ml/model")
CKPT_DIR = os.getenv("MDI_CHECKPOINT_DIR", "/opt/ml/checkpoints")


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate for optimizer"
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--log_interval", type=int, default=5, help="Interval for logging")
    parser.add_argument("--max_steps_per_epoch", type=int, default=20, help="Max steps per epoch")
    return parser.parse_args()


def main(args: argparse.Namespace):
    artifacts_dir = Path(ARTIFACT_DIR)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = Path(CKPT_DIR)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path(MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)

    logger = HafniaLogger(artifacts_dir)
    logger.log_configuration(vars(args))
    train_dataloader, test_dataloader = create_dataloaders(DATA_DIR, args.batch_size)
    model = create_model(num_classes=10)
    train_loop(
        logger,
        train_dataloader,
        test_dataloader,
        model,
        args.learning_rate,
        args.epochs,
        args.log_interval,
        ckpt_dir,
        max_steps_per_epoch=args.max_steps_per_epoch,
    )
    torch.save(model.state_dict(), model_dir / "model.pth")


if __name__ == "__main__":
    args = parse_args()
    main(args)
