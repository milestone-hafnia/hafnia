from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision.models import resnet18
from torchvision.transforms import v2

from hafnia.data import load_dataset
from hafnia.experiment import HafniaLogger


def create_transforms() -> v2.Compose:
    """
    Creates a composition of image transformations for data augmentation and normalization.

    Returns:
        transforms.Compose: A composed transform for image preprocessing.
    """
    return v2.Compose(
        [
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def create_dataloaders(
    data_root: str, batch_size: int, num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates training and testing DataLoaders.

    Args:
        data_root (str): Root directory of the dataset.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and testing DataLoaders.
    """
    transforms = create_transforms()
    hafnia_dataset = load_dataset(data_root)
    train_split = hafnia_dataset["train"]
    train_split.set_transform(transforms)
    test_split = hafnia_dataset["test"]
    test_split.set_transform(transforms)

    train_loader = DataLoader(
        train_split, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_split, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader


def create_model(num_classes: int) -> nn.Module:
    """
    Creates and returns a ResNet18 model adjusted for the specified number of classes.

    Args:
        num_classes (int): Number of output classes.

    Returns:
        nn.Module: The modified ResNet18 model.
    """
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def run_train_epoch(
    epoch: int,
    dataloader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    criterion: nn.Module,
    metrics: nn.Module,
    device: torch.device,
    ml_logger: HafniaLogger,
    log_interval: int,
    max_steps_per_epoch: int,
) -> Dict[str, float]:
    """
    Runs one epoch of training.

    Args:
        epoch (int): Current epoch number.
        dataloader (DataLoader): Training DataLoader.
        model (nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): Optimizer.
        criterion (nn.Module): Loss function.
        metrics (MulticlassAccuracy): Metrics calculator.
        device (torch.device): Computation device.
        ml_logger (HafniaLogger): Logger for metrics.
        log_interval (int): Interval for logging.
        max_steps_per_epoch (int): Maximum steps per epoch.

    Returns:
        Dict[str, float]: Dictionary containing average loss and accuracy.
    """
    model.train()
    metrics.reset()
    epoch_loss = 0.0
    total_samples = 0

    iteration = epoch * len(dataloader)

    for i, batch in enumerate(dataloader):
        inputs, targets = batch["image"], batch["label"]
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
        metrics.update(outputs.softmax(dim=1), targets)
        iteration += i

        if i % log_interval == 0:
            avg_loss = epoch_loss / total_samples
            accuracy = metrics.compute().item()
            ml_logger.log_scalar("train/loss", avg_loss, iteration)

        if i >= max_steps_per_epoch:
            print(f"Max steps per train epoch reached: {max_steps_per_epoch}")
            break

    avg_loss = epoch_loss / total_samples
    accuracy = metrics.compute().item()
    ml_logger.log_metric(name="train/accuracy", value=accuracy, step=iteration)

    return {
        "train/loss": avg_loss,
        "train/accuracy": accuracy,
    }


@torch.no_grad()
def run_eval(
    epoch: int,
    dataloader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    metrics: nn.Module,
    device: torch.device,
    ml_logger: HafniaLogger,
):
    """
    Runs evaluation on the test dataset.

    Args:
        epoch (int): Current epoch number.
        dataloader (DataLoader): Evaluation DataLoader.
        model (nn.Module): The model to evaluate.
        criterion (nn.Module): Loss function.
        metrics (MulticlassAccuracy): Metrics calculator.
        device (torch.device): Computation device.
        ml_logger (HafniaLogger): Logger for metrics.

    Returns:
        Dict[str, float]: Dictionary containing average loss and accuracy.
    """
    model.eval()
    metrics.reset()
    epoch_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        inputs, targets = batch["image"], batch["label"]
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        epoch_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
        metrics.update(outputs.softmax(dim=1), targets)

    avg_loss = epoch_loss / total_samples
    accuracy = metrics.compute().item()
    step = (epoch + 1) * len(dataloader)

    ml_logger.log_scalar("val/loss", avg_loss, step)
    ml_logger.log_metric("val/accuracy", accuracy, step)

    return {
        "val/loss": avg_loss,
        "val/accuracy": accuracy,
    }


def train_loop(
    logger: HafniaLogger,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    model: nn.Module,
    learning_rate: float,
    epochs: int,
    log_interval: int,
    ckpt_dir: str,
    max_steps_per_epoch: int,
):
    """
    Main training loop.

    Args:
        logger (HafniaLogger): Logger for metrics.
        train_dataloader (DataLoader): Training DataLoader.
        test_dataloader (DataLoader): Testing DataLoader.
        model (nn.Module): The model to train.
        learning_rate (float): Learning rate for the optimizer.
        epochs (int): Number of epochs to train.
        log_interval (int): Interval for logging metrics.
        ckpt_dir (str): Directory to save checkpoints.
        max_steps_per_epoch (int): Maximum steps per epoch.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    model.train()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    metrics = Accuracy(task="multiclass", num_classes=10).to(device)

    for epoch in range(epochs):
        run_train_epoch(
            epoch,
            train_dataloader,
            model,
            optimizer,
            scheduler,
            criterion,
            metrics,
            device,
            logger,
            log_interval,
            max_steps_per_epoch,
        )
        eval_metrics = run_eval(epoch, test_dataloader, model, criterion, metrics, device, logger)
        ckpt_fname = f"accuracy_{eval_metrics['accuracy']:.2f}_epoch_{epoch}.pth"
        torch.save(model.state_dict(), f"{ckpt_dir}/{ckpt_fname}")
