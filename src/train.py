from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import macro_roc_auc


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for mels, labels in tqdm(loader, desc="Train", leave=False):
        mels, labels = mels.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(mels)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for mels, labels in tqdm(loader, desc="Val", leave=False):
        mels, labels = mels.to(device), labels.to(device)
        logits = model(mels)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        all_preds.append(torch.sigmoid(logits).cpu().numpy())
        all_targets.append(labels.cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    auc = macro_roc_auc(targets, preds)
    return total_loss / len(loader), auc


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    output_dir: Path,
    device: torch.device,
) -> None:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    best_auc = 0.0
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_auc={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), output_dir / "best_model.pth")
            print(f"  Saved best model (auc={best_auc:.4f})")

    print(f"Training complete. Best val AUC: {best_auc:.4f}")
