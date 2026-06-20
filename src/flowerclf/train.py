import json

import torch
import torch.nn as nn
from torch import optim

from .config import (
    EPOCHS,
    LEARNING_RATE,
    METRICS_PATH,
    MODEL_PATH,
    MODELS_DIR,
    REPORTS_DIR,
    SEED,
)
from .data import build_dataloaders
from .model import build_model, get_device


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)

    device = get_device()
    train_loader, val_loader, _, classes = build_dataloaders()
    model = build_model(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=LEARNING_RATE
    )

    history, best_val_acc = [], 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        print(
            f"epoch {epoch}/{EPOCHS}  "
            f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}"
        )
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": model.state_dict(), "classes": classes}, MODEL_PATH)

    METRICS_PATH.write_text(
        json.dumps(
            {"history": history, "classes": classes, "best_val_acc": best_val_acc},
            indent=2,
        )
    )
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Saved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
