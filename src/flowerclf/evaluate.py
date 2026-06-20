import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from .config import FIGURES_DIR, METRICS_PATH, MODEL_PATH
from .data import IMAGENET_MEAN, IMAGENET_STD, build_dataloaders
from .model import build_model, get_device


def collect_predictions(model, loader, device):
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            y_pred.extend(outputs.argmax(1).cpu().tolist())
            y_true.extend(labels.tolist())
    return np.array(y_true), np.array(y_pred)


def plot_confusion_matrix(y_true, y_pred, classes, accuracy):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(
        ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45
    )
    ax.set_title(f"Confusion matrix (test accuracy {accuracy:.2f})")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=120)
    plt.close(fig)


def plot_sample_predictions(model, loader, classes, device, n=8):
    images, labels = next(iter(loader))
    with torch.no_grad():
        preds = model(images.to(device)).argmax(1).cpu()
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    fig, axes = plt.subplots(2, 4, figsize=(11, 6))
    for ax, img, pred, true in zip(axes.ravel(), images[:n], preds[:n], labels[:n]):
        ax.imshow((img * std + mean).permute(1, 2, 0).clamp(0, 1).numpy())
        ax.set_title(
            f"{classes[pred]}\n(true: {classes[true]})",
            color="green" if pred == true else "red",
            fontsize=9,
        )
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sample_predictions.png", dpi=120)
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    classes = checkpoint["classes"]
    model = build_model(len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    _, _, test_loader, _ = build_dataloaders()
    y_true, y_pred = collect_predictions(model, test_loader, device)
    accuracy = float((y_true == y_pred).mean())
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)

    plot_confusion_matrix(y_true, y_pred, classes, accuracy)
    plot_sample_predictions(model, test_loader, classes, device)

    metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    metrics["test_accuracy"] = accuracy
    metrics["classification_report"] = report
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Saved figures -> {FIGURES_DIR}")


if __name__ == "__main__":
    main()
