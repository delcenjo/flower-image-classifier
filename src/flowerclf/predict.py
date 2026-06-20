import argparse

import torch
from PIL import Image

from .config import MODEL_PATH
from .data import eval_transform
from .model import build_model, get_device


def predict(image_path):
    device = get_device()
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    classes = checkpoint["classes"]
    model = build_model(len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    image = Image.open(image_path).convert("RGB")
    tensor = eval_transform()(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    ranked = sorted(zip(classes, probabilities.tolist()), key=lambda x: x[1], reverse=True)
    return ranked


def main():
    parser = argparse.ArgumentParser(description="Classify a flower image")
    parser.add_argument("image", help="path to an image file")
    args = parser.parse_args()
    for label, probability in predict(args.image):
        print(f"{label:>12}  {probability:.1%}")


if __name__ == "__main__":
    main()
