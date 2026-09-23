import sys

import torch
import torchvision.transforms as T
from PIL import Image

from model import ViT

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

CHECKPOINT = "../checkpoints/vit_cifar10_best.pth"


def load_model(checkpoint_path=CHECKPOINT):
    model = ViT()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model


def predict(image_path, model):
    tf = T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    img = tf(Image.open(image_path).convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        probs = model(img).softmax(1)[0]

    top_idx = probs.argmax().item()
    return CLASSES[top_idx], probs[top_idx].item(), probs


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)

    model = load_model()
    label, confidence, all_probs = predict(sys.argv[1], model)
    print(f"Prediction: {label} ({confidence:.1%})")

    print("\nTop 3:")
    top3 = torch.topk(all_probs, 3)
    for score, idx in zip(top3.values, top3.indices):
        print(f"  {CLASSES[idx]:12s} {score.item():.1%}")
