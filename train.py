import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from model import ViT

device = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, BATCH, LR = 20, 128, 3e-4

print(f"Using device: {device}")

# CIFAR-10 normalization stats
mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)

train_tf = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean, std),
])
test_tf = T.Compose([
    T.ToTensor(),
    T.Normalize(mean, std),
])

train_ds = torchvision.datasets.CIFAR10("../data", train=True, download=True, transform=train_tf)
test_ds = torchvision.datasets.CIFAR10("../data", train=False, download=True, transform=test_tf)
train_dl = DataLoader(train_ds, BATCH, shuffle=True, num_workers=2)
test_dl = DataLoader(test_ds, BATCH, num_workers=2)

model = ViT().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
loss_fn = nn.CrossEntropyLoss()

best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        running_loss += loss.item() * x.size(0)
    sched.step()

    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
    acc = correct / len(test_ds)
    avg_loss = running_loss / len(train_ds)
    print(f"Epoch {epoch + 1}/{EPOCHS}  train loss: {avg_loss:.4f}  test acc: {acc:.3f}")

    os.makedirs("../checkpoints", exist_ok=True)
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "../checkpoints/vit_cifar10_best.pth")

torch.save(model.state_dict(), "../checkpoints/vit_cifar10_final.pth")
print(f"Done. Best test accuracy: {best_acc:.3f}")
print("Saved: checkpoints/vit_cifar10_best.pth and vit_cifar10_final.pth")
