# Vision Transformer (ViT) — CIFAR-10

A small Vision Transformer built from scratch in PyTorch, trained on CIFAR-10
(10 classes of 32x32 images: airplane, automobile, bird, cat, deer, dog,
frog, horse, ship, truck).

## How it works

1. The image is split into small patches (4x4 pixels).
2. Each patch is flattened and linearly embedded into a vector, like a "word" token.
3. A learnable `[CLS]` token is prepended, and positional embeddings are added.
4. The sequence passes through several Transformer encoder blocks (self-attention + MLP).
5. The final `[CLS]` token representation is passed through a linear layer to predict the class.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```bash
cd src
python train.py
```

This downloads CIFAR-10 automatically into `../data/` and saves the best
and final checkpoints into `../checkpoints/`. Expect roughly 75-80% test
accuracy after 20 epochs on a GPU.

## Predict on your own image

```bash
cd src
python predict.py path/to/your_image.jpg
```

## Project structure

```
vision-transformer/
├── src/
│   ├── model.py     # ViT architecture (PatchEmbedding, TransformerBlock, ViT)
│   ├── train.py     # training loop on CIFAR-10
│   └── predict.py   # run inference on a single image
├── configs/         # (for future hyperparameter files)
├── data/            # downloaded dataset (git-ignored)
├── checkpoints/      # saved model weights (git-ignored)
├── requirements.txt
├── .gitignore
└── README.md
```

## Notes

- This ViT is trained from scratch, so it will not match the accuracy of a
  pretrained ViT. For real-world use, fine-tune a pretrained model instead,
  e.g. `torchvision.models.vit_b_16(weights="DEFAULT")` or the Hugging Face
  `google/vit-base-patch16-224` checkpoint.
- Tweak `dim`, `depth`, and `heads` in `model.py` to make the model bigger
  or smaller.
