"""
train_shot_model.py  — FIXED VERSION
=====================================
Key fixes over original:
  1. 20 epochs instead of 5 (was the main reason for bad accuracy)
  2. 80/20 train/val split — now tracks validation accuracy
  3. Only saves the best model (highest val accuracy), not the last epoch
  4. LR scheduler — halves LR when val accuracy plateaus
  5. Freeze early ResNet layers, only fine-tune last block + FC layer
     (standard transfer-learning best practice for small datasets)
  6. More augmentations: ColorJitter + stronger rotation
  7. Dynamic num_classes read from dataset (no hardcoding)
  8. Prints per-epoch train loss + val accuracy table
"""

import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR       = "data/shot_dataset"
MODEL_PATH     = "models/shot_type_model.pth"
CLASS_MAP_PATH = "models/shot_type_classes.json"

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE  = 32
EPOCHS      = 20          # FIX 1: was 5 — needs at least 15–20 to actually learn
LR          = 3e-4
VAL_SPLIT   = 0.20        # FIX 2: hold out 20 % for validation
SEED        = 42

# ── Transforms ────────────────────────────────────────────────────────────────
# FIX 3: added ColorJitter and stronger rotation for better generalisation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),                          # was 10
    transforms.ColorJitter(brightness=0.3, contrast=0.3,   # NEW
                           saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])


def build_loaders():
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
    n_val   = int(len(full_dataset) * VAL_SPLIT)
    n_train = len(full_dataset) - n_val

    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )

    # Apply val transform to val split (no augmentation)
    val_ds.dataset = datasets.ImageFolder(DATA_DIR, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, full_dataset.classes


def build_model(num_classes: int) -> nn.Module:
    # FIX 4: freeze all layers except layer4 + FC so pretrained features are preserved
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False          # freeze early layers

    model.fc = nn.Linear(model.fc.in_features, num_classes)   # dynamic class count
    return model


def train():
    torch.manual_seed(SEED)
    os.makedirs("models", exist_ok=True)

    train_loader, val_loader, classes = build_loaders()
    num_classes = len(classes)

    print(f"\n{'='*55}")
    print(f"  Classes ({num_classes}): {classes}")
    print(f"  Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")
    print(f"  Epochs: {EPOCHS}  |  Batch size: {BATCH_SIZE}  |  LR: {LR}")
    print(f"{'='*55}\n")

    # Save class map for shot_classifier.py to load
    with open(CLASS_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)
    print(f"[train] Class map saved → {CLASS_MAP_PATH}")

    model    = build_model(num_classes)
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model    = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )

    # FIX 5: LR scheduler — halves LR when val accuracy stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3,
    )

    best_val_acc = 0.0
    print(f"{'Epoch':>6} {'Train Loss':>11} {'Val Acc':>9} {'Status':>8}")
    print("-" * 40)

    for epoch in range(1, EPOCHS + 1):

        # ── Training ──────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        val_acc = correct / total if total > 0 else 0.0
        scheduler.step(val_acc)

        # FIX 6: save only the best checkpoint
        status = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            status = "✅ saved"

        print(f"{epoch:>6} {avg_loss:>11.4f} {val_acc:>8.2%} {status:>8}")

    print(f"\n[train] Best validation accuracy : {best_val_acc:.2%}")
    print(f"[train] Model saved → {MODEL_PATH}")
    print("\nYou can now run:  python main.py")


if __name__ == "__main__":
    train()