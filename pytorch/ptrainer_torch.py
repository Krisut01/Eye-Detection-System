import os
import time
import math
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset

from torchvision import datasets, transforms
from contextlib import nullcontext
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


class SplitDataset(Dataset):
    """Top-level wrapper to enable Windows DataLoader multiprocessing pickling.

    Presents a subset of an ImageFolder with a specified transform.
    """
    def __init__(self, base: datasets.ImageFolder, indices, transform):
        self.base = base
        self.indices = list(indices)
        self.transform = transform
        self.classes = base.classes
        self.class_to_idx = base.class_to_idx
        # Keep samples list for class counts and potential samplers
        self.samples = [base.samples[i] for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path, target = self.base.samples[self.indices[idx]]
        img = self.base.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def get_device() -> torch.device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    return device


def build_dataloaders(train_dir: str, test_dir: str, img_size: int = 224, batch_size: int = 16,
                      val_ratio: float = 0.10, test_ratio: float = 0.10, seed: int = 42
                      ) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, int], Dict[int, float]]:
    """Build dataloaders.
    If train_dir == test_dir, perform an in-code split (train/val/test) using class-stratified indices.
    Otherwise, treat train_dir as training and test_dir as validation (legacy behavior) and mirror val to test.
    """
    # Use standard ImageNet normalization to avoid version-specific meta lookups
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=20, scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    test_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Helper dataset view is now defined at top-level (SplitDataset)

    if os.path.abspath(train_dir) == os.path.abspath(test_dir):
        # In-code stratified split
        base = datasets.ImageFolder(train_dir)
        num_classes = len(base.classes)
        # Collect indices per class
        class_to_indices: Dict[int, list] = {c: [] for c in range(num_classes)}
        for idx, (_, label) in enumerate(base.samples):
            class_to_indices[label].append(idx)
        g = torch.Generator().manual_seed(seed)
        train_idx, val_idx, test_idx = [], [], []
        for c, idxs in class_to_indices.items():
            perm = torch.randperm(len(idxs), generator=g).tolist()
            idxs = [idxs[i] for i in perm]
            n = len(idxs)
            n_test = int(round(n * test_ratio))
            n_val = int(round(n * val_ratio))
            n_train = max(n - n_val - n_test, 0)
            train_idx.extend(idxs[:n_train])
            val_idx.extend(idxs[n_train:n_train + n_val])
            test_idx.extend(idxs[n_train + n_val:])

        train_ds = SplitDataset(base, train_idx, train_tfms)
        val_ds = SplitDataset(base, val_idx, test_tfms)
        test_ds = SplitDataset(base, test_idx, test_tfms)

        # Class counts and weights from the training split
        class_counts: Dict[int, int] = {c: 0 for c in range(num_classes)}
        for _, label in train_ds.samples:
            class_counts[label] += 1
        total = len(train_ds)
        class_weights: Dict[int, float] = {c: (total / (len(class_counts) * cnt)) if cnt > 0 else 0.0 for c, cnt in class_counts.items()}
        sample_weights = [class_weights[label] for _, label in train_ds.samples]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        print(f"Found {num_classes} classes: {base.classes}")
        print(f"Train/Val/Test sizes: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
        # Optional: per-class counts in each split
        try:
            def split_counts(indices):
                counts = {c: 0 for c in range(num_classes)}
                for i in indices:
                    _, lbl = base.samples[i]
                    counts[lbl] += 1
                return counts
            trc = split_counts(train_idx)
            vac = split_counts(val_idx)
            tec = split_counts(test_idx)
            print("Per-class counts (train/val/test):")
            for c, name in enumerate(base.classes):
                print(f"  {name}: {trc[c]}/{vac[c]}/{tec[c]}")
        except Exception:
            pass
        return train_loader, val_loader, test_loader, class_counts, class_weights
    else:
        # Legacy two-split behavior: use provided test_dir for validation and final test (same loader)
        train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
        val_ds = datasets.ImageFolder(test_dir, transform=test_tfms)
        test_ds = datasets.ImageFolder(test_dir, transform=test_tfms)

        class_counts: Dict[int, int] = {}
        for _, label in train_ds.samples:
            class_counts[label] = class_counts.get(label, 0) + 1
        total = len(train_ds)
        class_weights: Dict[int, float] = {c: total / (len(class_counts) * cnt) for c, cnt in class_counts.items() if cnt > 0}

        sample_weights = [class_weights[label] for _, label in train_ds.samples]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        print(f"Found {len(train_ds.classes)} classes: {train_ds.classes}")
        print(f"Training samples: {len(train_ds)} | Validation/Test samples: {len(val_ds)}")
        return train_loader, val_loader, test_loader, class_counts, class_weights


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = preds.size(-1)
        log_probs = torch.log_softmax(preds, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


def build_model(num_classes: int) -> nn.Module:
    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model = efficientnet_v2_s(weights=weights)
    for p in model.features.parameters():
        p.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(64, num_classes),
    )
    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_targets, all_preds = [], []
    criterion = nn.CrossEntropyLoss()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss_sum += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += images.size(0)
        all_targets.extend(targets.detach().cpu().numpy())
        all_preds.extend(preds.detach().cpu().numpy())
    return loss_sum / max(total, 1), correct / max(total, 1), np.array(all_targets), np.array(all_preds)


def train(
    train_dir: str,
    test_dir: str,
    save_dir: str,
    img_size: int = 224,
    batch_size: int = 16,
    epochs: int = 50,
    lr: float = 1e-4,
    acc_threshold: float = 0.80,
    fine_tune_ratio: float = 0.3,
    fine_tune_epochs: int = 10,
):
    os.makedirs(save_dir, exist_ok=True)
    device = get_device()

    train_loader, val_loader, test_loader, class_counts, class_weights = build_dataloaders(
        train_dir, test_dir, img_size, batch_size
    )

    num_classes = len(class_counts)
    model = build_model(num_classes).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=8, min_lr=1e-6, verbose=True)

    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            ctx = torch.amp.autocast('cuda') if device.type == 'cuda' else nullcontext()
            with ctx:
                outputs = model(images)
                loss = criterion(outputs, targets)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += images.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d}/{epochs} - train_loss: {train_loss:.4f} acc: {train_acc:.4f} | val_loss: {val_loss:.4f} acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model": model.state_dict(),
                "classes": list(class_counts.keys())
            }, os.path.join(save_dir, "best_model.pt"))
            print(f"Saved new best model at epoch {epoch} with val_acc {val_acc:.4f}")

        # Early stopping if threshold reached
        if val_acc >= acc_threshold:
            print(f"Early stopping: validation accuracy {val_acc:.4f} >= threshold {acc_threshold:.4f}")
            break

    # Fine-tune top layers
    if fine_tune_epochs > 0:
        features = model.features
        num_layers = len(list(features.children()))
        freeze_until = int(num_layers * (1 - fine_tune_ratio))
        for i, m in enumerate(features.children()):
            for p in m.parameters():
                p.requires_grad = i >= freeze_until
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=8, min_lr=1e-6, verbose=True)
        print(f"Fine-tuning last {num_layers - freeze_until} feature blocks...")
        
        best_ft_val_acc = 0.0
        for epoch in range(1, fine_tune_epochs + 1):
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            for images, targets in train_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                ctx = torch.amp.autocast('cuda') if device.type == 'cuda' else nullcontext()
                with ctx:
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += images.size(0)
            train_loss = running_loss / max(total, 1)
            train_acc = correct / max(total, 1)
            val_loss, val_acc, _, _ = evaluate(model, val_loader, device)
            scheduler.step(val_loss)
            print(f"[FT] Epoch {epoch:02d}/{fine_tune_epochs} - train_loss: {train_loss:.4f} acc: {train_acc:.4f} | val_loss: {val_loss:.4f} acc: {val_acc:.4f}")
            
            if val_acc > best_ft_val_acc:
                best_ft_val_acc = val_acc
                torch.save({
                    "model": model.state_dict(),
                    "classes": list(class_counts.keys())
                }, os.path.join(save_dir, "best_ft_model.pt"))
                print(f"Saved new best fine-tuned model at epoch {epoch} with val_acc {val_acc:.4f}")
            # Early stop on threshold during fine-tune as well
            if val_acc >= acc_threshold:
                print(f"Early stopping (fine-tune): validation accuracy {val_acc:.4f} >= threshold {acc_threshold:.4f}")
                break

    # Final evaluation + report
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, device)
    print("\nTest Results:")
    print(f"Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    class_names = datasets.ImageFolder(train_dir).classes
    print(classification_report(y_true, y_pred, target_names=class_names))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_acc"], label="Training Accuracy")
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history["train_loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(save_dir, "training_history_torch.png")
    plt.savefig(out_path)
    print(f"Saved training curves to {out_path}")


def main():
    TRAIN_DIR = r"C:\Subjects 2024\4thYr\CSC-123\Files\Dataset1"
    TEST_DIR = r"C:\Subjects 2024\4thYr\CSC-123\Files\Dataset1"
    SAVE_DIR = r"C:\Subjects 2024\4thYr\CSC-123\Midterm\SavedModels"

    if not os.path.exists(TRAIN_DIR):
        print(f"Training directory not found: {TRAIN_DIR}")
        return
    if not os.path.exists(TEST_DIR):
        print(f"Test directory not found: {TEST_DIR}")
        return

    train(
        train_dir=TRAIN_DIR,
        test_dir=TEST_DIR,
        save_dir=SAVE_DIR,
        img_size=224,
        batch_size=16,
        epochs=30,
        lr=1e-4,
        acc_threshold=0.80,
        fine_tune_ratio=0.3,
        fine_tune_epochs=10,
    )


if __name__ == "__main__":
    main()


