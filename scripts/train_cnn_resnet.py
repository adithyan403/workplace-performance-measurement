"""Train a ResNet-18 CNN to detect KSRTC WPM activities from labeled frames.

Reads labeled frames from D:\\ksrtc\\frames_labeled\\<class>\\*.jpg
(classes: write, crew, check, wait), trains a transfer-learned ResNet-18
with class weighting for imbalance, and saves the best model + a JSON
report + confusion-matrix chart.

Usage:
    python train_cnn_resnet.py                     # defaults below
    python train_cnn_resnet.py --epochs 40 --batch 32 --lr 1e-4
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import models, transforms

ROOT = Path(r"D:\ksrtc\wpm_webapp")
CLASSES = ["write", "crew", "check", "wait"]
LABELED_DIR = Path(r"D:\ksrtc\frames_labeled")
MODEL_DIR = ROOT / "data"
MODEL_PATH = MODEL_DIR / "cnn_resnet18.pt"
REPORT_PATH = MODEL_DIR / "cnn_report.json"

IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class FrameDataset(Dataset):
    def __init__(self, root: Path, classes, transform=None):
        self.samples = []
        class_to_idx = {c: i for i, c in enumerate(classes)}
        for cls in classes:
            d = root / cls
            if not d.is_dir():
                continue
            for p in sorted(d.glob("*.jpg")):
                self.samples.append((str(p), class_to_idx[cls]))
        self.transform = transform
        if not self.samples:
            raise RuntimeError(f"no labeled frames found under {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, cls = self.samples[i]
        from PIL import Image
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, cls


def train():
    ap = argparse.ArgumentParser(description="Train ResNet-18 activity CNN")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    ds = FrameDataset(LABELED_DIR, CLASSES, transform=train_tf)
    n = len(ds)
    print(f"labeled frames: {n}  classes: {CLASSES}")

    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(n)
    n_test = int(n * args.test_frac)
    n_val = int(n * args.val_frac)
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    print(f"splits: train {len(train_idx)}  val {len(val_idx)}  test {len(test_idx)}")

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)
    test_ds = Subset(ds, test_idx)

    # class-weighted sampling to compensate for imbalance (crew is rare)
    labels = [ds.samples[i][1] for i in train_idx]
    counts = np.bincount(labels, minlength=len(CLASSES))
    weights = 1.0 / (counts.astype(float) + 1e-6)
    sample_w = np.array([weights[l] for l in labels])
    sampler = WeightedRandomSampler(torch.from_numpy(sample_w), num_samples=len(train_idx))

    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"device: {device}  trainable params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                           patience=3, factor=0.3)

    best_acc = 0.0
    history = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, correct, tot = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            tot += x.size(0)
        train_loss = running / tot
        train_acc = correct / tot

        model.eval()
        v_run, v_c, v_t = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                v_run += criterion(out, y).item() * x.size(0)
                v_c += (out.argmax(1) == y).sum().item()
                v_t += x.size(0)
        val_loss = v_run / v_t
        val_acc = v_c / v_t
        scheduler.step(val_acc)
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                        "train_acc": round(train_acc, 4),
                        "val_loss": round(val_loss, 4), "val_acc": round(val_acc, 4)})
        print(f"epoch {epoch:2d}/{args.epochs}  train {train_acc:.4f} ({train_loss:.3f})  "
              f"val {val_acc:.4f} ({val_loss:.3f})")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model_state": model.state_dict(), "classes": CLASSES,
                        "val_acc": val_acc}, MODEL_PATH)
            print(f"  * saved model (val acc {val_acc:.4f})")

    # final test-set evaluation
    best = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(best["model_state"])
    model.eval()
    t_c, t_t = 0, 0
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            t_c += (pred == y).sum().item()
            t_t += y.size(0)
            for a, b in zip(y.cpu().numpy(), pred.cpu().numpy()):
                cm[a, b] += 1
    test_acc = t_c / t_t
    per_class = {}
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]; denom = cm[i].sum()
        per_class[c] = {"test_support": int(denom),
                        "recall": round(tp / denom, 4) if denom else 0.0}
    print(f"\ntest accuracy: {test_acc:.4f}  ({t_c}/{t_t})")
    print("confusion matrix (rows=true, cols=pred):")
    print("      " + "  ".join(f"{c:>6}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"{c:>6} " + "  ".join(f"{cm[i, j]:6d}" for j in range(len(CLASSES))))

    report = {
        "model": "resnet18",
        "classes": CLASSES,
        "total_frames": n,
        "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
        "train_class_counts": {CLASSES[i]: int(counts[i]) for i in range(len(CLASSES))},
        "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
        "best_val_acc": round(best_acc, 4),
        "test_acc": round(test_acc, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "history": history,
        "train_seconds": round(time.time() - start, 1),
        "model_path": MODEL_PATH.name,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport: {REPORT_PATH}  model: {MODEL_PATH}")


if __name__ == "__main__":
    train()