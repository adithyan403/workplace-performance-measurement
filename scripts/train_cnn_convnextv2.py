"""Train a ConvNeXt V2-Tiny CNN to detect KSRTC WPM activities.

Reads labeled frames from D:\\ksrtc\\frames_labeled\\<class>\\*.jpg
(classes: write, crew, check, wait) and transfer-learns from the
timm "convnextv2_tiny.fcmae_ft_in22k_in1k" checkpoint, with class
weighting for the rare "crew" class.

Saves:
    data/cnn_convnextv2_tiny.pt     best val checkpoint
    data/cnn_convnextv2_report.json metrics + confusion matrix + history
    data/cnn_convnextv2_history.csv per-epoch log

Usage:
    python train_cnn_convnextv2.py
    python train_cnn_convnextv2.py --epochs 15 --batch 24 --lr 3e-4
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import transforms

ROOT = Path(r"D:\ksrtc\wpm_webapp")
CLASSES = ["write", "crew", "check", "wait"]
LABELED_DIR = Path(r"D:\ksrtc\frames_labeled")
MODEL_DIR = ROOT / "data"
MODEL_PATH = MODEL_DIR / "cnn_convnextv2_tiny.pt"
REPORT_PATH = MODEL_DIR / "cnn_convnextv2_report.json"
HISTORY_PATH = MODEL_DIR / "cnn_convnextv2_history.csv"

ARCH = "convnextv2_tiny.fcmae_ft_in22k_in1k"
MODEL_NAME = "convnextv2_tiny"

# ConvNeXt V2 (timm fcmae ft) normalisation
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
        if not self.samples:
            raise RuntimeError(f"no labeled frames found under {root}")
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, cls = self.samples[i]
        from PIL import Image
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, cls


def build_model(dropout=0.2):
    import timm

    model = timm.create_model(ARCH, pretrained=True, num_classes=len(CLASSES),
                              drop_rate=dropout)
    return model


def main():
    ap = argparse.ArgumentParser(description="Train ConvNeXt V2-Tiny activity CNN")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data", type=Path, default=LABELED_DIR,
                    help="labeled-frames root (<class>/*.jpg)")
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

    ds = FrameDataset(args.data, CLASSES, transform=train_tf)
    n = len(ds)
    print(f"labeled frames: {n}  classes: {CLASSES}")
    print(f"arch: {ARCH}")

    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(n)
    n_test = int(n * args.test_frac)
    n_val = int(n * args.val_frac)
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    print(f"splits: train {len(train_idx)}  val {len(val_idx)}  test {len(test_idx)}")

    train_ds = Subset(ds, train_idx.tolist())
    val_ds = Subset(ds, val_idx.tolist())
    test_ds = Subset(ds, test_idx.tolist())

    labels = [ds.samples[i][1] for i in train_idx]
    counts = np.bincount(labels, minlength=len(CLASSES))
    weights = 1.0 / (counts.astype(float) + 1e-6)
    sample_w = np.array([weights[l] for l in labels])
    sampler = WeightedRandomSampler(torch.from_numpy(sample_w), num_samples=len(train_idx))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"
    if pin:
        torch.backends.cudnn.benchmark = True

    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                              num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=0, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                             num_workers=0, pin_memory=pin)

    model = build_model()
    model.to(device)
    nparams = sum(p.numel() for p in model.parameters())
    ntrained = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"device: {device}  params: {nparams:,}  trainable: {ntrained:,}")

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
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
        train_loss, train_acc = running / tot, correct / tot

        model.eval()
        v_run, v_c, v_t = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                v_run += criterion(out, y).item() * x.size(0)
                v_c += (out.argmax(1) == y).sum().item()
                v_t += x.size(0)
        val_loss, val_acc = v_run / v_t, v_c / v_t
        scheduler.step(val_acc)
        el = time.time() - start
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                        "train_acc": round(train_acc, 4), "val_loss": round(val_loss, 4),
                        "val_acc": round(val_acc, 4), "seconds": round(el, 1)})
        print(f"epoch {epoch:2d}/{args.epochs}  train {train_acc:.4f} ({train_loss:.3f})  "
              f"val {val_acc:.4f} ({val_loss:.3f})  [{el/60:.1f} min]", flush=True)
        if val_acc > best_acc:
            best_acc = val_acc
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": {k: (v.half() if v.is_floating_point() else v)
                                        for k, v in model.state_dict().items()},
                        "classes": CLASSES,
                        "arch": ARCH, "model_name": MODEL_NAME, "val_acc": val_acc},
                       MODEL_PATH)
            print(f"  * saved model (val acc {val_acc:.4f})", flush=True)

    best = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict({k: (v.to(torch.float32) if v.is_floating_point() else v)
                           for k, v in best["model_state"].items()})
    model.eval()
    t_c, t_t = 0, 0
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    preds_all = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            t_c += (pred == y).sum().item()
            t_t += y.size(0)
            preds_all.extend(pred.cpu().numpy().tolist())
            for a, b in zip(y.cpu().numpy(), pred.cpu().numpy()):
                cm[a, b] += 1
    test_acc = t_c / t_t
    per_class = {}
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]
        sup = cm[i].sum()
        pred_n = cm[:, i].sum()
        per_class[c] = {"test_support": int(sup),
                        "precision": round(tp / pred_n, 4) if pred_n else 0.0,
                        "recall": round(tp / sup, 4) if sup else 0.0}
    print(f"\ntest accuracy: {test_acc:.4f}  ({t_c}/{t_t})")
    print("confusion matrix (rows=true, cols=pred):")
    print("      " + "  ".join(f"{c:>6}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"{c:>6} " + "  ".join(f"{cm[i, j]:6d}" for j in range(len(CLASSES))))

    report = {
        "model": MODEL_NAME,
        "arch": ARCH,
        "framework": f"timm + torch {torch.__version__}",
        "classes": CLASSES,
        "total_frames": n,
        "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
        "train_class_counts": {CLASSES[i]: int(counts[i]) for i in range(len(CLASSES))},
        "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
        "params": nparams,
        "best_val_acc": round(best_acc, 4),
        "test_acc": round(test_acc, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "history": history,
        "train_seconds": round(time.time() - start, 1),
        "model_path": MODEL_PATH.name,
        "note": "transfer learning from ImageNet-22k FCMAE fine-tuned ConvNeXt V2-Tiny",
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with open(HISTORY_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc",
                                          "val_loss", "val_acc", "seconds"])
        w.writeheader()
        w.writerows(history)
    print(f"\nreport: {REPORT_PATH}\nmodel: {MODEL_PATH}\nhistory: {HISTORY_PATH}")


if __name__ == "__main__":
    main()
