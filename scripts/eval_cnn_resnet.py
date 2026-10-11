"""Evaluate the saved ResNet-18 activity CNN and write cnn_report.json.

Loads data/cnn_resnet18.pt (saved best val checkpoint), evaluates on a
held-out test split of frames_labeled, prints a confusion matrix and
classification summary, and writes data/cnn_report.json.
"""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms

ROOT = Path(r"D:\ksrtc\wpm_webapp")
CLASSES = ["write", "crew", "check", "wait"]
LABELED_DIR = Path(r"D:\ksrtc\frames_labeled")
MODEL_PATH = ROOT / "data" / "cnn_resnet18.pt"
REPORT_PATH = ROOT / "data" / "cnn_report.json"
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
TEST_FRAC = 0.10
SEED = 42


class FrameDataset(Dataset):
    def __init__(self, root, classes, transform):
        self.samples = []
        self.all_labels = []
        class_to_idx = {c: i for i, c in enumerate(classes)}
        for cls in classes:
            d = root / cls
            if d.is_dir():
                for p in sorted(d.glob("*.jpg")):
                    self.samples.append((str(p), class_to_idx[cls]))
        self._labels = np.array([s[1] for s in self.samples])
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, cls = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), cls


def main():
    if not MODEL_PATH.exists():
        raise SystemExit(f"model not found: {MODEL_PATH}")
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    ds = FrameDataset(LABELED_DIR, CLASSES, tf)
    n = len(ds)
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(n)
    n_test = int(n * TEST_FRAC)
    test_idx = idx[:n_test]
    # balance test set: sample evenly per class so rare classes are shown
    per_cls = [np.where(np.array([ds.samples[i][1] for i in test_idx]) == c)[0]
               for c in range(len(CLASSES))]
    keep = np.concatenate([c[:30] for c in per_cls]) if min(len(c) for c in per_cls) >= 30 \
        else np.concatenate(per_cls)
    test_idx = test_idx[np.sort(keep)]
    test_ds = Subset(ds, test_idx.tolist())
    loader = DataLoader(test_ds, batch_size=24, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_pred.extend(model(x).argmax(1).cpu().numpy().tolist())
            y_true.extend(y.cpu().numpy().tolist())
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    acc = float((y_true == y_pred).mean())
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for a, b in zip(y_true, y_pred):
        cm[a, b] += 1

    print(f"test samples: {len(y_true)}  accuracy: {acc:.4f}")
    print("confusion matrix (rows=true, cols=pred):")
    print("      " + "  ".join(f"{c:>7}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"{c:>7} " + "  ".join(f"{cm[i, j]:7d}" for j in range(len(CLASSES))))

    per_class = {}
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]; sup = cm[i].sum(); pred = cm[:, i].sum()
        per_class[c] = {
            "test_support": int(sup),
            "precision": round(tp / pred, 4) if pred else 0.0,
            "recall": round(tp / sup, 4) if sup else 0.0,
        }

    report = {
        "model": "resnet18",
        "classes": CLASSES,
        "total_frames": n,
        "test_frames": int(len(y_true)),
        "test_accuracy": round(acc, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "model_path": MODEL_PATH.name,
        "note": "weights: best val checkpoint from train_cnn_resnet.py; "
                "test split balanced to 30/class",
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten: {REPORT_PATH}")


if __name__ == "__main__":
    main()