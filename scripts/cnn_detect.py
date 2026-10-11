"""Automatic activity detection with the trained ResNet-18 CNN.

Classifies every frame in a directory into write / crew / check / wait,
builds a run-length event log (start/end/duration/activity), prints a
summary and saves the event log + per-second predictions.

Usage:
    python cnn_detect.py --frames D:\\ksrtc\\frames clear
    python cnn_detect.py --frames DIR --out outputs/cnn_events.csv
    python cnn_detect.py --frame frame_00100_t3.0s.jpg   # single frame
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

ROOT = Path(r"D:\ksrtc\wpm_webapp")
CLASSES = ["write", "crew", "check", "wait"]
MODEL_PATH = ROOT / "data" / "cnn_resnet18.pt"
OUT_DIR = ROOT / "outputs"
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

VA_CLASSES = {"write"}


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    ckpt = torch.load(MODEL_PATH, map_location=device)
    if "classes" in ckpt and ckpt["classes"] != CLASSES:
        raise SystemExit(f"model classes {ckpt['classes']} != {CLASSES}")
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    return model, device


def classify_frame(model, device, path) -> str:
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    img = Image.open(path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x).argmax(1).item()
    return CLASSES[pred]


def predict_frames(model, device, frame_paths):
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    preds = []
    batch, batch_paths = [], []
    for p in frame_paths:
        x = tf(Image.open(str(p)).convert("RGB")).unsqueeze(0)
        batch.append(x); batch_paths.append(p)
        if len(batch) >= 32:
            xb = torch.cat(batch, 0).to(device)
            with torch.no_grad():
                out = model(xb).argmax(1).cpu().numpy()
            preds.extend([CLASSES[i] for i in out])
            batch, batch_paths = [], []
    if batch:
        xb = torch.cat(batch, 0).to(device)
        with torch.no_grad():
            out = model(xb).argmax(1).cpu().numpy()
        preds.extend([CLASSES[i] for i in out])
    return preds


def build_events(labels):
    events = []
    start = estart = labels[0][0]
    cur = labels[0][1]
    for sec, lab in labels[1:]:
        if lab != cur:
            events.append((estart, sec - 1, sec - estart, cur))
            estart = sec
            cur = lab
    events.append((estart, labels[-1][0], labels[-1][0] - estart + 1, cur))
    return events


def main():
    ap = argparse.ArgumentParser(description="CNN activity detection for KSRTC WPM")
    ap.add_argument("--frames", default=r"D:\ksrtc\frames clear")
    ap.add_argument("--frame", default=None, help="classify a single frame file")
    ap.add_argument("--out", default=None, help="event log CSV path")
    args = ap.parse_args()

    if args.frame:
        model, device = load_model()
        print(classify_frame(model, device, args.frame))
        return

    frames_dir = Path(args.frames)
    fps = sorted(frames_dir.glob("*.jpg"))
    if not fps:
        raise SystemExit(f"no frames in {frames_dir}")
    model, device = load_model()
    print(f"classifying {len(fps)} frames ...")
    preds = predict_frames(model, device, fps)

    labels = [(i, preds[i]) for i in range(len(preds))]
    events = build_events(labels)
    total = len(preds)

    from collections import Counter
    dist = Counter(preds)
    va_secs = sum(n for c, n in dist.items() if c in VA_CLASSES)
    nva_secs = total - va_secs
    print(f"\ndetected: {len(events)} events over {total} seconds")
    for c in CLASSES:
        print(f"  {c:>6}: {dist.get(c, 0):4d} s  ({dist.get(c, 0)/max(total,1):6.1%})")
    print(f"  VA    : {va_secs} s ({va_secs/max(total,1):6.1%})   NVA: {nva_secs} s ({nva_secs/max(total,1):6.1%})")

    csv_path = Path(args.out) if args.out else OUT_DIR / "cnn_events.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["start", "end", "duration", "activity", "VA_NVA"])
        for s, e, d, act in events:
            w.writerow([s, e, d, act, "VA" if act in VA_CLASSES else "NVA"])
    json_path = csv_path.with_suffix(".json")
    json_path.write_text(json.dumps({
        "events": [{"start": s, "end": e, "duration": d, "activity": a,
                    "va_nva": "VA" if a in VA_CLASSES else "NVA"}
                   for s, e, d, a in events],
        "activity_seconds": dict(dist), "va_secs": va_secs, "nva_secs": nva_secs,
    }, indent=2), encoding="utf-8")
    print(f"\nsaved event log: {csv_path}")
    print(f"saved predictions: {json_path}")


if __name__ == "__main__":
    main()