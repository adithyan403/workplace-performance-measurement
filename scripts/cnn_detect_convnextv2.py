"""Activity detection with the trained ConvNeXt V2-Tiny CNN (comparison arm).

Classifies every frame in a directory into write / crew / check / wait,
builds a run-length event log (start/end/duration/activity), prints a
summary and saves the event log + per-second predictions. Mirrors
scripts/cnn_detect.py so the ConvNeXt V2-Tiny and ResNet-18 arms are
directly comparable.

Usage:
    python cnn_detect_convnextv2.py --frames "D:\\ksrtc\\frames clear"
    python cnn_detect_convnextv2.py --frames DIR --out outputs/cnn_events_convnextv2.csv
    python cnn_detect_convnextv2.py --frame frame_00100_t0100.00s.jpg
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(r"D:\ksrtc\wpm_webapp")
CLASSES = ["write", "crew", "check", "wait"]
MODEL_PATH = ROOT / "data" / "cnn_convnextv2_tiny.pt"
OUT_DIR = ROOT / "outputs"
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
VA_CLASSES = {"write"}


def build_model(arch, n_classes):
    import timm
    return timm.create_model(arch, pretrained=False, num_classes=n_classes)


def load_model():
    import timm  # noqa: F401  (import check before torch.load)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(MODEL_PATH, map_location=device)
    arch = ckpt.get("arch", "convnextv2_tiny.fcmae_ft_in22k_in1k")
    classes = ckpt.get("classes", CLASSES)
    if classes != CLASSES:
        raise SystemExit(f"model classes {classes} != {CLASSES}")
    model = timm.create_model(arch, pretrained=False, num_classes=len(CLASSES))
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    return model, device


def make_tf():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def classify_frame(model, device, path):
    tf = make_tf()
    x = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        return CLASSES[model(x).argmax(1).item()]


def predict_frames(model, device, frame_paths, batch_size=16):
    tf = make_tf()
    preds, batch = [], []
    for p in frame_paths:
        batch.append(tf(Image.open(str(p)).convert("RGB")).unsqueeze(0))
        if len(batch) >= batch_size:
            with torch.no_grad():
                out = model(torch.cat(batch, 0).to(device)).argmax(1).cpu().numpy()
            preds.extend(CLASSES[i] for i in out)
            batch = []
    if batch:
        with torch.no_grad():
            out = model(torch.cat(batch, 0).to(device)).argmax(1).cpu().numpy()
        preds.extend(CLASSES[i] for i in out)
    return preds


def build_events(labels):
    events = []
    estart, cur = labels[0][0], labels[0][1]
    for sec, lab in labels[1:]:
        if lab != cur:
            events.append((estart, sec - 1, sec - estart, cur))
            estart, cur = sec, lab
    events.append((estart, labels[-1][0], labels[-1][0] - estart + 1, cur))
    return events


def main():
    ap = argparse.ArgumentParser(description="ConvNeXt V2-Tiny activity detection (KSRTC WPM)")
    ap.add_argument("--frames", default=r"D:\ksrtc\frames clear")
    ap.add_argument("--frame", default=None, help="classify a single frame file")
    ap.add_argument("--out", default=None, help="event log CSV path")
    ap.add_argument("--batch", type=int, default=16)
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
    print(f"classifying {len(fps)} frames with ConvNeXt V2-Tiny on {device} ...")
    preds = predict_frames(model, device, fps, args.batch)

    labels = list(enumerate(preds))
    events = build_events(labels)
    total = len(preds)
    dist = Counter(preds)
    va_secs = sum(n for c, n in dist.items() if c in VA_CLASSES)
    nva_secs = total - va_secs
    print(f"\ndetected: {len(events)} events over {total} seconds")
    for c in CLASSES:
        print(f"  {c:>6}: {dist.get(c, 0):4d} s  ({dist.get(c, 0)/max(total,1):6.1%})")
    print(f"  VA    : {va_secs} s ({va_secs/max(total,1):6.1%})   NVA: {nva_secs} s ({nva_secs/max(total,1):6.1%})")

    csv_path = Path(args.out) if args.out else OUT_DIR / "cnn_events_convnextv2.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["start", "end", "duration", "activity", "VA_NVA"])
        for s, e, d, act in events:
            w.writerow([s, e, d, act, "VA" if act in VA_CLASSES else "NVA"])
    json_path = csv_path.with_suffix(".json")
    json_path.write_text(json.dumps({
        "model": "convnextv2_tiny",
        "arch": "convnextv2_tiny.fcmae_ft_in22k_in1k",
        "frames_dir": str(frames_dir),
        "n_frames": total,
        "events": [{"start": s, "end": e, "duration": d, "activity": a,
                    "va_nva": "VA" if a in VA_CLASSES else "NVA"}
                   for s, e, d, a in events],
        "activity_seconds": dict(dist), "va_secs": va_secs, "nva_secs": nva_secs,
    }, indent=2), encoding="utf-8")
    print(f"\nsaved event log: {csv_path}")
    print(f"saved predictions: {json_path}")


if __name__ == "__main__":
    main()
