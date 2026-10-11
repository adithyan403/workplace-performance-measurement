"""CNN deep-learning detection module for the WPM webapp.

Loads data/cnn_resnet18.pt (ResNet-18, 4 classes) and provides:
  * classify_paths()  -> per-frame predictions for a list of image files
  * predictions_to_events() -> run-length event log from predictions
  * find_va_nva()     -> VA/NVA summary                  (shared w/ automation)

All heavy imports are lazy so the webapp still boots even when torch/PIL are
not installed (the /ml/cnn page then explains what to install). Within methods
we always reference the module-locals _torch/_nn/_models/_transforms/_pil so the
class remains import-safe.
"""
from pathlib import Path

# lazy heavy imports (filled by _import())
_torch = None
_models = None
_nn = None
_transforms = None
_pil = None


def _import():
    global _torch, _models, _nn, _transforms, _pil
    if _torch is None:
        import torch
        from torch import nn
        from torchvision import models, transforms
        from PIL import Image
        _torch, _nn, _models, _transforms, _pil = torch, nn, models, transforms, Image


def available():
    try:
        _import()
        return True
    except Exception:
        return False


CLASSES = ["write", "crew", "check", "wait"]
VA_CLASSES = {"write"}
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class CnnDetector:
    """ResNet-18 activity classifier over WPM frame images."""

    def __init__(self, model_path, device=None):
        _import()
        torch, nn, models = _torch, _nn, _models
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
        ckpt = torch.load(model_path, map_location=self.device)
        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
        elif "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        else:
            model.load_state_dict(ckpt)
        model.eval().to(self.device)
        self.model = model
        self.tf = _transforms.Compose([
            _transforms.Resize((IMG_SIZE, IMG_SIZE)),
            _transforms.ToTensor(),
            _transforms.Normalize(MEAN, STD),
        ])

    def classify_paths(self, paths, batch_size=32):
        """Return list of predicted class names, same order as paths."""
        torch, pil = _torch, _pil
        preds = []
        batch = []
        for i, p in enumerate(paths):
            img = pil.open(str(p)).convert("RGB")
            batch.append(self.tf(img).unsqueeze(0))
            if len(batch) >= batch_size or i == len(paths) - 1:
                xb = torch.cat(batch, 0).to(self.device)
                with torch.no_grad():
                    out = self.model(xb).argmax(1).cpu().numpy().tolist()
                preds.extend(CLASSES[j] for j in out)
                batch = []
        return preds

    def classify_path(self, path):
        return self.classify_paths([path])[0]


def predictions_to_events(preds):
    """Flatten a per-frame prediction list into (start, end, duration, activity)."""
    if not preds:
        return []
    events = []
    start = 0
    cur = preds[0]
    for i in range(1, len(preds) + 1):
        if i == len(preds) or preds[i] != cur:
            events.append({
                "start": start, "end": i - 1, "duration": i - start,
                "activity": cur,
                "va_nva": "VA" if cur in VA_CLASSES else "NVA",
            })
            if i < len(preds):
                start = i
                cur = preds[i]
    return events


def find_va_nva(events):
    """Aggregate VA/NVA seconds across an event list — mirrors automation. """
    tot = sum(e["duration"] for e in events)
    va = sum(e["duration"] for e in events if e["activity"] in VA_CLASSES)
    nva = tot - va
    return {
        "total": tot,
        "VA": {"sec": va, "pct": round(100 * va / tot, 2) if tot else 0},
        "NVA": {"sec": nva, "pct": round(100 * nva / tot, 2) if tot else 0},
    }


def extract_upload(zip_path, dest_dir):
    """Unpack an uploaded .zip of frames into dest_dir; returns file count."""
    import zipfile
    count = 0
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                info = z.getinfo(name)
                target = dest_dir / Path(name).name
                with z.open(name) as src, open(target, "wb") as dst:
                    dst.write(src.read())
                count += 1
    return count


def collect_images(src_dir):
    """Sorted list of .jpg/.png images under src_dir (flat)."""
    src = Path(src_dir)
    return [str(p) for p in sorted(
        [p for p in src.glob("*.jpg")] +
        [p for p in src.glob("*.jpeg")] +
        [p for p in src.glob("*.png")])]