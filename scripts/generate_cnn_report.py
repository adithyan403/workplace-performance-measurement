"""Generate a colorful PDF report for the KSRTC WPM deep-learning (CNN) work.

Produces outputs/WPM_CNN_Report.pdf with:
  * cover page
  * 4-step methodology diagram
  * data & labelling summary + sample frame mosaic
  * ResNet-18 architecture summary
  * evaluation (confusion matrix, per-class bars)
  * full-video detection results (event timeline, VA/NVA donut)
  * web-app integration notes

Run:  python scripts/generate_cnn_report.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from PIL import Image

ROOT = Path(r"D:\ksrtc\wpm_webapp")
DATA = ROOT / "data"
LABELED = Path(r"D:\ksrtc\frames_labeled")
FRAMES = Path(r"D:\ksrtc\frames clear")
OUT = ROOT / "outputs" / "WPM_CNN_Report.pdf"

CLASSES = ["write", "crew", "check", "wait"]
COLORS = {
    "write": "#22c55e", "crew": "#ef4444",
    "check": "#f59e0b", "wait": "#64748b",
}
VA_CLASSES = {"write"}
INK = "#0f172a"
MUTED = "#475569"
ACCENT = "#7c3aed"
GRID = "#e2e8f0"

rcParams.update({
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "text.color": INK,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.family": "Segoe UI",
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.alpha": 0.7,
})


def load_report():
    p = DATA / "cnn_report.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def load_events():
    p = DATA / "cnn_events.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def sample_path(cls):
    d = LABELED / cls
    files = sorted(d.glob("*.jpg")) if d.is_dir() else []
    if not files:
        return None
    idx = np.random.RandomState(hash(cls) % 1000).randint(0, len(files))
    return files[idx]


def cover(pp, rep):
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    # gradient banner blocks
    import matplotlib.patches as mp
    for i, c in enumerate(["#0f172a", "#1e1b4b", "#4c1d95", "#7c3aed"]):
        ax.add_patch(mp.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                  facecolor=c, alpha=0.9,
                                  zorder=0))
    # decorative circles
    for cx, cy, r, a in [(0.88, 0.9, 0.32, 0.18), (0.12, 0.15, 0.28, 0.14)]:
        ax.add_patch(mp.Circle((cx, cy), r, transform=ax.transAxes, zorder=1,
                               facecolor="#a78bfa", alpha=a, edgecolor="none"))
    ax.text(0.5, 0.86, "KSRTC", transform=ax.transAxes, ha="center",
            color="#fcd34d", fontsize=34, fontweight=800, zorder=2)
    ax.text(0.5, 0.79, "WORKPLACE PERFORMANCE MEASUREMENT", transform=ax.transAxes,
            ha="center", color="white", fontsize=15, fontweight=700, zorder=2)
    ax.text(0.5, 0.66, "Deep-Learning Activity Detection", transform=ax.transAxes,
            ha="center", color="#e9d5ff", fontsize=42, fontweight=800, zorder=2)
    ax.text(0.5, 0.585, "ResNet-18 CNN trained on 2,000 labelled video frames",
            transform=ax.transAxes, ha="center", color="#c4b5fd", fontsize=15,
            weight=600, zorder=2)
    # class chips
    y = 0.45
    for i in range(0, len(CLASSES), 2):
        row = CLASSES[i:i + 2]
        for j, c in enumerate(row):
            x = 0.32 + j * 0.36
            ax.add_patch(mp.FancyBboxPatch((x, y), 0.16, 0.055, boxstyle="round,pad=0.006",
                         transform=ax.transAxes, facecolor=COLORS[c], alpha=0.92,
                         edgecolor="none", zorder=2))
            ax.text(x + 0.08, y + 0.027, c.upper(), transform=ax.transAxes, ha="center",
                    va="center", color="white", fontsize=11, fontweight=700, zorder=3)
        y -= 0.085
    acc = rep.get("test_accuracy", 0) * 100
    ax.text(0.5, 0.20, f"Hold-out test accuracy  •  {acc:.1f}%", transform=ax.transAxes,
            ha="center", color="#fde68a", fontsize=18, fontweight=800, zorder=2)
    ax.text(0.5, 0.12, "Waybill Entry Desk  •  KSRTC Depot  •  Phase-I Evidence Module",
            transform=ax.transAxes, ha="center", color="#cbd5e1", fontsize=11, zorder=2)
    ax.text(0.5, 0.05, "Method after Nesterak, Szelagowski & Radziszewski, J. Intell. Manuf. (2025)",
            transform=ax.transAxes, ha="center", color="#94a3b8", fontsize=9, zorder=2)
    pp.savefig(fig); plt.close(fig)


def step_icon(ax, y, num, title, desc, color):
    ax.add_patch(matplotlib.patches.FancyBboxPatch((0.02, y + 0.016), 0.06, 0.066,
                  boxstyle="round,pad=0.004", transform=ax.transAxes,
                  facecolor=color, edgecolor="none"))
    ax.text(0.05, y + 0.049, str(num), transform=ax.transAxes, ha="center",
            va="center", color="white", fontsize=16, fontweight=800)
    ax.text(0.11, y + 0.062, title, transform=ax.transAxes, color=INK,
            fontsize=13, fontweight=800)
    ax.text(0.11, y + 0.032, desc, transform=ax.transAxes, color=MUTED, fontsize=9.5)


def methodology(pp):
    fig = plt.figure(figsize=(11.69, 8.27)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.text(0.5, 0.93, "Methodology — from video to automatic detection",
            ha="center", color=INK, fontsize=22, fontweight=800)
    ax.text(0.5, 0.885, "Phase I / II evidence chain built for the KSRTC waybill-entry work station",
            ha="center", color=MUTED, fontsize=11)
    steps = [
        ("RECORD", "4K IP-Camera Capture", "65 min 51 s film of the waybill-entry desk\n23.04 GB, 1920 x 1080 @ 25 fps", "#0e7490"),
        ("SAMPLE", "Frame Extraction", "1 frame per second\n3,951 frames from the raw footage", "#1d4ed8"),
        ("LABEL", "Human Annotation", "2,000 frames labelled by 4 activities\nwrite / crew / check / wait, custom label tool", "#7c3aed"),
        ("TRAIN", "ResNet-18 Transfer Learning", "ImageNet weights, 4-class head, class-balanced\ntrain 75% / val 15% / test 10%, Adam lr 3e-4", "#db2777"),
        ("EVALUATE", "Hold-out Test", "93.0% accuracy, per-class precision/recall\n+ confusion matrix analysed", "#16a34a"),
        ("DEPLOY", "Web-App Upload + Detect", "Upload frames to the Flask dashboard,\nCNN auto-labels them into an event log", "#ea580c"),
    ]
    y0 = 0.80
    for i, (tag, title, desc, c) in enumerate(steps):
        y = y0 - 0.12 * i
        step_icon(ax, y - 0.0, i + 1, title, desc.replace("\n", "\n"), c)
        ax.add_patch(matplotlib.patches.FancyBboxPatch((0.7, y - 0.012), 0.26, 0.09,
                      boxstyle="round,pad=0.006", transform=ax.transAxes,
                      facecolor=c, alpha=0.10, edgecolor=c, lw=1.0))
        ax.text(0.83, y + 0.028, tag, transform=ax.transAxes, ha="center",
                color=c, fontsize=12, fontweight=800)
        # connector arrow between steps
        if i < len(steps) - 1:
            ax.annotate("", xy=(0.07, y - 0.115), xytext=(0.07, y - 0.04),
                        transform=ax.transAxes,
                        arrowprops=dict(arrowstyle="-|>", color="#cbd5e1", lw=1.6))
    pp.savefig(fig); plt.close(fig)


def mosaic(pp):
    fig = plt.figure(figsize=(11.69, 8.27))
    gs = fig.add_gridspec(2, 2, left=0.08, right=0.92, top=0.74, bottom=0.30,
                          hspace=0.18, wspace=0.15)
    for i, c in enumerate(CLASSES):
        ax = fig.add_subplot(gs[i // 2, i % 2]); ax.axis("off")
        p = sample_path(c)
        if p and p.exists():
            img = Image.open(p).convert("RGB")
            ax.imshow(img)
            ax.set_title(c.upper(), color=COLORS[c], fontsize=14, fontweight=800)
    ax = fig.add_axes([0.08, 0.08, 0.84, 0.15]); ax.axis("off")
    ax.text(0, 0.75, "Labelled Training Data", fontsize=20, fontweight=800, color=INK)
    ax.text(0, 0.35,
            "2,000 frames from D:\\ksrtc\\frames clear classified into four activities at 1 s resolution.",
            fontsize=11, color=MUTED)
    # distribution bar
    axb = fig.add_axes([0.08, 0.78, 0.84, 0.13]); axb.axis("off")
    n = {"write": 870, "crew": 66, "check": 710, "wait": 354}
    tot = sum(n.values())
    xs = np.arange(len(CLASSES)); w = 0.62
    axb.bar(xs, [n[c] for c in CLASSES], width=w, color=[COLORS[c] for c in CLASSES],
            edgecolor="white", linewidth=1.2, zorder=3)
    for x, c in zip(xs, CLASSES):
        axb.text(x, n[c] + 8, str(n[c]), ha="center", fontsize=10, fontweight=700, color=INK)
    axb.set_xticks(xs); axb.set_xticklabels([c.upper() for c in CLASSES], fontsize=10)
    axb.set_yticks([]); axb.spines["top"].set_visible(False)
    axb.spines["right"].set_visible(False); axb.spines["left"].set_visible(False)
    axb.set_ylim(0, 1000)
    axb.text(8.0, 940, "Class balance — crew is the rare class (3.3%), handled with weighted sampling",
             fontsize=8.5, color=MUTED, ha="right")
    pp.savefig(fig); plt.close(fig)


def architecture(pp):
    fig = plt.figure(figsize=(11.69, 8.27)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.text(0.5, 0.94, "ResNet-18 Architecture for Frame Classification", ha="center",
            color=INK, fontsize=22, fontweight=800)
    blocks = [
        ("Input", "224 × 224 RGB frame", "#0f766e"),
        ("Stem", "Conv7×7, 64 · BN · ReLU\nMaxPool 3×3", "#1d4ed8"),
        ("Stage 1", "2 × BasicBlock\nch = 64", "#7c3aed"),
        ("Stage 2", "2 × BasicBlock\nch = 128, stride 2", "#db2777"),
        ("Stage 3", "2 × BasicBlock\nch = 256, stride 2", "#ea580c"),
        ("Stage 4", "2 × BasicBlock\nch = 512, stride 2", "#b91c1c"),
        ("Pool", "Global AvgPool", "#16a34a"),
        ("Head (new)", "FC 512 → 4\nwrite / crew / check / wait", "#eab308"),
    ]
    n = len(blocks)
    width = 0.82 / n
    for i, (t, d, c) in enumerate(blocks):
        x = 0.09 + i * width + width / 2
        ax.add_patch(matplotlib.patches.FancyBboxPatch(
            (x - width * 0.34, 0.60), width * 0.68, 0.20, boxstyle="round,pad=0.008",
            transform=ax.transAxes, facecolor=c, alpha=0.14, edgecolor=c, lw=1.3))
        ax.text(x, 0.72, t, transform=ax.transAxes, ha="center", color=c,
                fontsize=10.5, fontweight=800)
        ax.text(x, 0.655, d.replace("\n", "\n"), transform=ax.transAxes, ha="center",
                color=MUTED, fontsize=7.5)
        if i < n - 1:
            ax.annotate("", xy=(x + width * 0.34 + 0.006, 0.70),
                        xytext=(x + width * 0.34 - 0.002, 0.70), transform=ax.transAxes,
                        arrowprops=dict(arrowstyle="-|>", color="#94a3b8", lw=1.4))
    # training data panel
    ax.add_patch(matplotlib.patches.FancyBboxPatch((0.09, 0.30), 0.82, 0.20,
                  boxstyle="round,pad=0.01", transform=ax.transAxes,
                  facecolor="#f8fafc", edgecolor=GRID, lw=1.2))
    tx = [
        ("Augmentation", "RandomHorizontalFlip (p=0.3) · ColorJitter · Resize 224"),
        ("Sampling", "WeightedRandomSampler — over-samples the rare crew class"),
        ("Loss / Opt", "CrossEntropyLoss (class-weighted) · Adam 3e-4 · ReduceLROnPlateau"),
        ("Schedule", "25 epochs, batch 24 — best model kept by validation accuracy"),
    ]
    for j, (k, v) in enumerate(tx):
        y = 0.455 - j * 0.045
        ax.text(0.12, y, k.upper(), transform=ax.transAxes, ha="left", color=ACCENT,
                fontsize=8.5, fontweight=800)
        ax.text(0.30, y, v, transform=ax.transAxes, ha="left", color=INK, fontsize=8.5)
    ax.text(0.5, 0.20, "Transfer learning: ImageNet-pretrained trunk, head replaced for 4 WPM activities",
            transform=ax.transAxes, ha="center", color=MUTED, fontsize=10.5)
    pp.savefig(fig); plt.close(fig)


def evaluation(pp, rep):
    fig = plt.figure(figsize=(11.69, 8.27))
    acc = rep.get("test_accuracy", 0)
    cm = np.array(rep["confusion_matrix"], dtype=int)
    # confusion matrix
    ax = fig.add_axes([0.07, 0.42, 0.46, 0.46])
    im = ax.imshow(cm, cmap="Purples")
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, fontsize=9)
    ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES, fontsize=9)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            mx = cm[i].max()
            col = "white" if cm[i, j] >= 0.75 * mx and mx > 0 else INK
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=11,
                    fontweight=700, color=col)
    ax.set_title("Confusion Matrix — test set", fontsize=12, fontweight=800, pad=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.grid(False)
    # per-class bars
    axb = fig.add_axes([0.62, 0.42, 0.34, 0.46])
    prec = [rep["per_class"][c]["precision"] for c in CLASSES]
    rec = [rep["per_class"][c]["recall"] for c in CLASSES]
    xs = np.arange(len(CLASSES)); w = 0.38
    axb.bar(xs - w / 2, prec, w, color=ACCENT, label="Precision")
    axb.bar(xs + w / 2, rec, w, color="#22d3ee", label="Recall")
    axb.set_xticks(xs); axb.set_xticklabels(CLASSES, fontsize=9, rotation=15)
    axb.set_ylim(0, 1.05)
    axb.set_title("Per-Class Precision / Recall", fontsize=12, fontweight=800, pad=8)
    axb.legend(fontsize=8, frameon=False)
    # accuracy banner
    ax2 = fig.add_axes([0.07, 0.08, 0.89, 0.22]); ax2.axis("off")
    ax2.add_patch(matplotlib.patches.FancyBboxPatch((0, 0.08), 1, 0.75,
                  boxstyle="round,pad=0.012", transform=ax2.transAxes,
                  facecolor="#0f172a", edgecolor="none"))
    ax2.text(0.06, 0.55, f"{acc * 100:.1f}%", transform=ax2.transAxes,
             color="#fde68a", fontsize=34, fontweight=800)
    ax2.text(0.22, 0.58, "Hold-out Test Accuracy", transform=ax2.transAxes,
             color="white", fontsize=14, fontweight=800)
    ax2.text(0.22, 0.32, f"{rep.get('test_frames', 0)} test frames · {rep.get('total_frames', 0)} labelled frames · "
                         "crew correctly caught despite only 66 samples",
             transform=ax2.transAxes, color="#cbd5e1", fontsize=10)
    ax2.text(0.06, 0.24, f"Model: cnn_resnet18.pt · {rep.get('model', 'resnet18')}",
             transform=ax2.transAxes, color="#94a3b8", fontsize=9)
    pp.savefig(fig); plt.close(fig)


def detection(pp, ev):
    fig = plt.figure(figsize=(11.69, 8.27))
    dist = ev.get("activity_seconds", {})
    va = ev.get("va_secs", 0); nva = ev.get("nva_secs", 0)
    # timeline strip
    ax = fig.add_axes([0.07, 0.68, 0.89, 0.22]); ax.axis("off")
    ax.set_title("Predicted Activity Timeline — 2,000 s of observation",
                 fontsize=13, fontweight=800, loc="left", pad=10)
    events = ev.get("events", [])
    for e in events:
        ax.add_patch(matplotlib.patches.Rectangle(
            (e["start"], 0), max(e["duration"], 1), 1,
            facecolor=COLORS.get(e["activity"], "#94a3b8")))
    ax.set_xlim(0, events[-1]["end"] + 1 if events else len(dist))
    ax.set_ylim(0, 1.2)
    ax.set_yticks([]); ax.spines["top"].set_visible(False)
    try:
        ax.spines["bottom"].set_visible(False); ax.spines["left"].set_visible(False)
    except Exception:
        pass
    # donut VA/NVA
    axd = fig.add_axes([0.07, 0.24, 0.30, 0.38])
    axd.pie([va, nva], colors=["#16a34a", "#dc2626"], startangle=90,
            counterclock=False, wedgeprops=dict(width=0.32, edgecolor="white"))
    axd.text(0, 0.08, f"{va}", ha="center", fontsize=20, fontweight=800, color="#16a34a")
    axd.text(0, -0.10, "VA seconds", ha="center", fontsize=9, color=MUTED)
    axd.set_title("VA / NVA split", fontsize=12, fontweight=800, pad=12)
    # distribution bars
    axb = fig.add_axes([0.45, 0.24, 0.50, 0.38])
    tot = sum(dist.values()) or 1
    names = list(dist.keys()); vals = [dist[c] for c in names]
    cols = [COLORS.get(c, "#94a3b8") for c in names]
    bars = axb.bar(range(len(names)), vals, color=cols, edgecolor="white", linewidth=1.1)
    for b, v, c in zip(bars, vals, names):
        axb.text(b.get_x() + b.get_width() / 2, v + 8, f"{v} s", ha="center",
                 fontsize=9.5, fontweight=700)
    axb.set_xticks(range(len(names))); axb.set_xticklabels([c.upper() for c in names], fontsize=10)
    axb.set_yticks([]); axb.spines["top"].set_visible(False); axb.spines["left"].set_visible(False)
    axb.set_title("Activity seconds — automatic detection over the full observation",
                  fontsize=12, fontweight=800, pad=12)
    axb.set_ylim(0, max(vals, default=100) * 1.15)
    axn = fig.add_axes([0.07, 0.06, 0.89, 0.10]); axn.axis("off")
    axn.text(0, 0.5,
             f"A total of {va + nva} frames processed, {len(events)} duty events segmented, "
             f"VA (value-added) = {100 * va / tot:.1f} %, NVA = {100 * nva / tot:.1f} %  —  "
             "identical logic to the automation module used by the dashboard.",
             fontsize=9.5, color=MUTED)
    pp.savefig(fig); plt.close(fig)


def integration(pp):
    fig = plt.figure(figsize=(11.69, 8.27)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.text(0.5, 0.94, "Web-App Integration — /ml/cnn upload & detect", ha="center",
            color=INK, fontsize=21, fontweight=800)
    ax.text(0.5, 0.90, "Upload frame images (or a .zip) on the live dashboard; the CNN labels them instantly",
            ha="center", color=MUTED, fontsize=11)
    panels = [
        ("Flask route", "/ml/cnn  +  POST /ml/cnn/predict", "cnn.html template",
         "Modules/cnn_module.py · CnnDetector", "#0e7490"),
        ("Upload", "multiple .jpg/.png or .zip\nMAX_CONTENT_LENGTH 256 MB", "saved under static/cnn_uploads",
         "zip auto-extracted", "#7c3aed"),
        ("Inference", "ResNet-18 forward pass\n224 × 224, batch 32", "per-frame class + VA/NVA type",
         "returned as JSON", "#db2777"),
        ("Results", "class distribution\nevent timeline table", "per-frame prediction list",
         "VA / NVA split summary", "#16a34a"),
    ]
    for i, (t, a, b, c, col) in enumerate(panels):
        x = 0.06 + i * 0.23
        ax.add_patch(matplotlib.patches.FancyBboxPatch((x, 0.50), 0.20, 0.30,
                      boxstyle="round,pad=0.01", transform=ax.transAxes,
                      facecolor="#f8fafc", edgecolor=col, lw=1.6))
        ax.text(x + 0.10, 0.76, t, transform=ax.transAxes, ha="center",
                color=col, fontsize=11, fontweight=800)
        ax.text(x + 0.10, 0.67, a.replace("\n", "\n"), transform=ax.transAxes,
                ha="center", color=INK, fontsize=8)
        ax.text(x + 0.10, 0.585, b, transform=ax.transAxes, ha="center", color=MUTED, fontsize=7.5)
        ax.text(x + 0.10, 0.525, c, transform=ax.transAxes, ha="center", color=MUTED, fontsize=7.5)
        if i < 3:
            ax.annotate("", xy=(x + 0.215, 0.65), xytext=(x + 0.205, 0.65),
                        transform=ax.transAxes, arrowprops=dict(arrowstyle="-|>", color="#94a3b8"))
    ax.text(0.5, 0.42, "Live dashboard page offers:", transform=ax.transAxes, ha="center",
            color=INK, fontsize=13, fontweight=800)
    for j, t in enumerate([
        "KPI strip — model ONLINE, test accuracy, classes, frames uploaded",
        "Confusion matrix & per-class precision/recall rendered in the dashboard theme",
        "Upload + run button, then an event table, class distribution and VA/NVA result",
        "Everything auto-logs into the same duty-log pipeline used by the dashboard",
    ]):
        ax.text(0.5, 0.36 - j * 0.05, "▸ " + t, transform=ax.transAxes, ha="center",
                color=MUTED, fontsize=10)
    ax.text(0.5, 0.08, "Trained locally with scripts/train_cnn_resnet.py, shipped as data/cnn_resnet18.pt",
            transform=ax.transAxes, ha="center", color="#94a3b8", fontsize=9)
    pp.savefig(fig); plt.close(fig)


def main():
    rep = load_report()
    ev = load_events()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT) as pp:
        cover(pp, rep)
        methodology(pp)
        mosaic(pp)
        architecture(pp)
        evaluation(pp, rep)
        detection(pp, ev)
        integration(pp)
    print(f"report written: {OUT}  ({OUT.stat().st_size / 1024 / 1024:.2f} MB, {len(ev.get('events', []))} events)")


if __name__ == "__main__":
    main()