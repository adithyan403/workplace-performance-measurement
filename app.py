"""WPM - Workplace Performance Measurement web dashboard.

Flask application exposing the five WPM modules to a colorful dashboard:
  * /            - dashboard (KPIs, donuts, timeline)
  * /analysis    - analytics graphs (motion, zones, timestamps)
  * /logs        - MongoDB duty-log explorer (WPM.duty_logs)
  * /ml          - ML automation: model status, predict, auto-log, VA/NVA
  * /reports     - generate + download PDF reports
  * /api/...     - JSON endpoints backing the charts

Deploy target: Render (gunicorn). MongoDB URI read from env MONGO_URI.
"""
import csv, json, math, os, io, base64, time, tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from flask import (Flask, Response, jsonify, redirect, render_template,
                   request, send_file, url_for)
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

ROOT = Path(__file__).resolve().parent

# Keep large upload temp files off the system drive (usually C:): Werkzeug
# spills multipart parts >500 KB to a temp file, so point the process temp dir
# at a folder next to the app (same drive as the project).
_TMP_DIR = ROOT / "tmp"
try:
    _TMP_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(_TMP_DIR)
    os.environ["TEMP"] = str(_TMP_DIR)
    os.environ["TMP"] = str(_TMP_DIR)
    tempfile.tempdir = str(_TMP_DIR)
except OSError:
    pass

from modules.process_module import ProcessModule
from modules.video_module import VideoModule
from modules.analysis_module import AnalysisModule
from modules.automation_module import AutomationModule
from modules.reporting_module import ReportingModule
from modules.b2_module import B2Module
from modules import cnn_module

try:
    import pymongo
    _MONGO = True
except Exception:
    _MONGO = False

# --------------------------------------------------------------------------- #
# app + config                                                                #
# --------------------------------------------------------------------------- #
app = Flask(__name__)
# 0 (or unset) => no upload size limit. Set WPM_MAX_UPLOAD_MB to a positive
# number of megabytes to enforce a cap instead.
MAX_UPLOAD_MB = int(os.environ.get("WPM_MAX_UPLOAD_MB", "0"))
app.config["MAX_CONTENT_LENGTH"] = (MAX_UPLOAD_MB * 1024 * 1024
                                    if MAX_UPLOAD_MB > 0 else None)
app.secret_key = os.environ.get("SECRET_KEY", "wpm-secret")

# --------------------------------------------------------------------------- #
# JSON-friendly error handling for API/upload endpoints                        #
# --------------------------------------------------------------------------- #
_JSON_PREFIXES = ("/ml/", "/api/", "/frames/", "/frame/", "/reports/generate")

def _wants_json():
    return (request.path.startswith(_JSON_PREFIXES)
            or "application/json" in (request.headers.get("Accept") or "").lower())

@app.errorhandler(RequestEntityTooLarge)
def _too_large(e):
    limit = f"{MAX_UPLOAD_MB} MB" if MAX_UPLOAD_MB > 0 else "configured"
    return jsonify({"ok": False, "error":
                    f"Upload too large (server limit {limit} per request). "
                    "Upload fewer frames at a time or split the .zip into "
                    "smaller batches."}), 413

@app.errorhandler(Exception)
def _unhandled(e):
    code = e.code if isinstance(e, HTTPException) and e.code else 500
    desc = e.description if isinstance(e, HTTPException) else str(e)
    if not isinstance(e, HTTPException):
        app.logger.exception("unhandled error on %s", request.path)
    if _wants_json():
        return jsonify({"ok": False, "error": desc, "code": code}), code
    if isinstance(e, HTTPException):
        return e
    return "<h1>Internal Server Error</h1>", 500

MONGO_URI = os.environ.get("MONGO_URI", "")

def _real_frame_dir(path):
    """A usable ML frame dir must hold a meaningful number of f_*.jpg files.
    The bundled sample_frames/ (12 frames) does NOT count — it's a viewer
    fallback, not a training source. This keeps B2 materialisation in play."""
    if not path or not os.path.isdir(path):
        return False
    try:
        n = sum(1 for _ in Path(path).glob("f_*.jpg"))
    except OSError:
        return False
    return n >= 50

_NO_LOCAL = os.environ.get("WPM_NO_LOCAL_FRAMES", "").lower() in ("1", "true", "yes")

def _candidate_dirs():
    d = os.environ.get("FRAME_DIR", "")
    if d:
        return (d,)
    if _NO_LOCAL:
        return ()  # pure B2 mode: never fall back to local frames
    return (r"D:\ksrtc\frames_1s", str(ROOT / "static" / "sample_frames"))

FRAME_DIR = next((p for p in _candidate_dirs() if p and os.path.isdir(p)), "")
VIDEO_PATH = os.environ.get("VIDEO_PATH", r"D:\ksrtc\VID_20260819_135332150.mp4")
CLIP_DIR = ROOT / "static" / "clips"
DB_NAME = "WPM"

cfg_file = ROOT / "config" / "process_config.json"
proc = ProcessModule(cfg_file, ROOT)
analysis = AnalysisModule(proc, FRAME_DIR or ".")
automation = AutomationModule(proc.cfg, proc, ROOT)
reporting = ReportingModule(proc.cfg, proc, ROOT)
video = VideoModule(proc.cfg, ROOT)

MODEL_PATH = ROOT / proc.cfg["ml"]["model_path"]
CNN_ANALYSIS_PATH = ROOT / "data" / "cnn_analysis.json"
CNN_UPLOAD_DIR = ROOT / "static" / "cnn_uploads"
CNN_REPORT_PDF_PATH = ROOT / "outputs" / "WPM_CNN_Analysis_Report.pdf"

# Selectable CNN activity-detection models. Each checkpoint stores its "arch"
# (and "model_name"), so CnnDetector loads the right network automatically.
CNN_MODELS = {
    "resnet18": {
        "label": "ResNet-18",
        "path": ROOT / "data" / "cnn_resnet18.pt",
        "report": ROOT / "data" / "cnn_report.json",
    },
    "convnextv2_tiny": {
        "label": "ConvNeXt V2-Tiny",
        "path": ROOT / "data" / "cnn_convnextv2_tiny.pt",
        "report": ROOT / "data" / "cnn_convnextv2_report.json",
    },
}
CNN_DEFAULT_MODEL = "resnet18"
CNN_MODEL_PATH = CNN_MODELS[CNN_DEFAULT_MODEL]["path"]
CNN_REPORT_PATH = CNN_MODELS[CNN_DEFAULT_MODEL]["report"]
REPORT_PATH = ROOT / "outputs" / "WPM_Duty_Report.pdf"
CLIP_DIR = ROOT / "static" / "clips"

# --------------------------------------------------------------------------- #
# Backblaze B2 asset store (frames + clips)                                   #
#   Prefer Render's persistent disk (/data) so the frames.zip + clips download #
#   once and survive deploys, instead of re-fetching on every boot.           #
# --------------------------------------------------------------------------- #
if os.path.isdir("/data") and os.access("/data", os.W_OK):
    _B2_CACHE = Path("/data") / "b2cache"
else:
    _B2_CACHE = ROOT / "static" / "b2cache"
b2 = B2Module(_B2_CACHE)

_LAST_FRAME_N = 3951  # number of 1s frames extracted for this video


def resolve_frame_dir(override="", need_frames=None, strict=False):
    """Pick a usable frame directory for the ML options.

    Priority: explicit form override -> local FRAME_DIR (only if it holds a
    real frame set) -> materialise (need_frames or all) from Backblaze B2 ->
    bundled sample_frames (hosted demos). Returns "" when nothing is usable.
    When strict, a failing B2 materialisation re-raises instead of falling
    back to the bundled samples, so the real reason (e.g. download cap)
    reaches the job result.
    """
    if override and _real_frame_dir(override):
        return override
    if _real_frame_dir(FRAME_DIR):
        return FRAME_DIR
    if b2.enabled:
        try:
            return b2.materialize_frames(need_frames or _LAST_FRAME_N)
        except Exception as e:
            print(f"[wpm] B2 frame materialisation failed: {e}")
            if strict or not (ROOT / "static" / "sample_frames").is_dir():
                raise RuntimeError(f"B2 frame materialisation failed: {e}")
    bundled = ROOT / "static" / "sample_frames"
    if bundled.is_dir():
        return str(bundled)
    return ""

# ---------------------------------------------------------------- helpers -- #
# background job registry, persisted to .jobs/ so entries survive dev-reloader
# restarts and gunicorn multi-worker setups (shared via the filesystem)
import threading as _threading
_JOBS_LOCK = _threading.Lock()
_JOBS_DIR = ROOT / ".jobs"
_JOBS_DIR.mkdir(parents=True, exist_ok=True)
_TRUE = ("true", "1", "yes")

def _job_path(jid):
    return _JOBS_DIR / f"{jid}.json"

def _job_save(job):
    tmp = _job_path(job["id"]).with_suffix(".tmp")
    tmp.write_text(json.dumps(job), encoding="utf-8")
    tmp.replace(_job_path(job["id"]))

def start_job(kind, fn, *args, **kwargs):
    jid = f"{kind}-{os.urandom(3).hex()}"
    job = {"id": jid, "kind": kind, "status": "running",
           "progress": 0, "message": "started", "result": None, "error": None}
    _job_save(job)
    def _run():
        try:
            res = fn(*args, **kwargs)
            job["status"] = "done"
            job["progress"] = 100
            job["result"] = res
            _job_save(job)
        except Exception as e:
            job["status"] = "error"
            job["error"] = str(e)
            _job_save(job)
    t = _threading.Thread(target=_run, daemon=True)
    t.start()
    return jid

def get_job(jid):
    if os.sep in jid or "/" in jid or ".." in jid:
        return None  # don't let a jid escape .jobs/
    p = _job_path(jid)
    if not p.exists():
        return None
    with _JOBS_LOCK:
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

def prune_jobs(max_age_hours=24):
    """Delete old finished jobs so .jobs/ doesn't grow forever."""
    cutoff = time.time() - max_age_hours * 3600
    for p in _JOBS_DIR.glob("*.json"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
        except OSError:
            pass

def _job_update(jid, **fields):
    """Patch fields on an existing job record (used for live progress)."""
    p = _job_path(jid)
    with _JOBS_LOCK:
        try:
            j = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        except Exception:
            j = None
        if j is None:
            return
        j.update(fields)
        try:
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(j, default=str), encoding="utf-8")
            tmp.replace(p)
        except OSError:
            pass

def start_progress_job(kind, worker, *args):
    """Start a background job whose worker(jid, *args) reports live progress
    via _job_update(). Returns the job id immediately."""
    jid = f"{kind}-{int(time.time())}-{os.urandom(3).hex()}"
    _job_save({"id": jid, "kind": kind, "status": "running",
               "progress": 0, "message": "queued"})

    def run():
        try:
            res = worker(jid, *args)
            _job_update(jid, status="done", progress=100,
                        message="done", result=res)
        except Exception as e:
            _job_update(jid, status="error", error=str(e))
    _threading.Thread(target=run, daemon=True).start()
    return jid

def get_mongo():
    if not (_MONGO and MONGO_URI):
        return None
    if not getattr(get_mongo, "_client", None):
        get_mongo._client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=12000)
    return get_mongo._client

def load_duty_logs(limit=5000, activity=None, va=None, q=None):
    client = get_mongo()
    if client is None:
        return None
    db = client[DB_NAME]
    coll = db[proc.cfg["db"]["collections"]["duty_logs"]]
    filt = {}
    if activity:
        filt["activity"] = activity
    if va:
        filt["va_nva"] = va
    if q:
        filt["time_frame.start_ts"] = {"$regex": q, "$options": "i"}
    rows = list(coll.find(filt, {"_id": 0, "frames": 0}).limit(limit))
    rows.sort(key=lambda r: r.get("time_frame", {}).get("start_sec", 0))
    return rows

def summary_from_logs(rows):
    total = sum(r["time_frame"]["duration_s"] for r in rows) if rows else 0
    dist, va = {}, 0
    for r in rows:
        act = r.get("activity") or "UNKNOWN"
        d = dist.setdefault(act, {"count": 0, "sec": 0})
        d["count"] += 1
        d["sec"] += r["time_frame"]["duration_s"]
        if r.get("va_nva") == "VA":
            va += r["time_frame"]["duration_s"]
    nva = total - va
    return {
        "event_count": len(rows) if rows else 0,
        "total_sec": total,
        "activity_distribution": {
            a: {"count": d["count"], "sec": d["sec"],
                "pct": round(100*d["sec"]/total, 1) if total else 0}
            for a, d in dist.items()},
        "va_nva": {"VA": {"sec": va, "pct": round(100*va/total, 1) if total else 0},
                   "NVA": {"sec": nva, "pct": round(100*nva/total, 1) if total else 0}},
    }

@app.template_filter("format_ns")
def _format_ns(v):
    try:
        return f"{int(v):,}"
    except Exception:
        return str(v)

@app.template_filter("zone_color")
def _zone_color(v):
    """Map grid-cell motion energy value to a color scale (heatmap)."""
    try:
        v = float(v)
    except Exception:
        v = 0.0
    hi = 20.0
    colors = ["#dbeafe", "#93c5fd", "#60a5fa", "#38bdf8", "#fbbf24", "#f97316"]
    idx = min(len(colors)-1, int(v / max(hi, 1) * (len(colors)-1)))
    return colors[idx]

def chart_payload_from_summary(s):
    acts = [a["id"] for a in proc.cfg["activities"]]
    colors = [a["color"] for a in proc.cfg["activities"]]
    dist = s.get("activity_distribution", {})
    labels = [a for a in acts if a in dist]
    return {
        "labels": labels,
        "colors": [proc.color(a) for a in labels],
        "sec": [dist[a].get("sec", 0) for a in labels],
        "pct": [dist[a].get("pct", 0) for a in labels],
    }

def read_training_report():
    p = ROOT / "data" / "training_report.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))

# ----------------------------------------------------- CNN analysis sources -- #
CNN_ANALYSIS_SOURCE = "data/cnn_events.json"   # full-video prediction analysis

def load_cnn_analysis():
    """Latest CNN prediction analysis (per-frame labels + events + VA/NVA mix).

    Sources, in order:
      1) data/cnn_analysis.json   -> the most recent uploaded-prediction session
      2) data/cnn_events.json     -> the full-video detection run (committed)
    Returns a normalized dict (always with events, total, va_nva, activities,
    session) or None when neither exists.
    """
    paths = [CNN_ANALYSIS_PATH, ROOT / CNN_ANALYSIS_SOURCE]
    for p in paths:
        if p.exists():
            try:
                return _normalize_cnn_analysis(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                continue
    return None

def _normalize_cnn_analysis(a):
    """Fill the fields the templates/report rely on, whatever the source shape."""
    events = a.get("events", []) or []
    a["events"] = events
    a.setdefault("activities", list(cnn_module.CLASSES))
    total = int(a.get("total", 0) or a.get("n_frames", 0)
                or (max((e.get("end", 0) for e in events), default=-1) + 1))
    a["total"] = total
    a["n_frames"] = int(a.get("n_frames", 0) or total)
    if not a.get("va_nva"):
        va = sum(e.get("duration", 0) for e in events
                 if str(e.get("activity", "")).lower() in cnn_module.VA_CLASSES)
        nva = total - va
        a["va_nva"] = {"total": total,
                       "VA": {"sec": va, "pct": round(100*va/total, 2) if total else 0},
                       "NVA": {"sec": nva, "pct": round(100*nva/total, 2) if total else 0}}
    a.setdefault("session", "full-video run")
    a.setdefault("source", "full-video")
    a.setdefault("generated_at", "—")
    return a

def cnn_events_rows(analysis):
    """Build summary_from_logs()-compatible rows from a CNN analysis doc.

    The CNN works with lowercase class ids (write/crew/check/wait); the
    dashboard maps them back onto the config's canonical ids so existing
    colors/labels/VA rules keep working.
    """
    rows = []
    for e in analysis.get("events", []):
        act = (e.get("activity") or "").upper()
        if not act:
            continue
        rows.append({"activity": act,
                     "va_nva": "VA" if act in {a["id"] for a in proc.cfg["activities"] if a["va"]} else "NVA",
                     "time_frame": {"duration_s": e.get("duration", 0)}})
    return rows

def cnn_summary_payload(analysis):
    """Dashboard summary + chart payload derived from CNN predictions."""
    s = summary_from_logs(cnn_events_rows(analysis))
    return s, chart_payload_from_summary(s)

def cnn_timeline_bands(analysis, n=240):
    """Compact per-band activity band for a color-coded timeline strip.

    Aggregates the per-frame predictions/events into 'n' time buckets, each
    labelled with its dominant activity — used to draw the CNN prediction
    timeline on the dashboard / analysis page.
    """
    try:
        total = int(analysis.get("total", 0)) or int(analysis.get("n_frames", 0))
    except Exception:
        total = 0
    events = analysis.get("events", [])
    if not total and events:
        total = max(e.get("end", 0) for e in events) + 1
    if not total:
        return []
    # build a per-second activity array from events (sparse)
    sec_labels = [None] * total
    for e in events:
        for s in range(int(e.get("start", 0)), int(e.get("end", 0)) + 1):
            if 0 <= s < total:
                sec_labels[s] = (e.get("activity") or "").lower()
    step = max(1, (total + n - 1) // n)
    bands = []
    for s in range(0, total, step):
        bucket = [x for x in sec_labels[s:s + step] if x]
        act = Counter(bucket).most_common(1)[0][0] if bucket else "?"
        bands.append({"activity": act, "sec": s, "end": min(s + step, total),
                      "color": proc.color(act)})
    return bands

def save_cnn_analysis(analysis_dict):
    """Persist a prediction run so the whole site reads the same CNN results."""
    analysis_dict.setdefault("source", "upload")
    analysis_dict.setdefault("generated_at", time.strftime("%Y-%m-%d %H:%M:%S"))
    CNN_ANALYSIS_PATH.write_text(json.dumps(analysis_dict, indent=2, default=str),
                                 encoding="utf-8")
    return CNN_ANALYSIS_PATH

def cnn_model_info(key):
    """Registry entry for a CNN model key (falls back to the default model)."""
    return CNN_MODELS.get(key) or CNN_MODELS[CNN_DEFAULT_MODEL]

def cnn_models_list():
    """[{key,label,available,has_report}] used by the model-picker UI."""
    return [{"key": k, "label": m["label"], "available": m["path"].exists(),
             "has_report": m["report"].exists()}
            for k, m in CNN_MODELS.items()]

def _load_json(path):
    """Read a JSON file, returning None when missing/invalid."""
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except Exception:
        return None

def cnn_report_for(key):
    """Training report for a model key, field names normalised for the UI.

    The ResNet-18 trainer writes test_acc/n_test; the ConvNeXt trainer writes
    test_accuracy/test_frames — normalise both to the latter.
    """
    d = _load_json(cnn_model_info(key)["report"])
    if isinstance(d, dict):
        if "test_accuracy" not in d and "test_acc" in d:
            d["test_accuracy"] = d["test_acc"]
        if "test_frames" not in d and "n_test" in d:
            d["test_frames"] = d["n_test"]
        d.setdefault("classes", list(cnn_module.CLASSES))
    return d

# --------------------------------------------------------------- context -- #
@app.context_processor
def inject_globals():
    return {"db_name": DB_NAME, "col_duty": proc.cfg["db"]["collections"]["duty_logs"],
            "process_name": proc.cfg["process_name"]}

# ------------------------------------------------------------------ views -- #
@app.route("/")
def dashboard():
    cnn_analysis = load_cnn_analysis()
    if cnn_analysis and cnn_analysis.get("events"):
        # deep-learning analysis: everything on this page is driven by CNN
        # predictions (per-frame ResNet-18 labels -> duty events -> VA/NVA)
        s, chart = cnn_summary_payload(cnn_analysis)
        return render_template("dashboard.html",
                               summary=s, chart=chart,
                               analysis_source="cnn", cnn_analysis=cnn_analysis,
                               cnn_bands=cnn_timeline_bands(cnn_analysis),
                               colors={a["id"]: a["color"]
                                       for a in proc.cfg["activities"]},
                               activities=proc.cfg["activities"],
                               db_mode=False)
    # fallback: classic duty logs (Mongo or shipped sample)
    rows = load_duty_logs(limit=8000)
    if rows is None:
        # fallback: static event log shipped with the app
        ev = ROOT / "data" / "event_log_sample.json"
        rows = json.loads(ev.read_text(encoding="utf-8")) if ev.exists() else []
    s = summary_from_logs(rows)
    chart = chart_payload_from_summary(s)
    return render_template("dashboard.html",
                           summary=s, chart=chart,
                           analysis_source="logs",
                           colors={a["id"]: a["color"] for a in proc.cfg["activities"]},
                           activities=proc.cfg["activities"],
                           db_mode=(rows is not None and _MONGO and MONGO_URI))

@app.route("/analysis")
def analysis_view():
    cnn_analysis = load_cnn_analysis()
    cnn_report = cnn_report_for((cnn_analysis or {}).get("model"))

    metrics_path = ROOT / "data" / "metrics_perSecond.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else []
    heat = reporting.zone_heatmap(proc.cfg["zones"]["cell_mean"])

    cnn_ctx = None
    if cnn_analysis and cnn_analysis.get("events"):
        s, chart = cnn_summary_payload(cnn_analysis)
        cnn_ctx = {
            "analysis": cnn_analysis,
            "summary": s,
            "chart": chart,
            "bands": cnn_timeline_bands(cnn_analysis),
            "report": cnn_report,
            "events": cnn_analysis.get("events", [])[:200],
            "colors": {c: proc.color(c) for c in cnn_module.CLASSES},
        }
    return render_template("analysis.html", metrics=metrics[:600],
                           heatmap=heat, zones=proc.cfg["zones"],
                           cnn_ctx=cnn_ctx,
                           colors={a["id"]: a["color"]
                                   for a in proc.cfg["activities"]},
                           activities=proc.cfg["activities"])

@app.route("/logs")
def logs_view():
    activity = request.args.get("activity", "")
    va = request.args.get("va", "")
    q = request.args.get("q", "")
    rows = load_duty_logs(limit=5000, activity=activity or None, va=va or None, q=q or None)
    if rows is None:
        return render_template("logs.html", error="MongoDB not configured (set MONGO_URI).")
    return render_template("logs.html", rows=rows, activity=activity, va=va, q=q,
                           activities=[a["id"] for a in proc.cfg["activities"]])

@app.route("/ml")
def ml_view():
    report = read_training_report()
    model_ok = MODEL_PATH.exists()
    return render_template("ml.html", report=report, model_ok=model_ok,
                           classes=proc.cfg["ml"]["classes"],
                           colors={a["id"]: a["color"] for a in proc.cfg["activities"]})

def _cnn_uploaded_count():
    """Frames inside the most recent upload session (or 0)."""
    try:
        if not CNN_UPLOAD_DIR.is_dir():
            return 0
        runs = sorted([d for d in CNN_UPLOAD_DIR.iterdir()
                       if d.is_dir() and d.name.startswith("run_")])
        if not runs:
            return 0
        return len(list(runs[-1].iterdir()))
    except Exception:
        return 0


@app.route("/ml/cnn")
def ml_cnn_view():
    """Deep-learning activity detection over uploaded frames (model picker)."""
    key = request.args.get("model", CNN_DEFAULT_MODEL)
    info = cnn_model_info(key)
    return render_template("cnn.html",
                           cnn_ok=info["path"].exists(),
                           dl_ok=cnn_module.available(),
                           models=cnn_models_list(),
                           selected_model=key if key in CNN_MODELS
                           else CNN_DEFAULT_MODEL,
                           model_label=info["label"],
                           classes=cnn_module.CLASSES,
                           classes_colors={c: proc.color(c)
                                           for c in cnn_module.CLASSES},
                           va_classes=sorted(cnn_module.VA_CLASSES),
                           cnn_report=cnn_report_for(key),
                           uploaded=_cnn_uploaded_count())

@app.route("/ml/cnn/predict", methods=["POST"])
def ml_cnn_predict():
    """Accept uploaded frames (multiple images or a .zip), store them, then
    run the CNN as a background job so the browser can show live progress."""
    model_key = request.form.get("model", CNN_DEFAULT_MODEL)
    if model_key not in CNN_MODELS:
        return jsonify({"ok": False, "error":
                        f"Unknown model '{model_key}'."}), 400
    info = CNN_MODELS[model_key]
    if not info["path"].exists():
        return jsonify({"ok": False, "error":
                        f"Model '{info['label']}' not found "
                        f"({info['path'].name}). Train it locally and ship the "
                        "checkpoint to data/."}), 400
    if not cnn_module.available():
        return jsonify({"ok": False, "error":
                        "PyTorch is not installed on this server. Add torch + "
                        "torchvision (and Pillow) to requirements."}), 400
    files = request.files.getlist("frames")
    if not files or not files[0] or files[0].filename == "":
        return jsonify({"ok": False, "error": "Select at least one frame image "
                        "or a .zip before running detection."}), 400

    sess_dir = CNN_UPLOAD_DIR / f"run_{int(time.time())}"
    sess_dir.mkdir(parents=True, exist_ok=True)

    # keep only the newest 5 sessions to bound disk usage
    try:
        runs = sorted([d for d in CNN_UPLOAD_DIR.iterdir()
                       if d.is_dir() and d.name.startswith("run_")],
                      key=lambda d: d.name)
        for old in runs[:-5]:
            import shutil
            shutil.rmtree(old, ignore_errors=True)
    except Exception:
        pass

    n_uploaded = 0
    for f in files:
        name = os.path.basename(f.filename or "")
        if not name:
            continue
        safe = Path(name).name
        if safe.lower().endswith(".zip"):
            zpath = sess_dir / safe
            f.save(str(zpath))
            n_uploaded += cnn_module.extract_upload(zpath, sess_dir)
            continue
        if safe.lower().endswith((".jpg", ".jpeg", ".png")):
            f.save(str(sess_dir / safe))
            n_uploaded += 1

    img_paths = cnn_module.collect_images(sess_dir)
    if not img_paths:
        return jsonify({"ok": False, "error":
                        "No valid frame images (.jpg/.png) found in upload."}), 400

    jid = start_progress_job("cnn_predict", _do_cnn_predict, str(sess_dir),
                             model_key)
    return jsonify({"ok": True, "job": jid, "n_uploaded": n_uploaded,
                    "session": sess_dir.name, "model": model_key,
                    "model_label": info["label"]})

def _do_cnn_predict(jid, sess_dir, model_key=CNN_DEFAULT_MODEL):
    """Background worker: classify every uploaded frame, reporting progress."""
    info = cnn_model_info(model_key)
    img_paths = cnn_module.collect_images(sess_dir)
    total = len(img_paths)
    _job_update(jid, message=f"Loading {info['label']} model…")
    det = cnn_module.CnnDetector(info["path"])

    def on_progress(done, tot):
        _job_update(jid, progress=round(100 * done / tot) if tot else 0,
                    message=f"Classifying frames {done}/{tot}")

    preds = det.classify_paths(img_paths, on_progress=on_progress)
    events = cnn_module.predictions_to_events(preds)
    vanva = cnn_module.find_va_nva(events)
    dist = dict(Counter(preds))
    labels = [{"file": Path(p).name, "activity": a,
               "va_nva": "VA" if a in cnn_module.VA_CLASSES else "NVA",
               "index": i} for i, (p, a) in enumerate(zip(img_paths, preds))]

    # persist this run as the site-wide CNN analysis source
    save_cnn_analysis({
        "session": Path(sess_dir).name,
        "source": "upload",
        "model": model_key,
        "model_label": info["label"],
        "n_frames": len(preds),
        "activities": cnn_module.CLASSES,
        "events": events,
        "total": len(preds),
        "va_nva": vanva,
    })

    return {
        "model": model_key,
        "model_label": info["label"],
        "n_frames": len(preds),
        "activities": cnn_module.CLASSES,
        "activity_colors": {c: proc.color(c) for c in cnn_module.CLASSES},
        "va_classes": sorted(cnn_module.VA_CLASSES),
        "distribution": dist,
        "va_nva": vanva,
        "labels": labels[:400],
        "events": events[:120],
        "session": Path(sess_dir).name,
    }

@app.route("/ml/cnn/report", methods=["GET"])
def ml_cnn_report_view():
    """CNN analysis report: generate + download the latest prediction analysis."""
    analysis = load_cnn_analysis()
    info = cnn_model_info((analysis or {}).get("model"))
    cnn_report = cnn_report_for((analysis or {}).get("model"))
    cnn_chart = None
    if analysis and analysis.get("events"):
        _, cnn_chart = cnn_summary_payload(analysis)
    return render_template("cnn_report.html",
                           cnn_ok=info["path"].exists(),
                           analysis=analysis,
                           model_label=info["label"],
                           cnn_report=cnn_report,
                           cnn_chart=cnn_chart,
                           report_exists=CNN_REPORT_PDF_PATH.exists(),
                           classes=cnn_module.CLASSES,
                           classes_colors={c: proc.color(c)
                                           for c in cnn_module.CLASSES},
                           va_classes=sorted(cnn_module.VA_CLASSES))

@app.route("/ml/cnn/report/generate", methods=["POST"])
def ml_cnn_report_generate():
    """Build the CNN analysis PDF from the latest saved prediction run."""
    analysis = load_cnn_analysis()
    if not analysis or not analysis.get("events"):
        return jsonify({"ok": False, "error":
                        "No CNN prediction analysis yet. Run a detection first "
                        "(upload frames on the CNN page)."}), 400
    try:
        import importlib
        cnn_report = cnn_report_for(analysis.get("model"))
        class_colors = {c: proc.color(c) for c in cnn_module.CLASSES}
        path = reporting.generate_cnn_report(analysis, CNN_REPORT_PDF_PATH,
                                             cnn_report=cnn_report,
                                             class_colors=class_colors)
        return jsonify({"ok": True, "path": path,
                        "size": os.path.getsize(path)})
    except Exception as e:
        return jsonify({"ok": False, "error": f"report failed: {e}"}), 500

@app.route("/ml/cnn/report/download")
def ml_cnn_report_download():
    if not CNN_REPORT_PDF_PATH.exists():
        return jsonify({"ok": False, "error": "no report yet"}), 404
    return send_file(CNN_REPORT_PDF_PATH, as_attachment=True,
                     download_name="WPM_CNN_Analysis_Report.pdf")

@app.route("/ml/predict", methods=["POST"])
def ml_predict():
    """Prediction demo: needs FRAME_DIR with f_XXXX.jpg frames, or accept a
    frame-dir override via form. Predicts, builds events, auto-logs to Mongo,
    computes VA/NVA and returns everything as JSON for the charts."""
    n = 0
    try:
        n = int(request.form.get("max_frames", 0))  # 0 = all
    except (TypeError, ValueError):
        n = 0
    need = n if n > 0 else None
    frame_dir = resolve_frame_dir(request.form.get("frame_dir", ""),
                                  need_frames=need)
    if not frame_dir or not _real_frame_dir(frame_dir):
        return jsonify({"ok": False, "error":
                        "No full frame set available. Set FRAME_DIR or B2_KEY_ID/"
                        "B2_APPLICATION_KEY (Backblaze B2) for the ML options."}), 400
    if not MODEL_PATH.exists():
        return jsonify({"ok": False, "error": "model not trained yet."}), 400
    try:
        secs, preds = automation.predict_frames(frame_dir, MODEL_PATH, start_sec=0,
                                                max_frames=n or None)
        events = automation.predictions_to_events(secs, preds)
        for e in events:
            e["va_nva"] = "VA" if proc.is_va(e["activity"]) else "NVA"
        vanva = automation.find_va_nva(events)
        inserted = 0
        if request.form.get("auto_log", "1") == "1" and MONGO_URI:
            res = automation.auto_log(events, MONGO_URI)
            inserted = res.get("inserted", 0)
        payload = {
            "ok": True,
            "events": events[:400],
            "event_stats": summary_from_logs(
                [{"time_frame": {"duration_s": e["duration"]}, "activity": e["activity"],
                  "va_nva": "VA" if proc.is_va(e["activity"]) else "NVA"} for e in events]),
            "va_nva": vanva,
            "inserted_mongo": inserted,
            "n_frames": len(preds),
        }
        return jsonify(payload)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/ml/confusion")
def ml_confusion():
    report = read_training_report()
    if not report:
        return jsonify({"ok": False, "error": "no training report"}), 404
    return jsonify(report)

@app.route("/reports")
def reports_view():
    return render_template("reports.html", report_exists=REPORT_PATH.exists(),
                           has_model=MODEL_PATH.exists())

@app.route("/reports/generate", methods=["POST"])
def reports_generate():
    rows = load_duty_logs(limit=2000) or []
    s = summary_from_logs(rows)
    if not s["event_count"]:
        return jsonify({"ok": False, "error": "no duty logs in MongoDB to report"}), 400
    event_rows = [[r["time_frame"]["start_ts"], r["time_frame"]["end_ts"],
                   r["time_frame"]["duration_s"], r["activity"], r["va_nva"]]
                  for r in rows[:60]]
    path = reporting.generate_pdf(s, event_rows, REPORT_PATH,
                                  va_nva=s.get("va_nva"))
    return jsonify({"ok": True, "path": path, "size": os.path.getsize(path)})

@app.route("/reports/download")
def reports_download():
    if not REPORT_PATH.exists():
        return jsonify({"ok": False, "error": "no report yet"}), 404
    return send_file(REPORT_PATH, as_attachment=True,
                     download_name="WPM_Duty_Report.pdf")

def _local_frame(sec):
    """Local candidate path for a frame (FRAME_DIR or bundled samples)."""
    if FRAME_DIR and os.path.isdir(FRAME_DIR):
        return Path(FRAME_DIR) / f"f_{sec + 1:04d}.jpg"
    return ROOT / "static" / "sample_frames" / f"f_{sec + 1:04d}.jpg"

def _fetch_frame(sec):
    """Resolve a frame to a real file on disk (local first, then B2)."""
    local = _local_frame(sec)
    if local.exists():
        return local
    if b2.enabled:
        cached = b2.download(f"frames/f_{sec + 1:04d}.jpg")
        if cached:
            return Path(cached)
    return None

@app.route("/frame/<int:sec>")
def frame_sec(sec):
    """Serve the 1-second frame JPG for a given second (0-based)."""
    f = _fetch_frame(sec)
    if f is None:
        return jsonify({"ok": False,
                        "error": f"frame f_{sec + 1:04d}.jpg not found"}), 404
    return send_file(f, mimetype="image/jpeg", conditional=True)

@app.route("/frames/<int:start>/<int:end>")
def frames_range(start, end):
    """List the per-second frames covering one event (sec range inclusive)."""
    secs = list(range(max(0, start), max(start, end) + 1))
    step = 1 if len(secs) <= 60 else max(1, round(len(secs) / 60))
    out = []
    for s in secs[::step]:
        if _fetch_frame(s) is None:
            continue
        out.append({"sec": s, "ts": automation._ts(s),
                    "url": url_for("frame_sec", sec=s)})
    return jsonify({"ok": True, "frames": out, "trimmed": len(secs) != len(out)})

@app.route("/activities")
def activities_view():
    """List every activity detected in the video with a playable clip."""
    rows = load_duty_logs(limit=10000) or []
    clip_dir = CLIP_DIR
    clip_dir.mkdir(parents=True, exist_ok=True)
    # one batched B2 listing of already-uploaded clips (avoid per-row API calls)
    b2_clips = set()
    if b2.enabled:
        try:
            b2_clips = set(b2.list_keys_cached("clips/"))
        except Exception:
            b2_clips = set()
    # annotate cached clip existence for each event (local disk or B2)
    annotated = []
    for r in rows:
        r = dict(r)
        start = r["time_frame"]["start_sec"]; end = r["time_frame"]["end_sec"]
        r["_clip_url"] = url_for("event_clip", start=start, end=end)
        r["_clip_cached"] = (_clip_cache_file(start, end).exists()
                             or f"clips/clip_{start}_{end}.mp4" in b2_clips)
        annotated.append(r)
    video_ok = (os.path.exists(VIDEO_PATH) or b2.enabled)
    return render_template("activities.html", rows=annotated,
                           activities=[a["id"] for a in proc.cfg["activities"]],
                           colors={a["id"]: a["color"] for a in proc.cfg["activities"]},
                           video_ok=video_ok, video_path=VIDEO_PATH)

def _clip_cache_file(start, end):
    return CLIP_DIR / f"clip_{start}_{end}.mp4"

@app.route("/clip/<int:start>/<int:end>")
def event_clip(start, end):
    """Stream the video segment covering an event.

    Priority: local cache -> Backblaze B2 -> ffmpeg cut from source video.
    Any freshly generated clip is also uploaded to B2 for future calls.
    """
    cache = _clip_cache_file(start, end)
    key = f"clips/clip_{start}_{end}.mp4"
    if cache.exists() and cache.stat().st_size > 0:
        return send_file(cache, mimetype="video/mp4", conditional=True)
    if b2.enabled and b2.exists(key):
        got = b2.download(key)
        if got:
            return send_file(got, mimetype="video/mp4", conditional=True)
    # generate from local source video
    if not video._find_ffmpeg():
        return jsonify({"ok": False,
                        "error": "no ffmpeg and clip not in B2/cache"}), 400
    try:
        src = os.environ.get("VIDEO_SOURCE", VIDEO_PATH)
        video.cut_clip(src, start, max(1, end - start + 1), cache,
                       max_width=960, cap_duration=30)
    except Exception as e:
        return jsonify({"ok": False, "error": f"clip failed: {e}"}), 500
    if b2.enabled:
        try:
            b2.upload(key, cache)
        except Exception:
            pass  # local clip still usable
    return send_file(cache, mimetype="video/mp4", conditional=True)

@app.route("/api/jobs/<jid>")
def job_status(jid):
    j = get_job(jid)
    if j is None:
        return jsonify({"ok": False, "error": "unknown job"}), 404
    return jsonify({"ok": True, **j})

@app.route("/ml/retrain", methods=["POST"])
def ml_retrain():
    """Background retrain of the ML model from the labelled frames.
    Frame sourcing (local FRAME_DIR or B2 materialisation) runs inside the
    job thread so the browser always gets a job id immediately."""
    override = request.form.get("frame_dir", "")
    if not _real_frame_dir(FRAME_DIR) and not b2.enabled:
        return jsonify({"ok": False, "error":
                        "No full frame set available. Set FRAME_DIR or "
                        "B2_KEY_ID/B2_APPLICATION_KEY (Backblaze B2)."}), 400
    jid = start_job("retrain", _do_retrain, override)
    return jsonify({"ok": True, "job": jid})

def _do_retrain(override):
    from modules import b2_module
    frame_dir = resolve_frame_dir(override, strict=True)
    if not frame_dir or not _real_frame_dir(frame_dir):
        raise RuntimeError("no full frame set to train on: set FRAME_DIR or B2_KEY_ID/B2_APPLICATION_KEY")
    # labels come from the bundled Phase-I event log (start/end/activity)
    ev_sample = ROOT / "data" / "event_log_sample.json"
    locate_label = {}
    if ev_sample.exists():
        for e in json.loads(ev_sample.read_text(encoding="utf-8")):
            tf = e["time_frame"]
            for s in range(int(tf["start_sec"]), int(tf["end_sec"]) + 1):
                locate_label[s] = e["activity"]
    X, y, secs = automation.build_dataset(frame_dir, sec_labels=locate_label)
    if len(X) < 50:
        raise RuntimeError(f"too few frames ({len(X)}) to train")
    report = automation.train(X, y, MODEL_PATH)
    (ROOT / "data" / "training_report.json").write_text(
        json.dumps(report, default=str), encoding="utf-8")
    return {"accuracy": round(report["accuracy"], 4),
            "samples": int(len(X)), "n_train": int(report["n_train"]),
            "n_test": int(report["n_test"])}

@app.route("/ml/predict-full", methods=["POST"])
def ml_predict_full():
    """Background re-prediction of the whole video + auto-log to Mongo.
    Frame sourcing (local FRAME_DIR or B2 materialisation) runs inside the
    job thread so the browser always gets a job id immediately."""
    override = request.form.get("frame_dir", "")
    if not _real_frame_dir(FRAME_DIR) and not b2.enabled:
        return jsonify({"ok": False, "error":
                        "No full frame set available. Set FRAME_DIR or "
                        "B2_KEY_ID/B2_APPLICATION_KEY (Backblaze B2)."}), 400
    if not MODEL_PATH.exists():
        return jsonify({"ok": False, "error": "model not trained yet."}), 400
    max_frames = int(request.form.get("max_frames", 0) or 0)
    auto_log = request.form.get("auto_log", "1") == "1"
    jid = start_job("predict_full", _do_predict_full,
                    override, max_frames or None, auto_log)
    return jsonify({"ok": True, "job": jid})

def _do_predict_full(override, max_frames, auto_log):
    from modules import b2_module
    frame_dir = resolve_frame_dir(override, strict=True)
    if not frame_dir or not _real_frame_dir(frame_dir):
        raise RuntimeError("no full frame set to predict: set FRAME_DIR or B2_KEY_ID/B2_APPLICATION_KEY")
    secs, preds = automation.predict_frames(frame_dir, MODEL_PATH,
                                            start_sec=0, max_frames=max_frames)
    events = automation.predictions_to_events(secs, preds)
    for e in events:
        e["va_nva"] = "VA" if proc.is_va(e["activity"]) else "NVA"
    vanva = automation.find_va_nva(events)
    inserted = 0
    if auto_log and MONGO_URI:
        inserted = automation.auto_log(events, MONGO_URI).get("inserted", 0)
    return {"n_frames": len(preds), "n_events": len(events),
            "inserted_mongo": inserted, "va_nva": vanva,
            "activity_mix": {
                a: sum(1 for e in events if e["activity"] == a)
                for a in proc.cfg["ml"]["classes"]}}

@app.route("/api/health")
def health():
    mongo_info = {}
    client = get_mongo()
    if client:
        try:
            client.admin.command("ping")
            mongo_info = {"ok": True,
                          "docs": client[DB_NAME][proc.cfg["db"]["collections"]["duty_logs"]].count_documents({})}
        except Exception as e:
            mongo_info = {"ok": False, "error": str(e)}
    return jsonify({"status": "ok", "model": MODEL_PATH.exists(), "mongo": mongo_info,
                    "b2": {"enabled": b2.enabled,
                           "bucket": b2.bucket_name() if b2.enabled else None}})

# ------------------------------------------------------------------ main -- #
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)