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
import csv, json, math, os, io, base64, time
from pathlib import Path

import numpy as np
from flask import (Flask, Response, jsonify, redirect, render_template,
                   request, send_file, url_for)

ROOT = Path(__file__).resolve().parent

from modules.process_module import ProcessModule
from modules.video_module import VideoModule
from modules.analysis_module import AnalysisModule
from modules.automation_module import AutomationModule
from modules.reporting_module import ReportingModule
from modules.b2_module import B2Module

try:
    import pymongo
    _MONGO = True
except Exception:
    _MONGO = False

# --------------------------------------------------------------------------- #
# app + config                                                                #
# --------------------------------------------------------------------------- #
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024 * 1024
app.secret_key = os.environ.get("SECRET_KEY", "wpm-secret")

MONGO_URI = os.environ.get("MONGO_URI", "")
FRAME_DIR = next(
    (p for p in (os.environ.get("FRAME_DIR", ""),
                 r"D:\ksrtc\frames_1s",
                 str(ROOT / "static" / "sample_frames"))
     if p and os.path.isdir(p)),
    "")
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
REPORT_PATH = ROOT / "outputs" / "WPM_Duty_Report.pdf"
CLIP_DIR = ROOT / "static" / "clips"

# --------------------------------------------------------------------------- #
# Backblaze B2 asset store (frames + clips)                                   #
# --------------------------------------------------------------------------- #
b2 = B2Module(ROOT / "static" / "b2cache")

_LAST_FRAME_N = 3951  # number of 1s frames extracted for this video


def resolve_frame_dir(override=""):
    """Pick a usable frame directory for the ML options.

    Priority: explicit form override -> local FRAME_DIR -> materialise all
    3951 frames from Backblaze B2 -> bundled sample_frames (hosted demos).
    Returns "" when nothing is available.
    """
    if override and os.path.isdir(override):
        return override
    if FRAME_DIR and os.path.isdir(FRAME_DIR):
        return FRAME_DIR
    if b2.enabled:
        try:
            return b2.materialize_frames(_LAST_FRAME_N)
        except Exception as e:
            print(f"[wpm] B2 frame materialisation failed: {e}")
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

# --------------------------------------------------------------- context -- #
@app.context_processor
def inject_globals():
    return {"db_name": DB_NAME, "col_duty": proc.cfg["db"]["collections"]["duty_logs"],
            "process_name": proc.cfg["process_name"]}

# ------------------------------------------------------------------ views -- #
@app.route("/")
def dashboard():
    rows = load_duty_logs(limit=8000)
    if rows is None:
        # fallback: static event log shipped with the app
        ev = ROOT / "data" / "event_log_sample.json"
        rows = json.loads(ev.read_text(encoding="utf-8")) if ev.exists() else []
    s = summary_from_logs(rows)
    chart = chart_payload_from_summary(s)
    return render_template("dashboard.html",
                           summary=s, chart=chart,
                           colors={a["id"]: a["color"] for a in proc.cfg["activities"]},
                           activities=proc.cfg["activities"],
                           db_mode=(rows is not None and _MONGO and MONGO_URI))

@app.route("/analysis")
def analysis_view():
    metrics_path = ROOT / "data" / "metrics_perSecond.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else []
    heat = reporting.zone_heatmap(proc.cfg["zones"]["cell_mean"])
    return render_template("analysis.html", metrics=metrics[:600],
                           heatmap=heat, zones=proc.cfg["zones"],
                           colors={a["id"]: a["color"] for a in proc.cfg["activities"]},
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

@app.route("/ml/predict", methods=["POST"])
def ml_predict():
    """Prediction demo: needs FRAME_DIR with f_XXXX.jpg frames, or accept a
    frame-dir override via form. Predicts, builds events, auto-logs to Mongo,
    computes VA/NVA and returns everything as JSON for the charts."""
    frame_dir = resolve_frame_dir(request.form.get("frame_dir", ""))
    if not frame_dir or not os.path.isdir(frame_dir):
        return jsonify({"ok": False, "error":
                        "No frame source (local FRAME_DIR, B2 or bundled samples)."}), 400
    if not MODEL_PATH.exists():
        return jsonify({"ok": False, "error": "model not trained yet."}), 400
    try:
        n = int(request.form.get("max_frames", 0))  # 0 = all
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
    return jsonify(j)

@app.route("/ml/retrain", methods=["POST"])
def ml_retrain():
    """Background retrain of the ML model from the labelled frames.
    Frames resolve via local FRAME_DIR or are pulled from Backblaze B2."""
    frame_dir = resolve_frame_dir(request.form.get("frame_dir", ""))

    if not frame_dir or not os.path.isdir(frame_dir):
        return jsonify({"ok": False, "error":
                        "No frame source (local FRAME_DIR or B2) to train on."}), 400
    jid = start_job("retrain", _do_retrain, frame_dir)
    return jsonify({"ok": True, "job": jid})

def _do_retrain(frame_dir):
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
    Frames resolve via local FRAME_DIR or are pulled from Backblaze B2."""
    frame_dir = resolve_frame_dir(request.form.get("frame_dir", ""))

    if not frame_dir or not os.path.isdir(frame_dir):
        return jsonify({"ok": False, "error":
                        "No frame source (local FRAME_DIR or B2) to predict."}), 400
    if not MODEL_PATH.exists():
        return jsonify({"ok": False, "error": "model not trained yet."}), 400
    max_frames = int(request.form.get("max_frames", 0) or 0)
    auto_log = request.form.get("auto_log", "1") == "1"
    jid = start_job("predict_full", _do_predict_full, frame_dir,
                    max_frames or None, auto_log)
    return jsonify({"ok": True, "job": jid})

def _do_predict_full(frame_dir, max_frames, auto_log):
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