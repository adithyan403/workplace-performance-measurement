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
import csv, json, math, os, io, base64
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
FRAME_DIR = os.environ.get("FRAME_DIR", "")
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

# ---------------------------------------------------------------- helpers -- #
# simple background job registry (in-memory; per-process)
import threading as _threading
_JOBS = {}
_JOBS_LOCK = _threading.Lock()

def start_job(kind, fn, *args, **kwargs):
    jid = f"{kind}-{os.urandom(3).hex()}"
    with _JOBS_LOCK:
        _JOBS[jid] = {"id": jid, "kind": kind, "status": "running",
                      "progress": 0, "message": "started", "result": None, "error": None}
    def _run():
        try:
            res = fn(*args, **kwargs)
            with _JOBS_LOCK:
                _JOBS[jid]["status"] = "done"
                _JOBS[jid]["progress"] = 100
                _JOBS[jid]["result"] = res
        except Exception as e:
            with _JOBS_LOCK:
                _JOBS[jid]["status"] = "error"
                _JOBS[jid]["error"] = str(e)
    t = _threading.Thread(target=_run, daemon=True)
    t.start()
    return jid

def get_job(jid):
    with _JOBS_LOCK:
        return _JOBS.get(jid)

def get_mongo():
    if not (_MONGO and MONGO_URI):
        return None
    return pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=12000)

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
    client.close()
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
    frame_dir = request.form.get("frame_dir", "") or FRAME_DIR
    # fallback to bundled sample frames so ML works on hosted Render
    if not frame_dir or not os.path.isdir(frame_dir):
        bundled = ROOT / "static" / "sample_frames"
        if bundled.is_dir():
            frame_dir = str(bundled)
    if not frame_dir or not os.path.isdir(frame_dir):
        return jsonify({"ok": False, "error":
                        "FRAME_DIR not set / invalid on this host. Provide frame_dir."}), 400
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

@app.route("/frame/<int:sec>")
def frame_sec(sec):
    """Serve the 1-second frame JPG for a given second (0-based)."""
    if FRAME_DIR and os.path.isdir(FRAME_DIR):
        cand = Path(FRAME_DIR) / f"f_{sec + 1:04d}.jpg"
    else:
        cand = ROOT / "static" / "sample_frames" / f"f_{sec + 1:04d}.jpg"
    if not cand.exists():
        return jsonify({"ok": False, "error": f"frame f_{sec + 1:04d}.jpg not found"}), 404
    return send_file(cand, mimetype="image/jpeg", conditional=True)

@app.route("/frames/<int:start>/<int:end>")
def frames_range(start, end):
    """List the per-second frames covering one event (sec range inclusive)."""
    secs = list(range(max(0, start), max(start, end) + 1))
    step = 1 if len(secs) <= 60 else max(1, round(len(secs) / 60))
    out = []
    for s in secs[::step]:
        if FRAME_DIR and os.path.isdir(FRAME_DIR):
            p = Path(FRAME_DIR) / f"f_{s + 1:04d}.jpg"
        else:
            p = ROOT / "static" / "sample_frames" / f"f_{s + 1:04d}.jpg"
        if not p.exists():
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
    # annotate cached clip existence for each event
    annotated = []
    for r in rows:
        r = dict(r)
        start = r["time_frame"]["start_sec"]; end = r["time_frame"]["end_sec"]
        r["_clip_url"] = url_for("event_clip", start=start, end=end)
        r["_clip_cached"] = _clip_cache_file(start, end).exists()
        annotated.append(r)
    video_ok = os.path.exists(VIDEO_PATH)
    return render_template("activities.html", rows=annotated,
                           activities=[a["id"] for a in proc.cfg["activities"]],
                           colors={a["id"]: a["color"] for a in proc.cfg["activities"]},
                           video_ok=video_ok, video_path=VIDEO_PATH)

def _clip_cache_file(start, end):
    return CLIP_DIR / f"clip_{start}_{end}.mp4"

@app.route("/clip/<int:start>/<int:end>")
def event_clip(start, end):
    """Stream (or lazily generate + cache) the video segment covering an event."""
    if not video._find_ffmpeg():
        return jsonify({"ok": False, "error": "ffmpeg not available on this host."}), 400
    cache = _clip_cache_file(start, end)
    try:
        src = os.environ.get("VIDEO_SOURCE", VIDEO_PATH)
        video.cut_clip(src, start, max(1, end - start + 1), cache,
                       max_width=960, cap_duration=30)
    except Exception as e:
        return jsonify({"ok": False, "error": f"clip failed: {e}"}), 500
    return send_file(cache, mimetype="video/mp4", conditional=True)

@app.route("/api/jobs/<jid>")
def job_status(jid):
    j = get_job(jid)
    if j is None:
        return jsonify({"ok": False, "error": "unknown job"}), 404
    return jsonify(j)

@app.route("/ml/retrain", methods=["POST"])
def ml_retrain():
    """Background retrain of the ML model from the labelled frames."""
    frame_dir = request.form.get("frame_dir", "") or FRAME_DIR
    if not frame_dir or not os.path.isdir(frame_dir):
        return jsonify({"ok": False, "error":
                        "FRAME_DIR not set on this host; provide frame_dir."}), 400
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
    """Background re-prediction of the whole video + auto-log to Mongo."""
    frame_dir = request.form.get("frame_dir", "") or FRAME_DIR
    if not frame_dir or not os.path.isdir(frame_dir):
        return jsonify({"ok": False, "error":
                        "FRAME_DIR not set on this host; provide frame_dir."}), 400
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
        finally:
            client.close()
    return jsonify({"status": "ok", "model": MODEL_PATH.exists(), "mongo": mongo_info})

# ------------------------------------------------------------------ main -- #
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)