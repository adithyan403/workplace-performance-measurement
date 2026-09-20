"""Prepare static data files for the web app (dashboard fallback + analysis).

Reads Phase-I outputs and writes compact JSON into <root>/data:
  * event_log_sample.json       - full event log (fallback when Mongo absent)
  * metrics_perSecond.json      - downsampled per-second metrics for charts
  * training_report.json        - already produced by train_model.py
"""
import csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRCS = {
    "events": r"D:\ksrtc\wpm_study\mod_analysis\wpm_event_log.csv",
    "metrics": r"D:\ksrtc\wpm_study\mod_analysis\wpm_metrics_perSecond.csv",
}

def main():
    out = ROOT / "data"; out.mkdir(parents=True, exist_ok=True)

    events = []
    with open(SRCS["events"], newline="") as f:
        for r in csv.DictReader(f):
            events.append({
                "time_frame": {"start_sec": int(r["start"]), "end_sec": int(r["end"]),
                               "duration_s": int(r["duration"]),
                               "start_ts": _ts(int(r["start"])),
                               "end_ts": _ts(int(r["end"]))},
                "activity": r["activity"], "va_nva": r["VA_NVA"],
                "source": "WPM_PHASE1", "created_by": "ANALYSIS",
            })
    (out / "event_log_sample.json").write_text(json.dumps(events, ensure_ascii=False), encoding="utf-8")

    metrics = []
    with open(SRCS["metrics"], newline="") as f:
        for r in csv.DictReader(f):
            metrics.append({"sec": int(r["sec"]), "ts": r["ts"],
                            "global_diff": float(r["global_diff"]),
                            "cell_primary": float(r["cell_primary"]),
                            "cell_secondary": float(r["cell_secondary"]),
                            "hist_change": float(r["hist_change"]),
                            "activity": r["activity"]})
    (out / "metrics_perSecond.json").write_text(json.dumps(metrics, ensure_ascii=False), encoding="utf-8")

    print(f"events -> data/event_log_sample.json  ({len(events)} rows)")
    print(f"metrics -> data/metrics_perSecond.json  ({len(metrics)} rows)")

def _ts(sec):
    m, s = divmod(int(sec), 60); h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

if __name__ == "__main__":
    main()