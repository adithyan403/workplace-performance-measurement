"""Train the WPM automation model from the frame dataset.

Run from project root (requires local frame folder + event log csv):
    D:/python.exe scripts/train_model.py

Outputs:
  data/X_features.npz   - compact feature matrix + labels + sec mapping
  data/model.joblib     - trained RandomForest classifier (loaded by the web app)
  data/training_report.json - accuracy / confusion matrix / per-class report
"""
import json, sys, os, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from modules.process_module import ProcessModule
from modules.automation_module import AutomationModule

def main():
    cfg_file = ROOT / "config" / "process_config.json"
    frame_dir = sys.argv[1] if len(sys.argv) > 1 else r"D:\ksrtc\frames_1s"
    event_log = (sys.argv[2] if len(sys.argv) > 2 else
                 r"D:\ksrtc\wpm_study\mod_analysis\wpm_event_log.csv")
    out_npz = ROOT / "data" / "X_features.npz"
    out_model = ROOT / "data" / "model.joblib"
    out_report = ROOT / "data" / "training_report.json"

    proc = ProcessModule(cfg_file, ROOT)
    auto = AutomationModule(proc.cfg, proc, ROOT)

    print("=== WPM AUTOMATION | dataset build + training ===")
    t0 = time.time()
    X, y, secs = auto.build_dataset(frame_dir, event_log)
    print(f"Dataset: {X.shape[0]} samples x {X.shape[1]} features "
          f"({time.time()-t0:.1f}s, labels {sorted(set(y))})")

    np.savez_compressed(out_npz, X=X, y=y, secs=secs)
    print(f"Saved features: {out_npz} ({out_npz.stat().st_size/1e6:.1f} MB)")

    report = auto.train(X, y, out_model)
    print("Accuracy:", round(report["accuracy"], 4))
    print("Confusion matrix rows (true):", report["classes"])
    for row in report["confusion_matrix"]:
        print("  ", row)
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Model + report saved -> {out_model}")

if __name__ == "__main__":
    main()