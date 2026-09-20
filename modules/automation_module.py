"""WPM Module 4: AUTOMATION -- machine-learning classification of frames into
activities, automatic creation of duty logs and automatic VA/NVA discovery
(paper's AUTOMATION module, Phase II above the human analytics).

Pipeline:
  1. build_dataset()    - extract per-frame features + labels (from event log)
  2. train()            - fit Random Forest, persist model + report
  3. predict_frames()   - classify every frame of a folder/video
  4. auto_log()         - turn predictions into duty-log documents in MongoDB
  5. auto_report()      - VA/NVA + activity analytics JSON for the dashboard

Features per frame (small & robust on embedded/cloud hardware):
  * 48x48 grayscale downsampled pixels                (2304)
  * HSV histogram (12x12x8 won't fit; use 16x16x8)     (2048)
  * primary/secondary zone Canny edge energy           (2)
  * Laplacian variance (sharpness)                     (1)
  Total ~4355 floats per frame.
"""
import json, os
from pathlib import Path

import cv2
import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from joblib import dump, load
    from sklearn.preprocessing import LabelEncoder
    _SK = True
except Exception:
    _SK = False


class AutomationModule:
    FEAT_H, FEAT_W = 48, 48
    H_BINS, S_BINS, V_BINS = 16, 16, 8

    def __init__(self, cfg, proc_module, root="."):
        self.cfg = cfg
        self.proc = proc_module
        self.root = Path(root)

    # -------------------------------------------------------------------- #
    # feature extraction                                                   #
    # -------------------------------------------------------------------- #
    @staticmethod
    def _img_feature(img):
        if img is None:
            raise ValueError("null frame")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (AutomationModule.FEAT_W, AutomationModule.FEAT_H),
                           interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None,
                            [AutomationModule.H_BINS, AutomationModule.S_BINS,
                             AutomationModule.V_BINS],
                            [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        hist = hist.flatten().astype(np.float32)
        canny = cv2.Canny(gray, 60, 160)
        h, w = canny.shape[:2]
        ch, cw = h // 3, w // 4
        z1 = canny[0:ch*1, 3*cw:4*cw].mean()      # primary zone (bottom-right)
        z2 = canny[ch*1:ch*2, 2*cw:3*cw].mean() if h >= ch*2 and w >= cw*3 else 0.0
        lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return np.concatenate([small.ravel(), hist, [z1, z2, lap]])

    def frame_feature(self, sec, frame_dir):
        p = Path(frame_dir) / f"f_{sec + 1:04d}.jpg"
        img = cv2.imread(str(p))
        if img is None:
            raise RuntimeError(f"cannot read {p}")
        return self._img_feature(img)

    # -------------------------------------------------------------------- #
    # dataset building: frames of a folder + labels from an event log      #
    # -------------------------------------------------------------------- #
    def build_dataset(self, frame_dir, event_log_path=None, sec_labels=None):
        """Returns X (n_samples x feats), y (strings), secs (list[int])."""
        files = sorted(Path(frame_dir).glob("f_*.jpg"))
        if not files:
            raise RuntimeError(f"no frames in {frame_dir}")
        X, secs = [], []
        for i, p in enumerate(files):
            img = cv2.imread(str(p))
            if img is None:
                continue
            X.append(self._img_feature(img))
            secs.append(i)               # 0-based == idx-1 matching analysis module
        # labels
        if sec_labels is None:
            sec_labels = self._labels_from_csv(event_log_path, len(files))
        y = [sec_labels.get(s, "WAIT") for s in secs]
        return np.asarray(X, dtype=np.float32), np.asarray(y), np.asarray(secs)

    @staticmethod
    def _labels_from_csv(path, n):
        """event log rows {start,end,duration,activity,..} -> {sec: activity}"""
        labels = {}
        with open(path, newline="") as f:
            import csv
            for r in csv.DictReader(f):
                for s in range(int(r["start"]), int(r["end"]) + 1):
                    labels[s] = r["activity"]
        return labels

    # -------------------------------------------------------------------- #
    # training + persistence                                               #
    # -------------------------------------------------------------------- #
    def train(self, X, y, model_path, test_size=0.2, n_estimators=300):
        if not _SK:
            raise RuntimeError("scikit-learn not installed")
        X = np.asarray(X, dtype=np.float32)
        y_enc = LabelEncoder().fit(y)
        y_num = y_enc.transform(y)
        Xtr, Xte, ytr, yte = train_test_split(X, y_num, test_size=test_size,
                                              stratify=y_num, random_state=42)
        clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1,
                                     random_state=42, class_weight="balanced")
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        acc = float(accuracy_score(yte, pred))
        cm = confusion_matrix(yte, pred).tolist()
        classes = list(y_enc.classes_)
        report = classification_report(yte, pred, target_names=classes,
                                       output_dict=True, zero_division=0)
        # trim report to per-class metrics
        report = {k: v for k, v in report.items() if k in classes or k in ("accuracy",)}
        payload = {
            "classes": classes,
            "test_size": float(test_size),
            "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
            "accuracy": acc,
            "confusion_matrix": cm,
            "report": report,
            "feature_importance": np.asarray(clf.feature_importances_, dtype=np.float64).tolist(),
        }
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        import io, zlib
        buf = io.BytesIO()
        dump({"clf": clf, "le": classes}, buf)
        Path(model_path).write_bytes(zlib.compress(buf.getvalue(), 9))
        return payload

    def load_model(self, model_path):
        if not _SK or not Path(model_path).exists():
            return None
        import io, zlib
        return load(io.BytesIO(zlib.decompress(Path(model_path).read_bytes())))

    # -------------------------------------------------------------------- #
    # inference                                                            #
    # -------------------------------------------------------------------- #
    def predict_frames(self, frame_dir, model_path, start_sec=0, max_frames=None):
        model = self.load_model(model_path)
        if model is None:
            raise RuntimeError("model not found; run train first")
        clf, classes = model["clf"], model["le"]
        Le = LabelEncoder(); Le.classes_ = np.asarray(classes)
        files = sorted(Path(frame_dir).glob("f_*.jpg"))
        if max_frames:
            files = files[:max_frames]
        secs, preds = [], []
        for p in files:
            img = cv2.imread(str(p))
            if img is None:
                continue
            feats = self._img_feature(img).reshape(1, -1)
            lab = Le.inverse_transform(clf.predict(feats))[0]
            secs.append(int(p.stem.split("_")[1]) - 1 + start_sec)
            preds.append(str(lab))
        return secs, preds

    # -------------------------------------------------------------------- #
    # events + auto logging / reporting (the "automatically logs & finds   #
    # VA/NVA" part of the requirement)                                     #
    # -------------------------------------------------------------------- #
    @staticmethod
    def predictions_to_events(secs, preds, start_sec=0):
        events, cur = [], None
        for s, lab in zip(secs, preds):
            if cur and cur["activity"] == lab:
                cur["end"] = s; cur["duration"] += 1
            else:
                if cur:
                    events.append(cur)
                cur = {"start": s, "end": s, "duration": 1, "activity": lab}
        if cur:
            events.append(cur)
        return events

    def auto_log(self, events, mongo_uri, source="ML_AUTO", collection=None):
        import datetime
        try:
            import pymongo
        except Exception:
            return {"inserted": 0, "error": "pymongo missing"}
        collection = collection or self.cfg["db"]["collections"]["duty_logs"]
        client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=15000)
        db = client[self.cfg["db"]["name"]]
        docs = []
        for e in events:
            docs.append({
                "time_frame": {
                    "start_sec": e["start"],
                    "start_ts": self._ts(e["start"]),
                    "end_sec": e["end"],
                    "end_ts": self._ts(e["end"]),
                    "duration_s": e["duration"],
                },
                "activity": e["activity"],
                "activity_color": self.proc.color(e["activity"]),
                "va_nva": "VA" if self.proc.is_va(e["activity"]) else "NVA",
                "source": source,
                "created_by": "AUTOMATION-ML",
                "created_at": datetime.datetime.utcnow(),
            })
        res = db[collection].insert_many(docs, ordered=False) if docs else None
        client.close()
        return {"inserted": len(res.inserted_ids) if res else 0}

    def find_va_nva(self, events):
        total = sum(e["duration"] for e in events)
        va = sum(e["duration"] for e in events if self.proc.is_va(e["activity"]))
        nva = total - va
        return {"VA": {"sec": va, "pct": round(100*va/total, 1) if total else 0,
                       "events": sum(1 for e in events if self.proc.is_va(e["activity"]))},
                "NVA": {"sec": nva, "pct": round(100*nva/total, 1) if total else 0,
                        "events": sum(1 for e in events if not self.proc.is_va(e["activity"]))}}

    @staticmethod
    def _ts(sec):
        m, s = divmod(int(sec), 60); h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"