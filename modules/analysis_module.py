"""WPM Module 3: ANALYSIS -- computes motion/zone metrics from digitalized
frames, classifies seconds into activities, assembles the event log and
produces the VA/NVA analytics (paper's P_Analysis_3 / process analytics).

Metrics per second (digital traces):
  global_diff   - abs frame-to-frame luminance difference (whole image)
  cell_*        - motion energy inside each workspace grid cell
  hist_change   - colour-histogram change between consecutive seconds
"""
import csv, json, os
from pathlib import Path

import cv2
import numpy as np


class AnalysisModule:
    def __init__(self, proc_module, frame_dir, rows=3, cols=4):
        self.proc = proc_module              # ProcessModule
        self.frame_dir = Path(frame_dir)
        self.rows, self.cols = rows, cols
        self._cache = {}

    # ------------------------------------------------------------------ #
    # metric computation over a directory of f_XXXX.jpg frames           #
    # ------------------------------------------------------------------ #
    def _load(self, idx):
        """frame index (1-based) -> (sec_index, bgr image)."""
        p = self.frame_dir / f"f_{idx:04d}.jpg"
        img = cv2.imread(str(p))
        if img is None:
            raise RuntimeError(f"cannot read {p}")
        return img

    def _motion(self, f1, f2):
        g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(g2, g1)
        return diff

    def compute_metrics(self, start_idx=1, end_idx=None, step=1, progress=None):
        """Single pass over frames; returns list of per-second dicts."""
        files = sorted(p for p in self.frame_dir.glob("f_*.jpg"))
        if not files:
            raise RuntimeError(f"no frames in {self.frame_dir}")
        if end_idx is None:
            end_idx = len(files)
        idxs = list(range(start_idx, end_idx + 1, step))
        out, prev = [], None
        for i, idx in enumerate(idxs):
            img = self._load(idx)
            metrics = {}
            if prev is not None:
                diff = self._motion(prev, img)
                metrics["global_diff"] = float(diff.sum())
                # grid cell motion energy
                h, w = diff.shape[:2]
                ch, cw = h // self.rows, w // self.cols
                cells = []
                for r in range(self.rows):
                    for c in range(self.cols):
                        cells.append(float(diff[r*ch:(r+1)*ch, c*cw:(c+1)*cw].sum()))
                metrics["cells"] = cells
                metrics["cell_" + ("primary" if False else "primary")] = cells[self.proc.cfg["zones"]["primary"]]
                metrics["cell_secondary"] = cells[self.proc.cfg["zones"]["secondary"]]
                # histogram change
                h1 = cv2.calcHist([cv2.cvtColor(prev, cv2.COLOR_BGR2HSV)], [0], None, [16], [0, 180])
                h2 = cv2.calcHist([cv2.cvtColor(img, cv2.COLOR_BGR2HSV)], [0], None, [16], [0, 180])
                cv2.normalize(h1, h1); cv2.normalize(h2, h2)
                metrics["hist_change"] = float(cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA))
            else:
                metrics = {"global_diff": 0.0, "cells": [0.0] * (self.rows * self.cols),
                           "cell_primary": 0.0, "cell_secondary": 0.0, "hist_change": 0.0}
            metrics["idx"] = idx
            metrics["sec"] = idx - 1
            out.append(metrics)
            prev = img
            if progress and i % max(1, len(idxs)//20) == 0:
                progress(i + 1, len(idxs))
        return out

    # ------------------------------------------------------------------ #
    # classification from thresholds (paper: matching to process params) #
    # ------------------------------------------------------------------ #
    def classify(self, metrics_list, thresholds=None, grid=None):
        t = thresholds or self.proc.cfg["thresholds"]
        if grid is None:
            rows, cols = self.rows, self.cols
        else:
            rows, cols = grid
        ops_kind = []
        for m in metrics_list:
            gd = m["global_diff"]
            cp = m.get("cells", [0.0]*rows*cols) or [0.0]*rows*cols
            prim = cp[self.proc.cfg["zones"]["primary"]]
            if gd > t["p_hi"]:
                act = "CREW"
            elif prim > t["p_cell"] * 2.2 or gd > t["p_hi"] * 0.62:
                act = "WRITE"
            elif gd > t["p_lo"]:
                act = "CHECK"
            elif prim > t["p_cell"] * 1.1 or m["hist_change"] > t["p_hist"] * 1.6:
                act = "PAGE"
            else:
                act = "WAIT"
            ops_kind.append(act)
        return ops_kind

    # ------------------------------------------------------------------ #
    # event log assembly (paper: event log with start/end timestamps)    #
    # ------------------------------------------------------------------ #
    def build_event_log(self, metrics_list, labels):
        events, cur = [], None
        for m, lab in zip(metrics_list, labels):
            sec = m["sec"]
            if cur and cur["activity"] == lab:
                cur["end"] = sec; cur["duration"] += 1
            else:
                if cur:
                    events.append(cur)
                cur = {"start": sec, "end": sec, "duration": 1, "activity": lab}
        if cur:
            events.append(cur)
        for e in events:
            e["VA_NVA"] = "VA" if self.proc.is_va(e["activity"]) else "NVA"
        return events

    @staticmethod
    def _ts(sec):
        m, s = divmod(int(sec), 60); h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    # ------------------------------------------------------------------ #
    # analytics / reporting payloads                                     #
    # ------------------------------------------------------------------ #
    def summarize(self, events, total_sec, metrics_list=None):
        import collections
        act = collections.Counter(e["activity"] for e in events)
        dist = {}
        for e in events:
            d = dist.setdefault(e["activity"], {"count": 0, "sec": 0})
            d["count"] += 1; d["sec"] += e["duration"]
        va_sec = sum(d["sec"] for a, d in dist.items() if self.proc.is_va(a))
        nva_sec = total_sec - va_sec
        return {
            "event_count": len(events),
            "total_sec": total_sec,
            "activity_distribution": {a: {"count": d["count"], "sec": d["sec"],
                                           "pct": round(100*d["sec"]/total_sec, 1)} for a, d in dist.items()},
            "va_nva": {"VA": {"sec": va_sec, "pct": round(100*va_sec/total_sec, 1),
                              "count": sum(d["count"] for a, d in dist.items() if self.proc.is_va(a))},
                       "NVA": {"sec": nva_sec, "pct": round(100*nva_sec/total_sec, 1),
                               "count": sum(d["count"] for a, d in dist.items() if not self.proc.is_va(a))}},
        }

    def save_csv(self, events, path):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["start", "end", "duration", "activity", "VA_NVA"])
            w.writeheader(); w.writerows(events)
        return path

    def load_csv(self, path):
        with open(path, newline="") as f:
            return list(csv.DictReader(f))