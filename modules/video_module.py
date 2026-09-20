"""WPM Module 1: VIDEO -- source video acquisition and digitalization to frames.

After the paper: this module selects the camera/source signal, registers it
in the database (P_Video_1) and digitalizes it into analysable frames at a
configurable sampling rate. It never classifies content; it only extracts
time-stamped digital traces.
"""
import os, shutil, subprocess
from pathlib import Path

import cv2

try:
    import pymongo
    import bson.binary as bb
except Exception:  # pragma: no cover
    pymongo = None


class VideoModule:
    def __init__(self, cfg, root="."):
        self.cfg = cfg
        self.root = Path(root)

    # ------------------------------------------------------------------ #
    # source registration (paper: the P_Video_1 source endpoint)         #
    # ------------------------------------------------------------------ #
    def register_source(self, uri, mongo_uri=None):
        """Register the video source in the WPM database (collection duty_logs
        keeps the dummy source document; the real payload is in video_sources)."""
        if pymongo is None or not mongo_uri:
            return {"registered": False, "reason": "no mongo_uri"}
        client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=15000)
        db = client[self.cfg["db"]["name"]]
        db["video_sources"].update_one(
            {"uri": uri},
            {"$set": {"uri": uri, "registered_at": __import__("datetime").datetime.utcnow()}},
            upsert=True)
        client.close()
        return {"registered": True, "uri": uri}

    # ------------------------------------------------------------------ #
    # digitalization: video -> time-stamped frames                       #
    # ------------------------------------------------------------------ #
    def probe(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return {"fps": round(fps, 3), "frames": n, "width": w, "height": h,
                "seconds": round(n / fps, 2) if fps else None, "path": str(video_path)}

    def extract_frames(self, video_path, out_dir, interval_sec=1.0, max_width=None,
                       start_sec=0, end_sec=None):
        """Sample one frame every `interval_sec` seconds into <out_dir>/f_XXXX.jpg.
        Fast path: uses ffmpeg when available; else cv2 (slower, dependable)."""
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        for f in out_dir.glob("f_*.jpg"):
            f.unlink()

        ffmpeg = self._find_ffmpeg()
        if ffmpeg:
            return self._extract_ffmpeg(str(ffmpeg), video_path, out_dir,
                                        interval_sec, max_width, start_sec, end_sec)
        return self._extract_cv2(video_path, out_dir, interval_sec, max_width,
                                 start_sec, end_sec)

    def _find_ffmpeg(self):
        for cand in (os.environ.get("FFMPEG"),
                     r"D:\ksrtc\ffbin\ffmpeg-9.0-full_build\bin\ffmpeg.exe",
                     "ffmpeg"):
            if cand and os.path.exists(cand):
                return cand
        return None

    def _extract_ffmpeg(self, ffmpeg, video, out_dir, interval_sec, max_width,
                        start_sec, end_sec):
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(video),
               "-vf", f"fps=1/{interval_sec}"]
        if max_width:
            cmd[-1] = f"fps=1/{interval_sec},scale={max_width}:-2"
        cmd += ["-q:v", "3", str(out_dir / "f_%04d.jpg")]
        subprocess.run(cmd, check=True)
        files = sorted(out_dir.glob("f_*.jpg"))
        return self._frames_meta(files, offset=start_sec)

    def _extract_cv2(self, video, out_dir, interval_sec, max_width, start_sec, end_sec):
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        step = max(1, int(round(fps * interval_sec)))
        frame_i = 0
        files = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            sec = frame_i / fps
            if sec + interval_sec >= (start_sec or 0) and (end_sec is None or sec <= end_sec):
                if frame_i % step == 0 and sec >= (start_sec or 0) - 0.5:
                    fname = out_dir / f"f_{int(sec // interval_sec) + 1:04d}.jpg"
                    if max_width and frame.shape[1] > max_width:
                        r = max_width / frame.shape[1]
                        frame = cv2.resize(frame, (max_width, int(frame.shape[0] * r)))
                    if cv2.imwrite(str(fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 88]):
                        files.append(fname)
            frame_i += 1
        cap.release()
        return self._frames_meta(sorted(files), offset=start_sec or 0)

    def _frames_meta(self, files, offset=0):
        return [{"idx": i + 1, "sec": offset + i,
                 "ts": self._ts(offset + i), "path": str(f)} for i, f in enumerate(files)]

    @staticmethod
    def _ts(sec):
        m, s = divmod(int(sec), 60); h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def read_frame(self, sec, frame_dir):
        p = Path(frame_dir) / f"f_{sec + 1:04d}.jpg"
        if not p.exists():
            raise FileNotFoundError(p)
        img = cv2.imread(str(p))
        if img is None:
            raise RuntimeError(f"decode failed {p}")
        return img