"""Interactive frame labeler for the KSRTC WPM CNN training set.

Shows each frame one at a time; clicking a class button (or pressing 1-5)
labels the frame, copies it into frames_labeled/<class>/f_XXXX.jpg and
automatically advances to the next unlabeled frame.

Usage:
    python label_frames.py                # defaults below
    python label_frames.py --frames DIR   # alternate source
    python label_frames.py --out DIR      # alternate output
    python label_frames.py --move         # MOVE instead of copy (consumes source)
"""
import argparse
import json
import shutil
import sys
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk

DEFAULT_FRAMES = Path(r"D:\ksrtc\frames clear")
DEFAULT_OUT = Path(r"D:\ksrtc\frames_labeled")
CLASSES = ["write", "crew", "check", "wait"]
INDEX_NAME = "labels.json"


def load_index(out_dir: Path):
    index = {}
    path = out_dir / INDEX_NAME
    if path.exists():
        try:
            index = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            index = {}
    return index


class LabelApp:
    def __init__(self, root, frames_dir, out_dir, move=False):
        self.root = root
        self.root.title("Frame Labeler - KSRTC WPM CNN")
        self.frames = sorted(p for p in Path(frames_dir).glob("*.jpg"))
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        for c in CLASSES:
            (self.out / c).mkdir(parents=True, exist_ok=True)
        self.move = move
        self.index = load_index(self.out)
        self.frames = [f for f in self.frames if f.name not in self.index]
        self.pos = 0
        self.total = len(self.frames)

        self.root.geometry("1600x1000")
        self.root.minsize(1100, 700)
        self.root.bind("<F11>", lambda e: self.toggle_fullscreen())
        self.root.bind("<Escape>", lambda e: self.toggle_fullscreen())

        header = tk.Frame(self.root, bg="#0f172a")
        header.pack(fill="x")
        self.progress_lbl = tk.Label(header, text="", font=("Segoe UI", 12),
                                     fg="white", bg="#0f172a", anchor="w")
        self.progress_lbl.pack(side="left", padx=14, pady=8)
        self.hint_lbl = tk.Label(header, text="1-5 label \u00b7 Space next \u00b7 Backspace undo \u00b7 F11 fullscreen \u00b7 Q quit",
                                 font=("Segoe UI", 10), fg="#94a3b8", bg="#0f172a", anchor="e")
        self.hint_lbl.pack(side="right", padx=14, pady=8)

        self.canvas = tk.Canvas(self.root, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.redraw())

        colors = {
            "write": "#22c55e",
            "check": "#f59e0b",
            "crew": "#ef4444",
            "wait": "#64748b",
        }
        btn_frame = tk.Frame(self.root, bg="#f1f5f9")
        btn_frame.pack(fill="x", pady=10)
        self.buttons = {}
        for i, c in enumerate(CLASSES, start=1):
            b = tk.Button(btn_frame, text=f"{i}  {c.replace('_', ' ').title()}",
                          font=("Segoe UI", 13, "bold"), bg=colors[c], fg="white",
                          activebackground=colors[c], relief="flat", cursor="hand2",
                          command=lambda cls=c: self.label(cls))
            b.pack(side="left", expand=True, fill="x", padx=5)
            self.buttons[c] = b
            self.root.bind(str(i), lambda e, cls=c: self.label(cls))

        undo = tk.Button(btn_frame, text="\u2190 Undo (Backspace)", font=("Segoe UI", 12),
                         bg="#e2e8f0", relief="flat", command=self.undo)
        undo.pack(side="left", padx=5)
        self.root.bind("<BackSpace>", lambda e: self.undo())
        self.root.bind("<space>", lambda e: self.next_frame())
        self.root.bind("<q>", lambda e: self.quit())
        self.root.bind("<Q>", lambda e: self.quit())
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

        if not self.frames:
            self._waiting = True
            self.show_message(f"All {len(self.index)} frames are already labeled. Nice work!")
        else:
            self._waiting = False
            self.show_frame()

    def show_message(self, msg):
        self.canvas.delete("all")
        self.canvas.create_text(self.canvas.winfo_width() // 2 or 600,
                                self.canvas.winfo_height() // 2 or 400,
                                text=msg, fill="white", font=("Segoe UI", 18))
        self.progress_lbl.config(text=msg)

    def current(self):
        return self.frames[self.pos] if 0 <= self.pos < len(self.frames) else None

    def show_frame(self):
        frame = self.current()
        if frame is None:
            self.show_message(f"Done! {len(self.index)} frames labeled.")
            return
        self.update_progress(frame)
        try:
            self.raw = Image.open(frame).convert("RGB")
        except Exception as e:
            print("open failed", frame, e)
            self.next_frame()
            return
        self.photo = None
        self.redraw()
        self.root.title(f"Frame Labeler - {frame.name} [{(self.pos + 1)}/{self.total} unlabeled]")

    def redraw(self):
        """Re-render the current frame at the canvas size. Called on show and
        on every window resize (<Configure>), so the frame always fills the
        available area."""
        if not hasattr(self, "raw") or self.raw is None:
            return
        self.root.update_idletasks()
        cw = max(self.canvas.winfo_width(), 200)
        ch = max(self.canvas.winfo_height(), 300)
        tw, th = self.raw.width, self.raw.height
        # fill the canvas: fit the larger dimension, allow upscaling
        scale = max(cw / tw, ch / th)
        self._img = self.raw.resize((max(1, int(tw * scale)), max(1, int(th * scale))),
                                    Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(self._img)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self.photo, anchor="center")

    def toggle_fullscreen(self):
        self.root.attributes("-fullscreen", not self.root.attributes("-fullscreen"))
        self.root.update_idletasks()
        self.redraw()

    def update_progress(self, frame):
        remaining = len(self.frames) - self.pos
        self.progress_lbl.config(
            text=f"Labeled: {len(self.index):4d}   Remaining: {remaining:4d}   "
                 f"[{frame.name}]")

    def label(self, cls):
        frame = self.current()
        if frame is None:
            return
        dest = self.out / cls / frame.name
        if self.move:
            shutil.move(str(frame), str(dest))
        else:
            shutil.copy2(str(frame), str(dest))
        self.index[frame.name] = cls
        self._save_index()
        self.next_frame()

    def next_frame(self):
        if not self.frames:
            return
        if self.pos >= len(self.frames) - 1:
            self.pos = len(self.frames)
            self.show_frame()
        else:
            self.pos += 1
            self.show_frame()

    def undo(self):
        if not self.frames or self.pos <= 0:
            return
        self.pos -= 1
        self.show_frame()
        frame = self.current()
        if frame and frame.name in self.index:
            old = self.index.pop(frame.name)
            self._save_index()
            to_del = self.out / old / frame.name
            if to_del.exists():
                try:
                    to_del.unlink()
                except OSError:
                    pass

    def _save_index(self):
        (self.out / INDEX_NAME).write_text(
            json.dumps(self.index, indent=0), encoding="utf-8")

    def quit(self):
        self._save_index()
        self.root.destroy()


def main():
    ap = argparse.ArgumentParser(description="Label KSRTC WPM frames for CNN training")
    ap.add_argument("--frames", default=str(DEFAULT_FRAMES), help="source frames dir")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="labeled output dir")
    ap.add_argument("--move", action="store_true", help="move instead of copy")
    args = ap.parse_args()
    root = tk.Tk()
    LabelApp(root, args.frames, args.out, move=args.move)
    root.mainloop()


if __name__ == "__main__":
    main()