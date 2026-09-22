"""Generate a clip for EVERY unique duty event and upload it to Backblaze B2.

Resumable: skips keys already present in B2 and clips already generated on disk.
Naming matches the webapp lookup: clips/clip_<start>_<end>.mp4 (960px, <=30s).
Usage:  python scripts/b2_gen_all_clips.py  [--workers N] [--dry]
"""
import argparse, json, os, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]

KEY_ID = os.environ.get("B2_KEY_ID")
APP_KEY = os.environ.get("B2_APPLICATION_KEY")
BUCKET = os.environ.get("B2_BUCKET", "ksrtc-wpm-assets")
VIDEO = os.environ.get("VIDEO_SOURCE",
                       os.environ.get("VIDEO_PATH", r"D:\ksrtc\VID_20260819_135332150.mp4"))
RANGES = ROOT / "data" / "event_clip_ranges.json"
TMP = ROOT / "static" / "clips"
FFMPEG = r"D:\ksrtc\ffbin\ffmpeg-9.0-full_build\bin\ffmpeg.exe"

_lock = threading.Lock()
_done, _fail = 0, 0


def get_bucket():
    import b2sdk.v2 as b2
    info = b2.InMemoryAccountInfo()
    api = b2.B2Api(info)
    api.authorize_account("production", KEY_ID, APP_KEY)
    return api.get_bucket_by_name(BUCKET), b2


def existing_keys(bucket):
    keys = set()
    for (v, _) in bucket.ls(folder_to_list="clips/"):
        keys.add(v.file_name)
    print(f"[b2] already in bucket: {len(keys)} keys")
    return keys


def gen_clip(start, end, out):
    dur = min(end - start + 1, 30)
    cmd = [FFMPEG, "-y", "-loglevel", "error",
           "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
           "-i", VIDEO, "-vf", "scale=960:-2",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
           "-an", "-movflags", "+faststart", str(out)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg {p.stderr[-300:] if p.stderr else 'empty'}")


def worker(bucket, item):
    global _done, _fail
    key = f"clips/clip_{item['start']}_{item['end']}.mp4"
    out = TMP / f"clip_{item['start']}_{item['end']}.mp4"
    try:
        if not (out.exists() and out.stat().st_size > 0):
            gen_clip(item["start"], item["end"], out)
        bucket.upload_local_file(local_file=str(out), file_name=key)
        with _lock:
            _done += 1
            if _done % 50 == 0:
                print(f"[clip] {_done} uploaded, {_fail} failed, "
                      f"{int(time.time()) % 60}s", flush=True)
    except Exception as e:
        with _lock:
            _fail += 1
            if _fail <= 10:
                print(f"[fail] {key}: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="only process the first LIMIT items (for testing)")
    a = ap.parse_args()

    if not (KEY_ID and APP_KEY):
        sys.exit("set B2_KEY_ID and B2_APPLICATION_KEY env vars")
    if not os.path.exists(VIDEO):
        sys.exit(f"source video not found: {VIDEO}")
    items = [{"start": r["start"], "end": r["end"]}
             for r in json.loads(RANGES.read_text(encoding="utf-8"))]
    print(f"[gen] {len(items)} unique event ranges, source={VIDEO}")

    bucket, _ = get_bucket()
    if a.dry:
        print(f"[dry] would generate/upload {len(items)} clips "
              f"({sum(i['end'] - i['start'] + 1 for i in items)} clip-sec)")
        return
    have = existing_keys(bucket)
    todo = [i for i in items
            if f"clips/clip_{i['start']}_{i['end']}.mp4" not in have]
    if a.limit:
        todo = todo[:a.limit]
    print(f"[gen] generating+uploading {len(todo)} remaining clips "
          f"({sum(i['end'] - i['start'] + 1 for i in todo)} clip-sec)")
    if not todo:
        print("all clips already in B2")
        return
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(lambda i: worker(bucket, i), todo))
    print(f"[gen] DONE uploaded={_done} failed={_fail}")


if __name__ == "__main__":
    main()