"""Upload WPM frames + generated clips to Backblaze B2.
Credentials MUST come from env: B2_KEY_ID, B2_APPLICATION_KEY (never hardcode).
Layout in bucket 'ksrtc-wpm-assets':
  frames/f_0001.jpg ... f_3951.jpg
  clips/clip_<start>_<end>.mp4
Usage:  python scripts/b2_upload.py [--frames DIR] [--clips DIR] [--bucket NAME]
"""
import argparse, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
PARENT = Path(__file__).resolve().parents[1]

KEY_ID = os.environ.get("B2_KEY_ID")
APP_KEY = os.environ.get("B2_APPLICATION_KEY")
BUCKET = os.environ.get("B2_BUCKET", "ksrtc-wpm-assets")


def get_api():
    import b2sdk.v2 as b2
    info = b2.InMemoryAccountInfo()
    api = b2.B2Api(info)
    api.authorize_account("production", KEY_ID, APP_KEY)
    return api, b2


def existing_keys(bucket, prefix):
    keys = set()
    for (v, _) in bucket.ls(folder_to_list=prefix):
        keys.add(v.file_name)
    return keys


def upload_many(api, b2, bucket, local_dir, prefix, ext, max_workers=12):
    from concurrent.futures import ThreadPoolExecutor
    local_dir = Path(local_dir)
    files = sorted(local_dir.glob(f"*.{ext}"))
    total = len(files)
    if not total:
        print(f"[{prefix}] no {ext} files in {local_dir}")
        return 0, 0
    seen = existing_keys(bucket, prefix)
    todo = [(f"{prefix}/{p.name}", str(p)) for p in files
            if f"{prefix}/{p.name}" not in seen]
    done, skipped = 0, total - len(todo)
    t0 = time.time()
    lock = __import__("threading").Lock()

    def _one(kp):
        key, path = kp
        bucket.upload_local_file(local_file=path, file_name=key)
        with lock:
            nonlocal done
            done += 1
            if done % 250 == 0:
                el = time.time() - t0
                print(f"[{prefix}] {done}/{len(todo)} uploaded ({el:.0f}s)")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(_one, todo))
    print(f"[{prefix}] done: uploaded={done} skipped={skipped} total={total}")
    return done, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=str(PARENT / ".." / "frames_1s"))
    ap.add_argument("--clips", default=str(PARENT / "static" / "clips"))
    ap.add_argument("--bucket", default=BUCKET)
    a = ap.parse_args()

    if not (KEY_ID and APP_KEY):
        sys.exit("set B2_KEY_ID and B2_APPLICATION_KEY env vars")

    api, b2 = get_api()
    bucket = api.get_bucket_by_name(a.bucket)
    print("bucket:", a.bucket, bucket.id_)

    if os.path.isdir(a.frames):
        upload_many(api, b2, bucket, a.frames, "frames", "jpg")
    else:
        print("skip frames:", a.frames, "not found")

    if os.path.isdir(a.clips):
        upload_many(api, b2, bucket, a.clips, "clips", "mp4")
    else:
        print("skip clips:", a.clips, "not found")


if __name__ == "__main__":
    main()