"""Backblaze B2 access for WPM webapp assets (frames + video clips).

No secrets are stored in this file. Everything comes from env vars:
  B2_KEY_ID, B2_APPLICATION_KEY, B2_BUCKET (optional, default ksrtc-wpm-assets)

The bucket layout mirrors local disk:
  frames/f_0001.jpg ... f_3951.jpg
  clips/clip_<start>_<end>.mp4
"""
import os, threading, time
from pathlib import Path

_DEFAULTS = {
    "bucket": "ksrtc-wpm-assets",
    "key_id": "",
    "application_key": "",
}

def available():
    return bool(os.environ.get("B2_KEY_ID") and os.environ.get("B2_APPLICATION_KEY"))


def _extract_zip(zip_local, frames_dir, log=print):
    import zipfile
    with zipfile.ZipFile(zip_local) as z:
        z.extractall(frames_dir)
    log(f"[b2.frames] extracted from zip: {len(list(frames_dir.glob('f_*.jpg')))} frames")


class B2Module:
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._api = None
        self._bucket = None
        self._lock = threading.Lock()
        self.enabled = available()

    # ------------------------------------------------------------------ #
    def _connect(self):
        if self._api is not None:
            return self._bucket
        with self._lock:
            if self._api is not None:
                return self._bucket
            import b2sdk.v2 as b2
            info = b2.InMemoryAccountInfo()
            api = b2.B2Api(info)
            api.authorize_account("production",
                                  os.environ.get("B2_KEY_ID"),
                                  os.environ.get("B2_APPLICATION_KEY"))
            self._api = api
            self._bucket = api.get_bucket_by_name(
                os.environ.get("B2_BUCKET", _DEFAULTS["bucket"]))
            return self._bucket

    def bucket_name(self):
        return os.environ.get("B2_BUCKET", _DEFAULTS["bucket"])

    # ------------------------------------------------------------------ #
    # small object in / out: streams into a local cache file             #
    # ------------------------------------------------------------------ #
    def download(self, key, refresh=False, raise_on_error=False):
        """Fetch key from B2 into cache_dir; return local path (or None)."""
        bucket = self._connect()
        dest = self.cache_dir / Path(key).name
        if dest.exists() and dest.stat().st_size > 0 and not refresh:
            return str(dest)
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            bucket.download_file_by_name(key).save_to(str(tmp))
        except Exception as e:
            if raise_on_error:
                raise RuntimeError(f"B2 download failed for {key}: {e}")
            return None
        tmp.replace(dest)
        return str(dest)

    def upload(self, key, local_path):
        bucket = self._connect()
        bucket.upload_local_file(local_file=str(local_path), file_name=key)

    def exists(self, key):
        bucket = self._connect()
        try:
            bucket.get_file_info_by_name(key)
            return True
        except Exception:
            return False

    def list_keys(self, prefix, limit=None):
        bucket = self._connect()
        keys, it = [], bucket.ls(folder_to_list=prefix)
        for (v, _) in it:
            keys.append(v.file_name)
            if limit and len(keys) >= limit:
                break
        return keys

    def list_keys_cached(self, prefix, ttl=120):
        """List keys with a short TTL cache (avoid repeated API round-trips)."""
        now = time.time()
        with self._lock:
            cc = getattr(self, "_list_cache", {})
            hit = cc.get(prefix)
            if hit and (now - hit[0]) < ttl:
                return hit[1]
        keys = self.list_keys(prefix)
        with self._lock:
            cc = getattr(self, "_list_cache", {})
            cc[prefix] = (now, keys)
            self._list_cache = cc
        return keys

    # ------------------------------------------------------------------ #
    # frame directory materialisation: pull f_XXXX.jpg frames from B2     #
    # into a local folder so automation/training can read them like the   #
    # original FRAME_DIR.                                                 #
    #                                                                     #
    # Prefers a single frames/frames.zip payload (1 download transaction   #
    # -> 1 frame = 1 Class-B transaction otherwise, which blows the B2    #
    # free-tier download cap). Falls back to per-frame downloads only if  #
    # the zip is not present.                                             #
    # ------------------------------------------------------------------ #
    def materialize_frames(self, num_frames=3951, max_workers=16, force=False):
        """Download frames 1..num_frames from B2 into <cache_dir>/frames/
        (named f_0001.jpg ...). Returns the folder path."""
        frames_dir = self.cache_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        if not self.enabled:
            return str(frames_dir)

        zip_local = self.cache_dir / "frames.zip"
        have_local = zip_local.exists() and zip_local.stat().st_size > 0
        zip_key = "frames/frames.zip"
        if have_local:
            _extract_zip(zip_local, frames_dir, print)
            return str(frames_dir)
        # Prefer the single zip payload. Only fall back to per-frame files
        # when the zip genuinely does not exist (NotFoundError); any other
        # failure (e.g. daily download cap exhausted) is re-raised so the
        # caller sees the real reason instead of silently failing per-frame.
        try:
            got = self.download(zip_key, refresh=True, raise_on_error=True)
        except RuntimeError as e:
            if "not found" in str(e).lower() or "does not exist" in str(e).lower() \
               or "bad_file_id" in str(e).lower() or "404" in str(e):
                pass  # zip not uploaded -> per-frame fallback below
            else:
                raise
        else:
            if got:
                _extract_zip(zip_local, frames_dir, print)
                return str(frames_dir)

        # per-frame fallback (expensive: N Class-B transactions)
        from concurrent.futures import ThreadPoolExecutor
        bucket = self._connect()
        todo = []
        for i in range(1, num_frames + 1):
            dest = frames_dir / f"f_{i:04d}.jpg"
            if dest.exists() and dest.stat().st_size > 0 and not force:
                continue
            todo.append((f"frames/f_{i:04d}.jpg", str(dest)))
        if not todo:
            return str(frames_dir)
        lock = threading.Lock()
        ok, fail = 0, 0

        def _one(kp):
            nonlocal ok, fail
            key, dest = kp
            tmp = dest + ".part"
            try:
                bucket.download_file_by_name(key).save_to(tmp)
                import os as _os
                _os.replace(tmp, dest)
                with lock:
                    ok += 1
                    if ok % 500 == 0:
                        print(f"[b2.frames] {ok}/{len(todo)} materialised")
            except Exception:
                with lock:
                    fail += 1

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            list(ex.map(_one, todo))
        print(f"[b2.frames] done: downloaded={ok} failed={fail} total_frames={num_frames}")
        return str(frames_dir)