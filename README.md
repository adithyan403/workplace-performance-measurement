# WPM · KSRTC — Workplace Performance Measurement Web App

Colourful Flask dashboard + ML automation implementing the WPM method of
Nesterak, Szelagowski & Radziszewski, "Workplace performance measurement:
digitalization of work observation and analysis", *J. Intell. Manuf.* 36, 3569–3585 (2025).

Live analytics from a KSRTC station-master waybill-entry duty shift.
Data lives in **MongoDB Atlas** (`WPM.duty_logs`); the ML model classifies
frames into activities, auto-creates duty logs and flags value-added vs
non-value-added time.

## Pages

| Route        | What it does                                                       |
|--------------|--------------------------------------------------------------------|
| `/`          | KPI dashboard — VA/NVA, activity donut, event counts, module map   |
| `/analysis`  | Motion energy line charts, workspace zone heatmap, histogram       |
| `/logs`      | Searchable/filterable duty-log explorer (live MongoDB)             |
| `/ml`        | Confusion matrix, per-class report, live frame prediction + auto-log |
| `/reports`   | On-demand colourful PDF duty report (download)                     |
| `/api/health`| Health probe (model + MongoDB status)                              |

## WPM modules (the paper's five)

```
modules/
  video_module.py       MODULE 1 · VIDEO        acquisition + digitalization → 1 Hz frames
  process_module.py     MODULE 2 · PROCESS      activity catalogue, zones, thresholds, event registration
  analysis_module.py    MODULE 3 · ANALYSIS     motion metrics → event log → VA/NVA discovery
  automation_module.py  MODULE 4 · AUTOMATION   ML classifier → predict → auto duty logs → VA/NVA
  reporting_module.py   MODULE 5 · REPORTING    dashboards, chart payloads, PDF reports
```

## Local run (Windows dev env)

```powershell
$env:PYTHONPATH="D:\pylib"
$env:MONGO_URI="mongodb+srv://USER:PASS@cluster0.xxx.mongodb.net/?appName=Cluster0"
$env:B2_KEY_ID="your_b2_keyId"
$env:B2_APPLICATION_KEY="your_b2_applicationKey"
$env:FRAME_DIR="D:\ksrtc\frames_1s"     # optional
D:\python.exe app.py                    # http://127.0.0.1:5000
```

If `MONGO_URI` is absent the dashboard/analysis fall back to the bundled
`data/event_log_sample.json` (1697 Phase-I events).

## Backblaze B2 asset store

The app can stream everything from Backblaze B2, so a 23 GB video never has
to ship. Bucket layout mirrors local disk:

```
frames/f_0001.jpg ... f_3951.jpg     # 1 Hz frames
clips/clip_0_2.mp4 ...               # one clip per duty event (960px, <=30 s)
```

Set `B2_KEY_ID` / `B2_APPLICATION_KEY` (and optionally `B2_BUCKET`, default
`ksrtc-wpm-assets`) and the resolution order for every asset becomes:

**Web clips & frames** (`/clip/...`, `/frame/...`):  local cache → **B2** → ffmpeg cut.

**ML options** (retrain, whole-video re-predict, live predict): form override
→ local `FRAME_DIR` → **materialise all 3951 frames from B2 into a local
cache** → bundled `static/sample_frames/`. So all ML buttons work even when no
frames exist on the host.

## Re-train the ML model (from local frames + Phase-I event log)

```powershell
D:\python.exe scripts\train_model.py D:\ksrtc\frames_1s D:\ksrtc\wpm_study\mod_analysis\wpm_event_log.csv
```

Writes `data/model.joblib` (compressed ~5.6 MB) + `data/training_report.json`.
Re-run `scripts\prepare_static_data.py` to refresh the dashboard JSON files.

## Deploy to Render

1. Put this folder in a GitHub repo (commit `data/model.joblib`!).
2. In Render → **New → Web Service**, pick the repo.
3. Set env vars:
   - `MONGO_URI` → your Atlas connection string
   - `SECRET_KEY` → random string
   - `B2_KEY_ID` / `B2_APPLICATION_KEY` → Backblaze B2 (frames + clips, recommended)
   - `FRAME_DIR` → (optional) full 1-second frame folder; NOT `static/sample_frames`
     (the 12 bundled frames are a viewer fallback only — the ML options need a
     full frame set, which they pull from B2 when `B2_*` vars are set)
4. Build command: `pip install -r requirements.txt`
5. Start command: `gunicorn app:app --workers 2 --threads 4 --timeout 300 --bind 0.0.0.0:$PORT`
6. Health check path: `/api/health`

The app bundles `static/sample_frames/` (12 representative frames) and the
pre-trained model, so **ML prediction works on Render out of the box** without
shipping the 23 GB source video. Point `frame_dir` at a real frame folder
when available for full-shift predictions.

## Method alignment to the paper

- Digitalization of a manual workstation process from camera video (Module 1)
- Process parameterisation: activity catalogue, zones, thresholds (Module 2)
- Metric extraction + event log → value-added vs non-value-added analytics (Module 3)
- ML automation of observation (Module 4) — the paper's Phase-II trajectory
- Stakeholder reporting: dashboards + PDF (Module 5)