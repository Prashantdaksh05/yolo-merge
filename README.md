# YOLO Merge

Run two YOLO detection models over one image set. Output: `annotated/`, a single `labels/` folder, `classes.txt`, `data.yaml` (plus `class_mapping.json` showing how each model's class IDs map to the unified list).

```
frontend/  Next.js + Tailwind  -> Vercel
backend/   FastAPI + Ultralytics (Docker) -> Railway / Render / Modal / any Docker host
```

Vercel can't run PyTorch (size and timeout limits), so inference lives in the backend. The browser uploads straight to the backend, which also avoids Vercel's 4.5 MB request limit.

## API
- `POST /jobs` multipart: `model_a`, `model_b`, `images` (files or a zip), `conf`, `iou`, `merge_mode` (`merge` | `separate`) -> `{job_id}`
- `GET /jobs/{id}` -> status (`queued|running|done|failed`), progress, previews, classes
- `GET /jobs/{id}/preview/{name}`, `GET /jobs/{id}/download`

## Run locally
```bash
# backend
cd backend && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# frontend (new terminal)
cd frontend && cp .env.example .env.local && npm install && npm run dev
```
Open http://localhost:3000.

## Deploy the backend
1. Push the repo, create a service from `backend/` using its Dockerfile (Railway or Render).
2. Set env vars:
   - `ALLOWED_ORIGINS` = your Vercel URL, e.g. `https://yolo-merge.vercel.app` (comma-separated for several)
   - `RETENTION_HOURS` = hours before results are deleted (default 6)
   - `DATA_DIR` = scratch directory (default `/tmp/yolo-merge`)
3. Copy the public URL. Check `/health`.

For a GPU, use a GPU host (Modal, RunPod, a GPU VM) with a CUDA base image. Ultralytics picks the GPU automatically.

## Deploy the frontend to Vercel
1. Import the repo, set **Root Directory** to `frontend`.
2. Add env var `NEXT_PUBLIC_BACKEND_URL` = the backend URL (no trailing slash).
3. Deploy.

## Notes
- Jobs are held in memory and run one at a time, so run a single backend instance. Disk is ephemeral; results vanish on restart or after the retention period.
- Merging: with `merge`, classes with the same name share one ID. Duplicate boxes from the two models (same class, IoU above the threshold) are removed, keeping the higher confidence.
- `.pt` files are Python pickles and can run code when loaded. Only upload weights you trust, and don't expose this backend publicly without authentication.
