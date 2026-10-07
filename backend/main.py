import os, shutil, threading, time, uuid, zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from processing import run_job

DATA = Path(os.getenv("DATA_DIR", "/tmp/yolo-merge"))
RETENTION = float(os.getenv("RETENTION_HOURS", "6")) * 3600
ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
JOBS: dict = {}
POOL = ThreadPoolExecutor(max_workers=1)  # one inference job at a time

app = FastAPI(title="YOLO Merge")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["*"], allow_headers=["*"])


def save(up: UploadFile, dest: Path):
    with open(dest, "wb") as f:
        shutil.copyfileobj(up.file, f)


def dest_for(d: Path, name: str, stems: set) -> Path:
    """Unique path; stems are kept unique so label files never collide (a.jpg vs a.png)."""
    p, stem, i = Path(name), Path(name).stem, 1
    while stem in stems:
        stem = f"{p.stem}_{i}"; i += 1
    stems.add(stem)
    return d / f"{stem}{p.suffix}"


def worker(jid: str, conf: float, iou: float, mode: str):
    job = JOBS[jid]
    job["status"] = "running"

    def progress(n, name):
        job["processed"] = n
        if name and len(job["previews"]) < 48:
            job["previews"].append(name)

    try:
        job["classes"], job["skipped"] = run_job(job["dir"], conf, iou, mode, progress)
        job["status"] = "done"
    except Exception as e:
        job["status"], job["error"] = "failed", str(e)


@app.post("/jobs")
def create_job(model_a: UploadFile = File(...), model_b: UploadFile = File(...),
               images: list[UploadFile] = File(...), conf: float = Form(0.25),
               iou: float = Form(0.5), merge_mode: str = Form("merge")):
    for m in (model_a, model_b):
        if not (m.filename or "").lower().endswith(".pt"):
            raise HTTPException(400, f"'{m.filename}' is not a .pt file")
    if merge_mode not in ("merge", "separate"):
        raise HTTPException(400, "merge_mode must be 'merge' or 'separate'")
    if not (0 < conf < 1 and 0 < iou < 1):
        raise HTTPException(400, "conf and iou must be between 0 and 1")

    jid = uuid.uuid4().hex[:12]
    root = DATA / jid
    img_dir = root / "input" / "images"
    img_dir.mkdir(parents=True)
    save(model_a, root / "input" / "a.pt")
    save(model_b, root / "input" / "b.pt")

    n, stems = 0, set()
    for up in images:
        name = Path(up.filename or "").name
        ext = Path(name).suffix.lower()
        if ext == ".zip":
            tmp = root / "input" / "upload.zip"
            save(up, tmp)
            try:
                with zipfile.ZipFile(tmp) as z:
                    for info in z.infolist():
                        base = Path(info.filename).name
                        if info.is_dir() or base.startswith(".") or Path(base).suffix.lower() not in IMG_EXT:
                            continue
                        with z.open(info) as s, open(dest_for(img_dir, base, stems), "wb") as d:
                            shutil.copyfileobj(s, d)
                        n += 1
            except zipfile.BadZipFile:
                shutil.rmtree(root)
                raise HTTPException(400, "The uploaded zip file is corrupt")
            tmp.unlink()
        elif ext in IMG_EXT:
            save(up, dest_for(img_dir, name, stems))
            n += 1
    if n == 0:
        shutil.rmtree(root)
        raise HTTPException(400, "No images found. Upload jpg, png, jpeg or webp files, or a zip of them.")

    JOBS[jid] = {"status": "queued", "processed": 0, "total": n, "error": None, "previews": [],
                 "classes": [], "skipped": [], "created": time.time(), "dir": root}
    POOL.submit(worker, jid, conf, iou, merge_mode)
    return {"job_id": jid, "total": n}


def get_job(jid: str) -> dict:
    if jid not in JOBS:
        raise HTTPException(404, "Job not found or expired")
    return JOBS[jid]


@app.get("/jobs/{jid}")
def job_status(jid: str):
    return {k: v for k, v in get_job(jid).items() if k not in ("dir", "created")}


@app.get("/jobs/{jid}/preview/{name}")
def preview(jid: str, name: str):
    p = get_job(jid)["dir"] / "output" / "annotated" / Path(name).name
    if not p.exists():
        raise HTTPException(404, "Preview not found")
    return FileResponse(p)


@app.get("/jobs/{jid}/download")
def download(jid: str):
    job = get_job(jid)
    if job["status"] != "done":
        raise HTTPException(409, "Job is not finished yet")
    return FileResponse(job["dir"] / "result.zip", filename="yolo-merge-output.zip", media_type="application/zip")


def cleanup_loop():
    while True:
        time.sleep(300)
        for jid, job in list(JOBS.items()):
            if job["status"] in ("done", "failed") and time.time() - job["created"] > RETENTION:
                shutil.rmtree(job["dir"], ignore_errors=True)
                JOBS.pop(jid, None)


@app.on_event("startup")
def startup():
    DATA.mkdir(parents=True, exist_ok=True)
    for old in DATA.iterdir():  # leftovers from a previous process
        shutil.rmtree(old, ignore_errors=True)
    threading.Thread(target=cleanup_loop, daemon=True).start()


@app.get("/health")
def health():
    return {"ok": True}
