import colorsys, json, shutil
from pathlib import Path

import cv2, torch, yaml
from torchvision.ops import batched_nms
from ultralytics import YOLO


def load(path: Path, label: str) -> YOLO:
    try:
        m = YOLO(str(path))
        if m.task != "detect" or not isinstance(m.names, dict):
            raise ValueError(f"task is '{m.task}', expected 'detect'")
        return m
    except Exception as e:
        raise ValueError(f"{label} is not a valid Ultralytics detection model: {e}")


def unify(names_a: dict, names_b: dict, mode: str):
    """Returns (classes, map_a, map_b). mode='merge' shares IDs for equal names; 'separate' offsets B."""
    classes, idx, ma, mb = [], {}, {}, {}

    def add(name, merge):
        if merge and name in idx:
            return idx[name]
        classes.append(name)
        idx[name] = len(classes) - 1
        return idx[name]

    for i, n in sorted(names_a.items()):
        ma[i] = add(n, mode == "merge")
    for i, n in sorted(names_b.items()):
        mb[i] = add(n, mode == "merge")
    return classes, ma, mb


def color(cid: int):
    r, g, b = colorsys.hsv_to_rgb((cid * 0.618) % 1, 0.8, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)  # BGR


def run_job(root: Path, conf: float, iou: float, mode: str, progress):
    inp, out = root / "input", root / "output"
    (out / "annotated").mkdir(parents=True)
    (out / "labels").mkdir()
    A, B = load(inp / "a.pt", "Model A"), load(inp / "b.pt", "Model B")
    classes, ma, mb = unify(A.names, B.names, mode)
    files = sorted((inp / "images").iterdir())
    skipped = []

    for k, f in enumerate(files, 1):
        img = cv2.imread(str(f))
        if img is None:
            skipped.append(f.name)
            progress(k, None)
            continue
        h, w = img.shape[:2]
        boxes, scores, cls = [], [], []
        for M, mp in ((A, ma), (B, mb)):
            r = M.predict(img, conf=conf, iou=iou, verbose=False)[0]
            if len(r.boxes):
                boxes.append(r.boxes.xyxy.cpu())
                scores.append(r.boxes.conf.cpu())
                cls.append(torch.tensor([mp[int(c)] for c in r.boxes.cls.tolist()]))
        lines = []
        if boxes:
            b, s, c = torch.cat(boxes), torch.cat(scores), torch.cat(cls)
            fs, th = max(0.5, w / 1600), max(2, w // 400)
            for i in batched_nms(b, s, c, iou).tolist():  # cross-model NMS, per class
                x1, y1, x2, y2 = b[i].tolist()
                ci, sc = int(c[i]), float(s[i])
                lines.append(f"{ci} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
                col, label = color(ci), f"{classes[ci]} {sc:.2f}"
                p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
                cv2.rectangle(img, p1, p2, col, th)
                (tw, tt), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
                ty = max(p1[1], tt + 6)
                cv2.rectangle(img, (p1[0], ty - tt - 6), (p1[0] + tw + 4, ty), col, -1)
                cv2.putText(img, label, (p1[0] + 2, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out / "annotated" / f.name), img)
        (out / "labels" / f"{f.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        progress(k, f.name)

    (out / "classes.txt").write_text("\n".join(classes) + "\n")
    (out / "data.yaml").write_text(yaml.safe_dump(
        {"path": ".", "train": "images", "val": "images", "nc": len(classes), "names": classes}, sort_keys=False))
    (out / "class_mapping.json").write_text(json.dumps({"model_a": ma, "model_b": mb}, indent=2))
    shutil.make_archive(str(root / "result"), "zip", root, "output")
    return classes, skipped
