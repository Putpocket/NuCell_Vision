

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from classical_model import PARAMETER_COUNT, predict_classical, segment_classical

ROOT = Path(__file__).resolve().parent


def write_png(path, array):
    ok, encoded = cv2.imencode(".png", array)
    if not ok:
        raise ValueError(f"PNG encoding failed: {path}")
    encoded.tofile(str(path))


def load_rgb(data_dir, image_id):
    paths = sorted((data_dir / image_id / "images").glob("*.png"))
    if len(paths) != 1:
        raise ValueError(f"Expected one image: {data_dir / image_id / 'images'}")
    # Same decoder/channel order as the team's U-Net and Mask R-CNN.
    with Image.open(paths[0]) as source:
        return np.asarray(source.convert("RGB"))


def read_ids(folder):
    splits, hashes = {}, {}
    for name, expected in (("train", 469), ("val", 100), ("test", 101)):
        raw = (folder / f"{name}.txt").read_bytes()
        ids = [s.strip() for s in raw.decode("utf-8-sig").splitlines()]
        if len(ids) != expected or len(set(ids)) != expected or any(not s or "/" in s or "\\" in s or s in (".", "..") for s in ids):
            raise ValueError(f"Invalid split IDs: {name}")
        splits[name] = ids
        hashes[name] = hashlib.sha256(raw).hexdigest()
    if any(set(splits[a]) & set(splits[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Dataset splits overlap.")
    return splits, hashes


def save_stages(path, image, labels, stages):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    top, left, nh, nw, _, _ = stages["geometry"]
    colors = np.random.default_rng(42).integers(50, 256, (int(labels.max()) + 1, 3), dtype=np.uint8)
    colors[0] = 0
    arrays = [image, stages["otsu"][top:top + nh, left:left + nw],
              stages["distance"][top:top + nh, left:left + nw],
              stages["seeds"][top:top + nh, left:left + nw], colors[labels]]
    titles = ["Original", f"Otsu (invert={stages['inverted']})", "Distance transform",
              f"Markers: {stages['seed_count']}", f"Watershed: {int(labels.max())}"]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))
    try:
        for index, (ax, array, title) in enumerate(zip(axes, arrays, titles)):
            ax.imshow(array, cmap="magma" if index == 2 else "gray", interpolation="nearest")
            ax.set_title(title)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=100)
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/stage1_train")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "splits")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--polarity", choices=["bright", "auto"], default="auto")
    parser.add_argument("--seed-ratio", type=float, default=0.7)
    parser.add_argument("--min-area", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--save-stages", action="store_true")
    parser.add_argument("--final-test", action="store_true")
    args = parser.parse_args()
    if args.split == "test" and not args.final_test:
        parser.error("Test predictions require frozen validation settings and --final-test.")
    if not 0 < args.seed_ratio < 1 or args.min_area < 1 or args.warmup < 1:
        parser.error("Invalid seed-ratio/min-area/warmup.")
    for name in ("data_dir", "split_dir", "output"):
        path = getattr(args, name)
        if path is not None and not path.is_absolute():
            setattr(args, name, ROOT / path)
    output = args.output or ROOT / "results" / f"classical_{args.split}_{args.polarity}_predictions"
    if output.exists():
        parser.error(f"Output already exists. Use a new --output folder: {output}")
    splits, hashes = read_ids(args.split_dir)
    ids = splits[args.split]
    kwargs = dict(polarity=args.polarity, size=512, seed_ratio=args.seed_ratio, min_area=args.min_area)
    cv2.setNumThreads(1)
    first = load_rgb(args.data_dir, ids[0])
    for _ in range(args.warmup):
        predict_classical(first, **kwargs)
    output.mkdir(parents=True)
    rows = []
    for number, image_id in enumerate(ids, 1):
        image = load_rgb(args.data_dir, image_id)
        started = time.perf_counter()
        labels, stages = segment_classical(image, **kwargs)
        binary = labels > 0
        count = int(labels.max())
        elapsed_ms = (time.perf_counter() - started) * 1000
        if count > 65535:
            raise ValueError("Instance count exceeds uint16 PNG range.")
        folder = output / "predictions" / image_id
        folder.mkdir(parents=True)
        write_png(folder / "prediction.png", binary.astype(np.uint8) * 255)
        write_png(folder / "instances.png", labels.astype(np.uint16))
        if args.save_stages:
            save_stages(folder / "stages.png", image, labels, stages)
        rows.append(dict(image_id=image_id, predicted_count=count, time_ms=elapsed_ms,
            prediction_path=f"predictions/{image_id}/prediction.png",
            instances_path=f"predictions/{image_id}/instances.png",
            otsu_threshold=stages["threshold"], inverted=stages["inverted"]))
        print(f"[{number}/{len(ids)}] {image_id[:12]} | nuclei={count} | {elapsed_ms:.2f} ms", flush=True)
    with (output / "inference.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = dict(model="classical", split=args.split, images=len(rows), params=PARAMETER_COUNT,
        device="cpu", input_size=512, input_order="RGB uint8",
        config=kwargs, warmup=args.warmup, opencv_threads=1,
        python=sys.version, opencv=cv2.__version__, numpy=np.__version__, platform=platform.platform(),
        split_sha256=hashes,
        timing_scope="RGB uint8 in RAM -> letterbox -> Otsu -> distance -> Watershed -> original-size labels -> binary/count; excludes disk I/O, ground truth, metrics and visualization",
        evaluation_status="predictions_only; evaluator owns per_image.csv, summary.json and ground_truth.png")
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} predictions: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
