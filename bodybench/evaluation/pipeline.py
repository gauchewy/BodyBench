import csv
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..exceptions import BodyBenchError

ROW_COLUMNS = [
    "stem", "nationality", "gender", "background",
    "protection_method", "protection_variant", "inpainting_mask", "prompt_key", "prompt_text", "seed",
    "image_path", "source_path",
]
CSV_DTYPES = {"stem": str, "protection_variant": str}


def metric_columns(enabled):
    from ..metrics.nudenet_skin import REGION_NAMES

    columns = []
    if "nudenet" in enabled:
        columns += [f"nn_{r}" for r in REGION_NAMES] + ["skin_exposure"]
    for name in ("clip_t", "pickscore", "image_reward"):
        if name in enabled:
            columns.append(name)
    if "arcface" in enabled:
        columns.append("faceid_arcface")
    if "ssim_psnr" in enabled:
        columns += ["ssim", "psnr"]
    return columns


def collect_metrics(rows, csv_path, *, enabled, checkpoints_dir):
    csv_path = Path(csv_path)
    columns = ROW_COLUMNS + metric_columns(enabled)
    done = set()
    if csv_path.is_file() and csv_path.stat().st_size:
        previous = pd.read_csv(csv_path, dtype=CSV_DTYPES)
        if list(previous.columns) != columns:
            raise BodyBenchError(f"{csv_path} was written with different metrics, delete it to rescore")
        done = set(previous["image_path"])
    pending = [r for r in rows if r["image_path"] not in done]
    print(f"[score] {len(pending)} of {len(rows)} images to score")
    if pending:
        _score(pending, rows, csv_path, columns, enabled, checkpoints_dir)
    scores = pd.read_csv(csv_path, dtype=CSV_DTYPES)
    scores["protection_variant"] = scores["protection_variant"].fillna("")
    return scores[scores["image_path"].isin({r["image_path"] for r in rows})]


def _match_key(row):
    return row["stem"], row["inpainting_mask"], row["prompt_key"], row["seed"]


def _score(pending, rows, csv_path, columns, enabled, checkpoints_dir):
    from ..metrics import _common

    clip = pickscore = nudenet = traditional = image_reward = arcface = embedding = None
    if {"clip_t", "pickscore"} & enabled:
        from ..metrics import embedding
    if "clip_t" in enabled:
        clip = embedding.Embedder(*embedding.CLIP)
    if "pickscore" in enabled:
        pickscore = embedding.Embedder(*embedding.PICKSCORE)
    if "nudenet" in enabled:
        from ..metrics import nudenet_skin as nudenet

        nudenet.set_checkpoints_dir(checkpoints_dir)
    if "ssim_psnr" in enabled:
        from ..metrics import traditional
    if "image_reward" in enabled:
        from ..metrics import image_reward
    if "arcface" in enabled:
        from ..metrics import arcface

    clean = {_match_key(r): r["image_path"] for r in rows if r["protection_method"] == "clean"}

    def cpu_metrics(row):
        out = {}
        if nudenet is not None:
            out.update(nudenet.compute(row["image_path"]))
        if traditional is not None:
            ref = None if row["protection_method"] == "clean" else clean.get(_match_key(row))
            out["ssim"], out["psnr"] = traditional.compute(ref, row["image_path"]) if ref else (np.nan, np.nan)
        return out

    by_stem = {}
    for row in pending:
        by_stem.setdefault(row["stem"], []).append(row)

    new_file = not csv_path.is_file() or csv_path.stat().st_size == 0
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="") as fh, \
            ThreadPoolExecutor(_common.CPU_WORKERS) as pool, \
            tqdm(total=len(pending), desc="metrics") as bar:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        for stem_rows in by_stem.values():
            images = sorted({r["image_path"] for r in stem_rows})
            clip_images = clip.images(images) if clip else None
            pick_images = pickscore.images(images) if pickscore else None
            faces = arcface.batch_embed(images + [stem_rows[0]["source_path"]]) if arcface else None
            for row, extra in zip(stem_rows, pool.map(cpu_metrics, stem_rows)):
                row = {**row, **extra}
                path, prompt = row["image_path"], row["prompt_text"]
                if clip:
                    row["clip_t"] = embedding.score(clip_images[path], clip.text(prompt))
                if pickscore:
                    row["pickscore"] = embedding.score(pick_images[path], pickscore.text(prompt))
                if image_reward:
                    row["image_reward"] = image_reward.score(path, prompt)
                if arcface:
                    row["faceid_arcface"] = arcface.score(faces[path], faces[row["source_path"]])
                writer.writerow(row)
                bar.update(1)
            fh.flush()
            os.fsync(fh.fileno())
