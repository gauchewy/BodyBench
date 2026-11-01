import numpy as np
import pandas as pd
from scipy import stats

CELL_KEYS = ["protection_method", "protection_variant", "inpainting_mask", "prompt_key"]
METRICS = [
    "defense_score", "baseline_defense_score", "S", "nudity_flag", "skin_exposure",
    "clip_t", "pickscore", "image_reward", "faceid_arcface", "ssim", "psnr", "backfire",
]


def seed_level_ci(group, column):
    per_seed = group[column].astype(float).groupby(group["seed"]).mean().dropna()
    k = len(per_seed)
    if k == 0:
        return np.nan, np.nan
    if k == 1:
        return float(per_seed.mean()), 0.0
    half_width = stats.t.ppf(0.975, k - 1) * per_seed.std(ddof=1) / np.sqrt(k)
    return float(per_seed.mean()), float(half_width)


def summarize(df):
    df = df.copy()
    if "normalized_S" in df:
        df["baseline_defense_score"] = 1 - df["normalized_S"]
    metrics = [m for m in METRICS if m in df]
    records = []
    for cell, group in df.groupby(CELL_KEYS, dropna=False):
        record = dict(zip(CELL_KEYS, cell), n_images=len(group), n_stems=group["stem"].nunique())
        for m in metrics:
            record[f"{m}_mean"], record[f"{m}_ci95"] = seed_level_ci(group, m)
        records.append(record)
    return pd.DataFrame(records)
