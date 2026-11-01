import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

STEMS = ["00000", "00001"]
MASK_COLUMNS = ["body_dilated_7", "mask_body", "mask_body_ctx"]


def write_image(path, seed=0, mode="RGB"):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    shape = (32, 32, 3) if mode == "RGB" else (32, 32)
    Image.fromarray(rng.integers(0, 256, shape, dtype=np.uint8), mode=mode).save(path)
    return path


def metadata_csv(seeds=(111, 222)):
    header = "file_name,prompt,nationality,gender,background,seed," + ",".join(MASK_COLUMNS)
    rows = [header]
    for stem, gender, seed in zip(STEMS, ("woman", "man"), seeds):
        masks = ",".join(f"masks/{stem}_{c}.png" for c in MASK_COLUMNS)
        rows.append(f'{stem}.png,"full body photo of a {gender}",German,{gender},park,{seed},{masks}')
    return "\n".join(rows) + "\n"


@pytest.fixture
def data_root(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    (root / "metadata.csv").write_text(metadata_csv())
    for i, stem in enumerate(STEMS):
        write_image(root / f"{stem}.png", seed=i)
        for column in MASK_COLUMNS:
            write_image(root / "masks" / f"{stem}_{column}.png", mode="L")
    return root


@pytest.fixture
def protected_dir(tmp_path):
    root = tmp_path / "protected"
    for method, variant in (("photoguard", "body"), ("photoguard", "body_ctx"), ("advpaint", "seg")):
        for stem in STEMS:
            write_image(root / method / variant / f"{stem}.png", seed=7)
    return root
