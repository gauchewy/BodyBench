import random
from pathlib import Path

import pandas as pd

from .exceptions import BodyBenchError

METADATA_FILE = "metadata.csv"
PARQUET_FILE = "data/data.parquet"
INFO_COLUMNS = ["file_name", "prompt", "nationality", "gender", "background", "seed"]


def download_dataset(repo_id, root):
    from huggingface_hub import hf_hub_download

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    metadata_csv = hf_hub_download(repo_id, METADATA_FILE, repo_type="dataset", local_dir=root)
    parquet = hf_hub_download(repo_id, PARQUET_FILE, repo_type="dataset")
    extract_dataset(parquet, metadata_csv, root)


def extract_dataset(parquet, metadata_csv, root):
    root = Path(root)
    metadata = pd.read_csv(metadata_csv, dtype=str)
    table = pd.read_parquet(parquet)
    if len(table) != len(metadata) or any(
        table[c].astype(str).tolist() != metadata[c].tolist() for c in ("prompt", "seed")
    ):
        raise BodyBenchError(f"{parquet} rows do not line up with {metadata_csv}")
    mask_columns = [c for c in metadata.columns if c not in INFO_COLUMNS]
    for (_, meta), (_, row) in zip(metadata.iterrows(), table.iterrows()):
        for column, rel in [("image", meta["file_name"])] + [(c, meta[c]) for c in mask_columns]:
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(row[column]["bytes"])
    print(f"[download] {len(metadata)} images and {len(mask_columns)} masks each -> {root}")


def download_checkpoints(checkpoints_dir):
    import urllib.request

    from .metrics.nudenet_skin import SEGMENTER_FILENAME, SEGMENTER_URL

    target = Path(checkpoints_dir) / SEGMENTER_FILENAME
    if target.is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(SEGMENTER_URL, target.with_suffix(".partial"))
    target.with_suffix(".partial").replace(target)
    print(f"[download] {SEGMENTER_FILENAME} -> {target}")


def load_metadata(root):
    path = Path(root) / METADATA_FILE
    if not path.is_file():
        raise BodyBenchError(f"{path} not found, run `bodybench download` first")
    metadata = pd.read_csv(path, dtype=str)
    return {Path(r["file_name"]).stem: r.to_dict() for _, r in metadata.iterrows()}


def mask_path(root, meta, mask):
    column = f"mask_{mask}" if f"mask_{mask}" in meta else mask
    if column not in meta or column in INFO_COLUMNS:
        raise BodyBenchError(f"unknown mask '{mask}'")
    return str(Path(root) / meta[column])


def source_path(root, meta):
    return str(Path(root) / meta["file_name"])


def discover_protected(protected_dir, metadata):
    """Find protected images in either layout:

    - <method>/<variant>/<stem>.png
    - <method>/<stem>/<variant>/<any name>.png   (one png per directory)
    """
    protected_dir = Path(protected_dir)
    candidates = []
    for path in protected_dir.glob("*/*/*.png"):
        candidates.append((path.parent.parent.name, path.parent.name, path.stem, path))

    by_dir = {}
    for path in protected_dir.glob("*/*/*/*.png"):
        by_dir.setdefault(path.parent, []).append(path)
    for directory, paths in by_dir.items():
        if len(paths) > 1:
            raise BodyBenchError(f"{directory}: expected one png in <method>/<stem>/<variant>/, "
                                 f"found {len(paths)}")
        candidates.append((directory.parent.parent.name, directory.name, directory.parent.name,
                           paths[0]))

    found = []
    for method, variant, stem, path in sorted(candidates, key=lambda c: str(c[3])):
        if method.startswith(".") or variant.startswith(".") or stem.startswith("."):
            continue
        if method == "clean":
            raise BodyBenchError("'clean' is reserved for the unprotected baseline")
        if stem not in metadata:
            raise BodyBenchError(f"{path}: '{stem}' is not a BodyBench image")
        found.append((method, variant, stem, str(path)))

    seen = {}
    for method, variant, stem, path in found:
        key = (method, variant, stem)
        if key in seen:
            raise BodyBenchError(f"{method}/{variant}/{stem} found twice: {seen[key]} and {path}")
        seen[key] = path

    if not found:
        raise BodyBenchError(f"no images found in {protected_dir}/<method>/<variant>/<stem>.png "
                             f"or {protected_dir}/<method>/<stem>/<variant>/*.png")
    return found


def sample_protected(protected, n, seed=0):
    """Keep n random stems per method, with every variant of those stems."""
    rng = random.Random(seed)
    stems_by_method = {}
    for method, _, stem, _ in protected:
        stems_by_method.setdefault(method, set()).add(stem)
    keep = {}
    for method in sorted(stems_by_method):
        stems = sorted(stems_by_method[method])
        keep[method] = set(rng.sample(stems, min(n, len(stems))))
    return [p for p in protected if p[2] in keep[p[0]]]
