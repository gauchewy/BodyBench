from pathlib import Path

from ..attack import build_jobs, run_attacks
from ..data import discover_protected, load_metadata, sample_protected
from ..metrics.aggregates import add_aggregates
from .aggregate import summarize
from .pipeline import collect_metrics


def evaluate(cfg, *, protected_dir, output_dir, data_root, sample=None, sample_seed=0):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(data_root)
    protected = discover_protected(protected_dir, metadata)
    if sample is not None:
        protected = sample_protected(protected, sample, sample_seed)
        print(f"[evaluate] quick test: {sample} random images per method (seed {sample_seed})")
    print(f"[evaluate] {len(protected)} protected images, "
          f"{len({(m, v) for m, v, _, _ in protected})} method/variant pairs")

    jobs = build_jobs(protected, metadata, data_root, cfg, output_dir / "attacks")
    run_attacks(jobs, cfg["attack"])

    enabled = set(cfg["metrics"])
    scores = collect_metrics(
        jobs, output_dir / "scores.csv", enabled=enabled, checkpoints_dir=cfg["checkpoints_dir"]
    )
    per_image = add_aggregates(scores) if "nudenet" in enabled else scores
    per_image.to_csv(output_dir / "per_image.csv", index=False)
    summarize(per_image).to_csv(output_dir / "summary.csv", index=False)
    print(f"[evaluate] wrote {output_dir / 'per_image.csv'} and {output_dir / 'summary.csv'}")
