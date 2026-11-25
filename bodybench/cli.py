import argparse
import sys
from pathlib import Path

from .config import load_config
from .exceptions import BodyBenchError


def cmd_download(args, cfg, data_root):
    from .data import download_checkpoints, download_dataset

    download_dataset(cfg["dataset"]["repo_id"], data_root)
    download_checkpoints(cfg["checkpoints_dir"])


def cmd_evaluate(args, cfg, data_root):
    from .evaluation.runner import evaluate

    evaluate(cfg, protected_dir=args.protected, output_dir=args.output, data_root=data_root)


def cmd_quicktest(args, cfg, data_root):
    from .evaluation.runner import evaluate

    evaluate(cfg, protected_dir=args.protected, output_dir=args.output, data_root=data_root,
             sample=args.n, sample_seed=args.seed)


def build_parser():
    parser = argparse.ArgumentParser(prog="bodybench")
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download", help="download the dataset and metric checkpoints")
    download.set_defaults(func=cmd_download)

    evaluate = sub.add_parser("evaluate", help="attack protected images and score the results")
    evaluate.add_argument("--protected", type=Path, required=True,
                          help="directory of <method>/<variant>/<stem>.png "
                               "or <method>/<stem>/<variant>/<image>.png")
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.set_defaults(func=cmd_evaluate)

    quicktest = sub.add_parser("quicktest",
                               help="evaluate on a few random images per method as a smoke test")
    quicktest.add_argument("--protected", type=Path, required=True,
                           help="same layout as `evaluate --protected`")
    quicktest.add_argument("--output", type=Path, default=Path("results_quick"))
    quicktest.add_argument("--n", type=int, default=1, help="images per method (default 1)")
    quicktest.add_argument("--seed", type=int, default=0, help="sampling seed (default 0)")
    quicktest.set_defaults(func=cmd_quicktest)

    for p in (download, evaluate, quicktest):
        p.add_argument("--config", type=Path, default=None)
        p.add_argument("--data-root", type=Path, default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    data_root = args.data_root or Path(cfg["dataset"]["root"])
    try:
        args.func(args, cfg, data_root)
    except BodyBenchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
