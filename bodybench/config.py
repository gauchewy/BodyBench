from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "paper.yaml"


def load_config(path=None):
    with open(path or DEFAULT_CONFIG) as fh:
        return yaml.safe_load(fh)
