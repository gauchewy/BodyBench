from ._common import DEVICE

_model = None


def _patch_transformers():
    # ImageReward 1.5 imports these from transformers.modeling_utils; newer transformers
    # only exposes them from transformers.pytorch_utils.
    from transformers import modeling_utils, pytorch_utils

    for name in ("apply_chunking_to_forward", "find_pruneable_heads_and_indices",
                 "prune_linear_layer"):
        if not hasattr(modeling_utils, name):
            setattr(modeling_utils, name, getattr(pytorch_utils, name))


def score(image_path, prompt):
    global _model
    if _model is None:
        _patch_transformers()
        import ImageReward

        _model = ImageReward.load("ImageReward-v1.0", device=DEVICE)
    return float(_model.score(prompt, image_path))
