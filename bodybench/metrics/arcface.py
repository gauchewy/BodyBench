import numpy as np

from ._common import GPU_BATCH, configure_tensorflow, l2_normalize


def _represent(images):
    configure_tensorflow()
    from deepface import DeepFace

    return DeepFace.represent(
        img_path=images,
        model_name="ArcFace",
        detector_backend="retinaface",
        enforce_detection=False,
        align=True,
    )


def _embedding(faces):
    return l2_normalize(faces[0]["embedding"]) if faces and "embedding" in faces[0] else None


def _embed_one(path):
    try:
        return _embedding(_represent(path))
    except Exception:
        return None


def batch_embed(paths):
    out = {}
    for i in range(0, len(paths), GPU_BATCH):
        chunk = list(paths[i:i + GPU_BATCH])
        try:
            results = _represent(chunk)
        except Exception:
            out.update({p: _embed_one(p) for p in chunk})
            continue
        out.update({p: _embedding(faces) for p, faces in zip(chunk, results)})
    return out


def score(image_embedding, source_embedding):
    if image_embedding is None or source_embedding is None:
        return float("nan")
    return float(np.dot(image_embedding, source_embedding))
