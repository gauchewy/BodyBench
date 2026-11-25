import os
import threading

import numpy as np
import torch


def _available_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 4


CPU_WORKERS = int(os.environ.get("BODYBENCH_CPU_WORKERS", max(1, _available_cpus() - 2)))
GPU_BATCH = int(os.environ.get("BODYBENCH_GPU_BATCH", 64))
DEVICE = os.environ.get("BODYBENCH_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
THREAD_LOCAL = threading.local()

_tf = None
_tf_lock = threading.Lock()


def configure_tensorflow():
    global _tf
    with _tf_lock:
        if _tf is None:
            os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
            os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
            import tensorflow as tf

            try:
                if DEVICE == "cpu":
                    tf.config.set_visible_devices([], "GPU")
                else:
                    for gpu in tf.config.list_physical_devices("GPU"):
                        tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
            _tf = tf
    return _tf


def l2_normalize(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-12)
