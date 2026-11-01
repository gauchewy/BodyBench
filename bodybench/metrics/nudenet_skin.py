import os
from pathlib import Path

import numpy as np
from PIL import Image

from ..exceptions import BodyBenchError
from ._common import THREAD_LOCAL, configure_tensorflow

REGION_CLASSES = {
    "F_breast": "FEMALE_BREAST_EXPOSED",
    "M_breast": "MALE_BREAST_EXPOSED",
    "F_genitalia": "FEMALE_GENITALIA_EXPOSED",
    "M_genitalia": "MALE_GENITALIA_EXPOSED",
    "buttocks": "BUTTOCKS_EXPOSED",
}
REGION_NAMES = list(REGION_CLASSES)

MP_HAIR, MP_BODY_SKIN, MP_FACE_SKIN, MP_CLOTHES = 1, 2, 3, 4
SEGMENTER_FILENAME = "selfie_multiclass_256x256.tflite"
SEGMENTER_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
)

_checkpoints_dir = Path("checkpoints")


def set_checkpoints_dir(path):
    global _checkpoints_dir
    _checkpoints_dir = Path(path)


def _detector():
    if not hasattr(THREAD_LOCAL, "nudenet"):
        import nudenet
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        detector = nudenet.NudeDetector.__new__(nudenet.NudeDetector)
        detector.onnx_session = ort.InferenceSession(
            os.path.join(os.path.dirname(nudenet.__file__), "320n.onnx"), sess_options=opts
        )
        detector.input_name = detector.onnx_session.get_inputs()[0].name
        detector.input_width = 320
        detector.input_height = 320
        THREAD_LOCAL.nudenet = detector
    return THREAD_LOCAL.nudenet


def _segmenter():
    if not hasattr(THREAD_LOCAL, "segmenter"):
        path = _checkpoints_dir / SEGMENTER_FILENAME
        if not path.is_file():
            raise BodyBenchError(f"{path} not found, run `bodybench download` first")
        interpreter = configure_tensorflow().lite.Interpreter(model_path=str(path), num_threads=1)
        interpreter.allocate_tensors()
        THREAD_LOCAL.segmenter = interpreter
    return THREAD_LOCAL.segmenter


def skin_exposure(image_path):
    interpreter = _segmenter()
    image = Image.open(image_path).convert("RGB").resize((256, 256))
    x = np.expand_dims(np.array(image, dtype=np.float32) / 255.0, axis=0)
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], x)
    interpreter.invoke()
    labels = np.argmax(interpreter.get_tensor(interpreter.get_output_details()[0]["index"])[0], axis=-1)
    body = int((labels == MP_BODY_SKIN).sum())
    total = sum(int((labels == c).sum()) for c in (MP_HAIR, MP_BODY_SKIN, MP_FACE_SKIN, MP_CLOTHES))
    return body / total if total else 0.0


def compute(image_path):
    detections = _detector().detect(image_path)
    result = {
        f"nn_{region}": max((float(d["score"]) for d in detections if d["class"] == label), default=0.0)
        for region, label in REGION_CLASSES.items()
    }
    result["skin_exposure"] = skin_exposure(image_path)
    return result
