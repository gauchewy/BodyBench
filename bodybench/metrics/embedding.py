import numpy as np
import torch
from PIL import Image

from ._common import DEVICE, GPU_BATCH

CLIP = ("openai/clip-vit-large-patch14", "openai/clip-vit-large-patch14")
PICKSCORE = ("yuvalkirstain/PickScore_v1", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K")


class Embedder:
    def __init__(self, model_name, processor_name):
        from transformers import AutoModel, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(processor_name)
        self.model = AutoModel.from_pretrained(model_name).eval().to(DEVICE)
        self.text_cache = {}

    @torch.inference_mode()
    def images(self, paths):
        out = {}
        for i in range(0, len(paths), GPU_BATCH):
            chunk = paths[i:i + GPU_BATCH]
            inputs = self.processor(
                images=[Image.open(p).convert("RGB") for p in chunk], return_tensors="pt"
            ).to(DEVICE)
            features = self.model.get_image_features(pixel_values=inputs["pixel_values"])
            out.update(zip(chunk, _normalize(features)))
        return out

    @torch.inference_mode()
    def text(self, text):
        if text not in self.text_cache:
            inputs = self.processor(
                text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77
            ).to(DEVICE)
            features = self.model.get_text_features(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
            self.text_cache[text] = _normalize(features)[0]
        return self.text_cache[text]


def _normalize(features):
    return (features / features.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32)


def score(image_embedding, text_embedding):
    return float(np.dot(image_embedding, text_embedding))
