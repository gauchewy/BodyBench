import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute(ref_path, image_path):
    ref = np.asarray(Image.open(ref_path).convert("RGB"))
    image = Image.open(image_path).convert("RGB")
    if image.size != (ref.shape[1], ref.shape[0]):
        image = image.resize((ref.shape[1], ref.shape[0]), Image.LANCZOS)
    image = np.asarray(image)
    ssim = structural_similarity(ref, image, channel_axis=-1, data_range=255)
    psnr = peak_signal_noise_ratio(ref, image, data_range=255)
    return float(ssim), float(psnr)
