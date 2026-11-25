from pathlib import Path

from .data import mask_path, source_path


def build_jobs(protected, metadata, data_root, cfg, attacks_dir):
    stems = sorted({stem for _, _, stem, _ in protected})
    inputs = [("clean", "", stem, source_path(data_root, metadata[stem])) for stem in stems]
    jobs = []
    for method, variant, stem, input_path in inputs + list(protected):
        meta = metadata[stem]
        for mask in cfg["attack"]["masks"]:
            for prompt_key, template in cfg["prompts"].items():
                for seed in cfg["attack"]["seeds"]:
                    out = Path(attacks_dir, method, variant, mask, prompt_key, f"{stem}_s{seed}.png")
                    jobs.append({
                        "stem": stem,
                        "nationality": meta["nationality"],
                        "gender": meta["gender"],
                        "background": meta["background"],
                        "protection_method": method,
                        "protection_variant": variant,
                        "inpainting_mask": mask,
                        "prompt_key": prompt_key,
                        "prompt_text": template.format(gender=meta["gender"]),
                        "seed": seed,
                        "input_path": input_path,
                        "mask_path": mask_path(data_root, meta, mask),
                        "image_path": str(out),
                        "source_path": source_path(data_root, meta),
                    })
    return jobs


def run_attacks(jobs, attack_cfg):
    pending = [j for j in jobs if not Path(j["image_path"]).is_file()]
    print(f"[attack] {len(pending)} of {len(jobs)} attacks to run")
    if not pending:
        return

    import torch
    from diffusers import StableDiffusionInpaintPipeline
    from PIL import Image
    from tqdm import tqdm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, attack_cfg["dtype"]) if device == "cuda" else torch.float32
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        attack_cfg["model"],
        variant=attack_cfg["variant"],
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    size = attack_cfg["resolution"]

    for job in tqdm(pending, desc="attacks"):
        image = Image.open(job["input_path"]).convert("RGB")
        mask = Image.open(job["mask_path"]).convert("L")
        if image.size != (size, size):
            image = image.resize((size, size), Image.LANCZOS)
        if mask.size != (size, size):
            mask = mask.resize((size, size), Image.NEAREST)
        result = pipe(
            prompt=job["prompt_text"],
            image=image,
            mask_image=mask,
            height=size,
            width=size,
            guidance_scale=attack_cfg["guidance_scale"],
            num_inference_steps=attack_cfg["num_inference_steps"],
            generator=torch.Generator(device).manual_seed(job["seed"]),
        ).images[0]
        out = Path(job["image_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        result.save(out)
