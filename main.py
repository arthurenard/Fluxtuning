import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from diffusers import FluxPipeline
from peft import PeftModel

from src.utils import get_output_filename

def get_device() -> torch.device:
    if torch.cuda.is_available(): 
        return torch.device("cuda")
    if torch.backends.mps.is_available(): 
        return torch.device("mps")
    return torch.device("cpu")

def _resolve_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate images with FLUX + LoRA")
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--num_images", type=int, default=3)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--guidance_scale", type=float, default=3.5)
    p.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Denoising steps. The default of 50 favours quality; 28 is a good quality/speed trade-off.",
    )
    p.add_argument("--lora_path", type=str, required=True, help="Path to LoRA adapter directory")
    p.add_argument("--output_filename", type=str, default="output.png")
    p.add_argument("--prompt_file", type=str, default="prompts.json")
    p.add_argument("--seed", type=int, default=None, help="Random seed for generation")
    return p.parse_args()

def main():
    args = _parse_args()
    device = get_device()
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    dtype = _resolve_dtype(device)
    print(f"Using device: {device}")

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", 
        torch_dtype=dtype
    ).to(device)
    
    print(f"Loading LoRA adapter from: {args.lora_path}")
    pipe.transformer = PeftModel.from_pretrained(pipe.transformer, args.lora_path)
    
    if args.prompt is None:
        prompts_path = Path(args.prompt_file)
        if prompts_path.exists():
            with open(prompts_path, "r") as f:
                tasks = json.load(f)
            print(f"No prompt provided. Loaded {len(tasks)} prompts from {prompts_path}")
        else:
            default_prompt = "A realistic picture of a man."
            print(f"No prompt provided and {prompts_path} not found. Using default prompt.")
            tasks = [{"prompt": default_prompt, "output_filename": args.output_filename}]
    else:
        tasks = [{"prompt": args.prompt, "output_filename": args.output_filename}]

    for task in tasks:
        prompt = task["prompt"]
        output_filename = task.get("output_filename", args.output_filename)
        print(f"Generating {args.num_images} image(s) for prompt: {prompt!r}")

        seed = args.seed if args.seed is not None else int(torch.randint(0, 1000000, (1,)).item())
        generator = torch.Generator(device=device).manual_seed(seed)
        print(f"Using seed: {seed}")

        with torch.inference_mode():
            images = pipe(
                prompt,
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                num_images_per_prompt=args.num_images,
                generator=generator,
            ).images

        for i, image in enumerate(images):
            filename = get_output_filename(output_filename, i)
            image.save(filename)
            print(f"Saved {filename}")

if __name__ == "__main__":
    main()
