import sys
import argparse
import json
from pathlib import Path
from mflux.models.flux.variants.txt2img.flux import Flux1
from mflux.config.model_config import ModelConfig
from mflux.config.config import Config
from src.utils import get_output_filename, get_unique_seed
from safetensors.torch import load_file, save_file

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate images with mflux + converted LoRA")
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--num_images", type=int, default=1)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--guidance_scale", type=float, default=3.5)
    p.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="28 is a good quality/speed default for FLUX; increase to 40-50 for max quality.",
    )
    p.add_argument("--quantize", type=int, default=8, choices=[0, 4, 8])
    p.add_argument("--lora_path", type=str, default="checkpoints/lora_epoch_2200")
    p.add_argument("--lora_scale", type=float, default=1.0)
    p.add_argument("--output_filename", type=str, default="output.png")
    p.add_argument("--prompt_file", type=str, default="prompts.json")
    return p.parse_args()

def ensure_mflux_lora(lora_dir: str) -> str:
    lora_path = Path(lora_dir)
    mflux_lora_path = lora_path / "adapter_model_mflux.safetensors"
    
    if mflux_lora_path.exists():
        return str(mflux_lora_path)
    
    # If mflux lora doesn't exist, check for original adapter_model.safetensors
    original_lora_path = lora_path / "adapter_model.safetensors"
    if not original_lora_path.exists():
        print(f"Error: LoRA file not found at {original_lora_path}")
        sys.exit(1)
        
    print(f"Converting {original_lora_path} to mflux format...")
    state_dict = load_file(str(original_lora_path))
    new_state_dict = {}
    
    # diffusers PeftModel adds "base_model.model." prefix.
    # mflux expects "transformer." prefix for the Flux transformer.
    prefix_map = {
        "base_model.model.": "transformer.",
    }
    
    for k, v in state_dict.items():
        new_k = k
        for old, new in prefix_map.items():
            if new_k.startswith(old):
                new_k = new_k.replace(old, new, 1)
                break
        new_state_dict[new_k] = v
        
    save_file(new_state_dict, str(mflux_lora_path))
    print(f"Conversion complete: {mflux_lora_path}")
    return str(mflux_lora_path)

def main():
    args = _parse_args()
    
    # Ensure LoRA exists and is converted
    mflux_lora_file = ensure_mflux_lora(args.lora_path)

    print(f"Initializing Flux1 model with LoRA: {mflux_lora_file}")
    flux = Flux1(
        model_config=ModelConfig.from_name("krea-dev"),
        quantize=args.quantize,
        lora_paths=[mflux_lora_file],
        lora_scales=[args.lora_scale],
    )
    
    # Prompt logic similar to main.py
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
    
    print(f"Generating image(s) with configuration: {args.num_inference_steps} steps, {args.height}x{args.width}, guidance {args.guidance_scale}")
    config = Config(
        num_inference_steps=args.num_inference_steps,
        height=args.height,
        width=args.width,
        guidance=args.guidance_scale,
    )
    
    for task in tasks:
        prompt = task["prompt"]
        output_filename = task.get("output_filename", args.output_filename)
        print(f"Generating {args.num_images} image(s) for prompt: {prompt!r}")
        
        for i in range(args.num_images):
            image = flux.generate_image(
                seed=get_unique_seed() + i, 
                prompt=prompt,
                config=config
            )
            
            filename = get_output_filename(output_filename, i)
            image.save(path=filename)
            print(f"Saved {filename}")

if __name__ == "__main__":
    main()
