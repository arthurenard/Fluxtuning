"""Convert diffusers/PEFT LoRA weights to mflux format.

Usage:
    python convert_lora.py --input lora_epoch_500 --output lora_epoch_500/adapter_model_mflux.safetensors
    
Or with just a directory (auto-detects input and output paths):
    python convert_lora.py --lora_dir lora_epoch_500
"""
import argparse
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file


def convert_lora(input_path: str, output_path: str) -> None:
    """Convert a diffusers/PEFT LoRA to mflux format."""
    if not Path(input_path).exists():
        print(f"Error: {input_path} not found.")
        sys.exit(1)

    print(f"Loading {input_path}...")
    state_dict = load_file(input_path)
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

    print(f"Saving {len(new_state_dict)} keys to {output_path}...")
    save_file(new_state_dict, output_path)
    print("Conversion complete.")


def main():
    parser = argparse.ArgumentParser(description="Convert diffusers/PEFT LoRA to mflux format")
    parser.add_argument("--input", type=str, help="Path to input adapter_model.safetensors")
    parser.add_argument("--output", type=str, help="Path to output mflux safetensors file")
    parser.add_argument("--lora_dir", type=str, help="LoRA directory (auto-detects input/output)")
    args = parser.parse_args()

    if args.lora_dir:
        lora_dir = Path(args.lora_dir)
        input_path = str(lora_dir / "adapter_model.safetensors")
        output_path = str(lora_dir / "adapter_model_mflux.safetensors")
    elif args.input and args.output:
        input_path = args.input
        output_path = args.output
    else:
        parser.error("Provide either --lora_dir or both --input and --output")

    convert_lora(input_path, output_path)


if __name__ == "__main__":
    main()

