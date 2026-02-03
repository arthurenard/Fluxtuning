import torch
from diffusers import FluxPipeline
from peft import get_peft_model, LoraConfig


def prepare_model(
    name: str,
    lora_r: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str | None = None,
    lora_alpha: int | None = None,
    lora_dropout: float = 0.0,
    target_modules: list[str] | None = None,
):
    if lora_alpha is None:
        lora_alpha = lora_r
    if target_modules is None:
        target_modules = ["to_q", "to_k", "to_v", "to_out.0"]

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )

    pipe = FluxPipeline.from_pretrained(name, torch_dtype=dtype)
    pipe.transformer = get_peft_model(pipe.transformer, lora_config)

    if device is not None:
        pipe = pipe.to(device)

    return pipe
