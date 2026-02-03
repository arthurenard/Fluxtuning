from __future__ import annotations

import argparse
import os
from typing import Any

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from diffusers import FlowMatchEulerDiscreteScheduler, FluxPipeline
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)

from src.prepare_model import prepare_model
from .data_utils import FluxDataModule, SampleCallback, SaveLoRACallback


class FluxFineTuner(pl.LightningModule):
    def __init__(
        self,
        *,
        model_id: str,
        image_size: int,
        max_sequence_length: int,
        lora_rank: int,
        lora_alpha: int | None,
        lora_dropout: float,
        learning_rate: float,
        weight_decay: float,
        guidance_scale: float,
        weighting_scheme: str,
        logit_mean: float,
        logit_std: float,
        mode_scale: float,
        gradient_checkpointing: bool,
        dtype: str,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.strict_loading = False

        torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(dtype, torch.bfloat16)
        self.pipe = prepare_model(model_id, lora_rank, dtype=torch_dtype, device=None, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

        self.transformer, self.vae = self.pipe.transformer, self.pipe.vae
        self.text_encoder, self.text_encoder_2 = self.pipe.text_encoder, self.pipe.text_encoder_2

        for m in [self.vae, self.text_encoder, self.text_encoder_2]:
            m.requires_grad_(False)
            m.eval()
        if gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.register_buffer("_schedule_timesteps", torch.tensor(self.noise_scheduler.timesteps, dtype=torch.float32), persistent=False)
        self.register_buffer("_schedule_sigmas", torch.tensor(self.noise_scheduler.sigmas, dtype=torch.float32), persistent=False)
        self.vae_scale_factor = getattr(
            self.pipe,
            "vae_scale_factor",
            2 ** (len(self.vae.config.block_out_channels) - 1),
        )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["state_dict"] = {
            k: v for k, v in checkpoint["state_dict"].items() 
            if self.get_parameter(k).requires_grad
        }

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        images, prompts = batch["image"].to(dtype=self.vae.dtype), batch["prompt"]
        bsz, _, height, width = images.shape

        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
            prompt_embeds, pooled_prompt_embeds, text_ids = self.pipe.encode_prompt(
                prompt=prompts, prompt_2=prompts, device=self.device, max_sequence_length=self.hparams.max_sequence_length)

        latents, noise = latents.to(dtype=self.transformer.dtype), torch.randn_like(latents)
        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.hparams.weighting_scheme, batch_size=bsz, logit_mean=self.hparams.logit_mean,
            logit_std=self.hparams.logit_std, mode_scale=self.hparams.mode_scale).to(device=self.device)

        n = self._schedule_sigmas.shape[0]
        indices = (u * n).long().clamp(0, n - 1)

        timesteps = self._schedule_timesteps[indices].to(self.device)
        sigmas = self._schedule_sigmas[indices].to(self.device, dtype=latents.dtype).view(bsz, 1, 1, 1)

        noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
        packed_noisy = FluxPipeline._pack_latents(noisy_latents, bsz, latents.shape[1], latents.shape[2], latents.shape[3])
        img_ids = FluxPipeline._prepare_latent_image_ids(
            bsz,
            latents.shape[2] // 2,
            latents.shape[3] // 2,
            self.device,
            latents.dtype,
        )
        
        guidance = torch.full((bsz,), float(self.hparams.guidance_scale), device=self.device, dtype=torch.float32) if getattr(self.transformer.config, "guidance_embeds", False) else None

        model_pred = self.transformer(hidden_states=packed_noisy, timestep=timesteps / 1000, guidance=guidance,
                                      pooled_projections=pooled_prompt_embeds, encoder_hidden_states=prompt_embeds,
                                      txt_ids=text_ids, img_ids=img_ids, return_dict=False)[0]

        model_pred = FluxPipeline._unpack_latents(model_pred, height=height,
                                                 width=width,
                                                 vae_scale_factor=self.vae_scale_factor)

        weighting = compute_loss_weighting_for_sd3(weighting_scheme=self.hparams.weighting_scheme, sigmas=sigmas)
        loss = torch.mean((weighting.float() * (model_pred.float() - (noise - latents).float()) ** 2).reshape(bsz, -1), 1).mean()

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.transformer.parameters())
        return torch.optim.AdamW(
            params,
            lr=float(self.hparams.learning_rate),
            weight_decay=float(self.hparams.weight_decay),
        )


def _resolve_precision(arg: str) -> str:
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"
    return "16-mixed" if torch.backends.mps.is_available() else "32-true"

def _resolve_model_dtype(dtype_arg: str, *, precision: str) -> str:
    if dtype_arg != "auto":
        return dtype_arg
    return "bf16" if precision.startswith("bf16") else "fp16" if precision.startswith("16") else "fp32"

def _resolve_accelerator() -> str:
    if torch.cuda.is_available():
        return "gpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

@rank_zero_only
def _mkdirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def train(args) -> None:
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    accel, precision = _resolve_accelerator(), _resolve_precision(args.precision)
    model_dtype = _resolve_model_dtype(args.dtype, precision=precision)
    strategy = args.strategy or ("ddp_find_unused_parameters_false" if accel == "gpu" and args.devices > 1 else "auto")

    # Resolve paths
    dataset_path = args.dataset
    raw_images_dir = args.images or os.path.dirname(os.path.abspath(dataset_path))
    output_dir = args.output_dir
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    samples_dir = os.path.join(output_dir, "samples")
    processed_images_dir = os.path.join(os.getcwd(), "preprocessed_images")

    _mkdirs(output_dir, ckpt_dir, samples_dir, processed_images_dir)

    datamodule = FluxDataModule(
        json_path=dataset_path,
        raw_images_dir=raw_images_dir, 
        processed_images_dir=processed_images_dir,
        image_size=args.image_size, 
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    model = FluxFineTuner(
        model_id=args.model_id, 
        image_size=args.image_size, 
        max_sequence_length=args.max_sequence_length,
        lora_rank=args.lora_rank, 
        lora_alpha=args.lora_alpha, 
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate, 
        weight_decay=args.weight_decay, 
        guidance_scale=args.guidance_scale,
        weighting_scheme=args.weighting_scheme,
        logit_mean=args.logit_mean, 
        logit_std=args.logit_std,
        mode_scale=args.mode_scale,
        gradient_checkpointing=args.gradient_checkpointing,
        dtype=model_dtype
        )

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="step{step}",
            save_last=args.save_last,
            every_n_train_steps=args.checkpoint_every_n_steps if args.checkpoint_every_n_steps > 0 else None
        ),
        SampleCallback(
            output_dir=samples_dir,
            every_n_epochs=args.sample_every_n_epochs,
            prompts=args.sample_prompts,
            num_inference_steps=args.sample_steps,
            guidance_scale=args.sample_guidance_scale,
            seed=args.sample_seed,
        ),
        SaveLoRACallback(output_dir=output_dir, every_n_epochs=args.save_every_n_epochs),
    ]

    trainer = pl.Trainer(
        max_epochs=args.max_epochs or 100000,
        accelerator=accel,
        devices=args.devices,
        strategy=strategy,
        precision=precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        enable_checkpointing=True,
        callbacks=callbacks
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume_from_checkpoint)
