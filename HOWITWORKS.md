# How Finetuning Works (Math)

This document describes the exact training math implemented in `src/finetune_model.py`
and `src/prepare_model.py`. Only LoRA adapters on the FLUX transformer are trained.

## Notation

- `x_raw` : input image pixels in `[0, 255]` with shape `(H, W, 3)`
- `x` : normalized image tensor in `[-1, 1]` with shape `(3, H, W)`
- `z` : VAE latent with shape `(C, H', W')`
- `epsilon` : standard Gaussian noise, `epsilon ~ N(0, I)`
- `sigma` : noise level from the scheduler
- `t` : discrete timestep paired with `sigma`
- `f_theta` : FLUX transformer with LoRA parameters `theta`
- `w(sigma)` : loss weight from `compute_loss_weighting_for_sd3`

## 1) Data Normalization

Each image is resized to its assigned aspect ratio bucket (see README for details on bucketing), converted to RGB, and normalized:

`x = 2 * (x_raw / 255) - 1`

This is exactly the `ToTensor()` + `Normalize([0.5]*3, [0.5]*3)` pipeline.

## 2) VAE Encoding (Frozen)

The VAE is frozen. For each image:

1. Sample from the VAE posterior:
   `z_tilde ~ q_phi(z | x)`
2. Apply model-specific shift/scale:
   `z = (z_tilde - shift_factor) * scaling_factor`

The resulting `z` is the clean latent used for training.

## 3) LoRA Parameterization (Trainable Weights)

LoRA is applied to the transformer attention projections:
`to_q`, `to_k`, `to_v`, `to_out.0`.

For each weight matrix `W`:

`W' = W + DeltaW`

`DeltaW = (alpha / r) * B * A`

Where:
- `A` has shape `(r, d_in)` and `B` has shape `(d_out, r)`
- `r` is `lora_rank`
- `alpha` is `lora_alpha`

Only `A` and `B` are trainable; all original weights are frozen.

## 4) Noise Schedule and Corruption

The scheduler provides arrays of `sigmas` and `timesteps`.

Sampling:

1. Draw `u` from the distribution defined by
   `compute_density_for_timestep_sampling(...)`.
2. Let `n = len(sigmas)`.
3. `i = floor(u * n)`, clamped to `[0, n-1]`.
4. `sigma = sigmas[i]`, `t = timesteps[i]`.

Corrupt the latent with a linear interpolation:

`z_sigma = (1 - sigma) * z + sigma * epsilon`

This is the exact noisy latent passed to the transformer.

## 5) Conditioning and Model Prediction

Prompts are encoded by two frozen text encoders to produce:
`prompt_embeds`, `pooled_prompt_embeds`, and `text_ids`.

The transformer predicts a latent-sized tensor:

`v_hat = f_theta(z_sigma, t / 1000, prompt_embeds, pooled_prompt_embeds, text_ids, guidance)`

(`guidance` is only provided if the model supports guidance embeddings.)

## 6) Training Objective (Flow Matching)

The target for flow matching is the constant velocity along the straight line
between `z` and `epsilon`:

`v = epsilon - z`

The weighted MSE loss is:

`L = mean_batch( mean_pixels( w(sigma) * || v_hat - v ||^2 ) )`

Where `w(sigma)` is computed by `compute_loss_weighting_for_sd3` using the
selected `weighting_scheme`.

## 7) Optimization

Only LoRA parameters are optimized with AdamW:

`theta <- AdamW(theta, lr, weight_decay)`

Optional gradient accumulation and clipping are handled by the trainer but do
not change the mathematical objective above.
