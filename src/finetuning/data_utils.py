from __future__ import annotations

import json
import os
from typing import Any, Iterator, Sequence

import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms

import pytorch_lightning as pl


class BucketManager:
    def __init__(self, target_area: int, step_size: int = 64):
        self.target_area = target_area
        self.step_size = step_size
        self.buckets: list[tuple[int, int]] = self._generate_buckets()

    def _generate_buckets(self) -> list[tuple[int, int]]:
        buckets = set()
        # Start with square
        base_size = int(self.target_area**0.5)
        base_size = (base_size // self.step_size) * self.step_size
        
        # Search for combinations (w, h) such that w * h approx target_area
        # and w, h are multiples of step_size
        for w in range(self.step_size, 2048 + self.step_size, self.step_size):
            h = (self.target_area // w // self.step_size) * self.step_size
            if h > 0:
                # Add (w, h) and (h, w)
                if abs(w * h - self.target_area) < self.target_area * 0.5:
                    buckets.add((w, h))
            
            # Also try h + step_size to see if it's closer
            h_plus = h + self.step_size
            if abs(w * h_plus - self.target_area) < self.target_area * 0.5:
                buckets.add((w, h_plus))

        # Sort by aspect ratio
        sorted_buckets = sorted(list(buckets), key=lambda x: x[0] / x[1])
        return sorted_buckets

    def get_bucket(self, width: int, height: int) -> tuple[int, int]:
        aspect_ratio = width / height
        best_bucket = min(self.buckets, key=lambda b: abs((b[0] / b[1]) - aspect_ratio))
        return best_bucket


class AspectRatioBatchSampler(Sampler[list[int]]):
    def __init__(self, dataset: FluxDataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.bucket_to_indices: dict[tuple[int, int], list[int]] = {}
        for idx, item in enumerate(dataset.data):
            bucket = item["bucket"]
            if bucket not in self.bucket_to_indices:
                self.bucket_to_indices[bucket] = []
            self.bucket_to_indices[bucket].append(idx)

    def __iter__(self) -> Iterator[list[int]]:
        all_batches = []
        for bucket, indices in self.bucket_to_indices.items():
            if self.shuffle:
                perm = torch.randperm(len(indices)).tolist()
                indices = [indices[i] for i in perm]
            
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                if len(batch) == self.batch_size:
                    all_batches.append(batch)
        
        if self.shuffle:
            perm = torch.randperm(len(all_batches)).tolist()
            all_batches = [all_batches[i] for i in perm]
            
        yield from all_batches

    def __len__(self) -> int:
        count = 0
        for indices in self.bucket_to_indices.values():
            count += len(indices) // self.batch_size
        return count


def preprocess_images(
    json_path: str, 
    src_dir: str, 
    dst_dir: str, 
    *, 
    target_area: int,
    save_bucket: bool = False
) -> list[dict[str, Any]]:
    os.makedirs(dst_dir, exist_ok=True)
    with open(json_path, "r") as f:
        data = json.load(f)

    bucket_manager = BucketManager(target_area)
    resample = getattr(Image, "Resampling", Image).LANCZOS
    
    processed_data = []
    print(f"Preprocessing {len(data)} images -> {dst_dir}")
    
    for item in data:
        src_path = os.path.join(src_dir, item["id"])
        dst_path = os.path.join(dst_dir, item["id"])
        
        try:
            with Image.open(src_path) as img:
                img = ImageOps.exif_transpose(img)
                w, h = img.size
                bucket_w, bucket_h = bucket_manager.get_bucket(w, h)
                
                new_item = item.copy()
                new_item["bucket"] = (bucket_w, bucket_h)
                processed_data.append(new_item)

                if os.path.exists(dst_path):
                    continue
                
                img = img.convert("RGB")
                img.resize((bucket_w, bucket_h), resample=resample).save(dst_path)
                
        except Exception as e:
            print(f"Error processing {src_path}: {e}")
            
    return processed_data


class FluxDataset(Dataset):
    def __init__(self, data: list[dict[str, Any]], img_dir: str):
        self.data = data
        self.img_dir = img_dir
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.data[idx]
        img_path = os.path.join(self.img_dir, item["id"])

        with Image.open(img_path) as img:
            rgb_img = img.convert("RGB")
            transformed_img = self.transform(rgb_img)

        return {"prompt": item["description"], "image": transformed_img}


class FluxDataModule(pl.LightningDataModule):
    def __init__(
        self,
        *,
        json_path: str,
        raw_images_dir: str,
        processed_images_dir: str,
        image_size: int,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()
        self.json_path = json_path
        self.raw_images_dir = raw_images_dir
        self.processed_images_dir = processed_images_dir
        self.target_area = image_size * image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self._train_dataset: FluxDataset | None = None
        self._processed_data: list[dict[str, Any]] = []

    def prepare_data(self) -> None:
        self._processed_data = preprocess_images(
            self.json_path,
            self.raw_images_dir,
            self.processed_images_dir,
            target_area=self.target_area,
        )

    def setup(self, stage: str | None = None) -> None:
        if not self._processed_data:
            self.prepare_data()
        self._train_dataset = FluxDataset(self._processed_data, self.processed_images_dir)

    def train_dataloader(self) -> DataLoader:
        assert self._train_dataset is not None
        sampler = AspectRatioBatchSampler(
            self._train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        pin_memory = torch.cuda.is_available()
        return DataLoader(
            self._train_dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
            persistent_workers=self.num_workers > 0,
        )


class SaveLoRACallback(pl.Callback):
    def __init__(self, *, output_dir: str, every_n_epochs: int):
        super().__init__()
        self.output_dir = output_dir
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        if self.every_n_epochs <= 0:
            return
        epoch_1 = trainer.current_epoch + 1
        if epoch_1 % self.every_n_epochs != 0:
            return

        out = os.path.join(self.output_dir, f"lora_epoch_{epoch_1}")
        os.makedirs(out, exist_ok=True)
        pl_module.transformer.save_pretrained(out)
        print(f"[rank0] Saved LoRA adapter to {out}")


class SampleCallback(pl.Callback):
    def __init__(
        self,
        *,
        output_dir: str,
        every_n_epochs: int,
        prompts: list[str] | None,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int = 42,
    ):
        super().__init__()
        self.output_dir = output_dir
        self.every_n_epochs = every_n_epochs
        self.prompts = prompts or []
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.seed = seed

    def _sample(self, trainer: pl.Trainer, pl_module: pl.LightningModule, prefix: str) -> None:
        if not trainer.is_global_zero or not self.prompts:
            return

        os.makedirs(self.output_dir, exist_ok=True)
        modules = [pl_module.transformer, pl_module.vae, pl_module.text_encoder, pl_module.text_encoder_2]
        states = [m.training for m in modules]

        for m in modules:
            m.eval()
        
        with torch.inference_mode():
            for i, prompt in enumerate(self.prompts):
                generator = torch.Generator(device=pl_module.device).manual_seed(self.seed)
                safe_prompt = "".join([c if c.isalnum() else "_" for c in prompt])[:50]
                save_path = os.path.join(self.output_dir, f"{prefix}_sample_{i}_{safe_prompt}.png")
                
                pl_module.pipe(
                    prompt,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    generator=generator,
                ).images[0].save(save_path)
                print(f"[rank0] Saved sample -> {save_path}")

        for m, was_training in zip(modules, states):
            if was_training:
                m.train()

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._sample(trainer, pl_module, "initial")

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.every_n_epochs <= 0 or (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        self._sample(trainer, pl_module, f"epoch_{trainer.current_epoch + 1}")
