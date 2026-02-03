import argparse
from src.finetuning.finetune_model import train

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    g_data = p.add_argument_group("Data")
    g_data.add_argument("--dataset", default="data.json", help="Path to the JSON file containing captions and image IDs")
    g_data.add_argument("--images", help="Directory containing the raw images (defaults to the dataset's directory)")
    g_data.add_argument("--output_dir", default="output", help="Directory for checkpoints and samples")
    g_data.add_argument("--image_size", type=int, default=1024)
    g_data.add_argument("--batch_size", type=int, default=8)
    g_data.add_argument("--num_workers", type=int, default=0)

    g_model = p.add_argument_group("Model")
    g_model.add_argument("--model_id", default="black-forest-labs/FLUX.1-dev")
    g_model.add_argument("--max_sequence_length", type=int, default=256)
    g_model.add_argument("--lora_rank", type=int, default=32)
    g_model.add_argument("--lora_alpha", type=int, default=16)
    g_model.add_argument("--lora_dropout", type=float, default=0.0)
    g_model.add_argument("--learning_rate", type=float, default=1e-4)
    g_model.add_argument("--weight_decay", type=float, default=1e-4)
    g_model.add_argument("--guidance_scale", type=float, default=1.0)
    g_model.add_argument("--gradient_clip_val", type=float, default=1.0)
    g_model.add_argument("--weighting_scheme", default="none", choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"])
    g_model.add_argument("--logit_mean", type=float, default=0.0)
    g_model.add_argument("--logit_std", type=float, default=1.0)
    g_model.add_argument("--mode_scale", type=float, default=1.29)
    g_model.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    g_model.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])

    g_train = p.add_argument_group("Trainer")
    g_train.add_argument("--max_epochs", type=int, default=None, help="Maximum number of epochs (default: unlimited)")
    g_train.add_argument("--accumulate_grad_batches", type=int, default=1)
    g_train.add_argument("--devices", type=int, default=1)
    g_train.add_argument("--strategy")
    g_train.add_argument("--precision", default="auto")

    g_side = p.add_argument_group("Side Effects")
    g_side.add_argument("--checkpoint_every_n_steps", type=int, default=50)
    g_side.add_argument("--save_last", action=argparse.BooleanOptionalAction, default=True)
    g_side.add_argument("--resume_from_checkpoint", help="Path to .ckpt file or 'last'")
    g_side.add_argument("--save_every_n_epochs", type=int, default=50)
    g_side.add_argument("--sample_every_n_epochs", type=int, default=25)
    g_side.add_argument("--sample_prompts", nargs="+", default=None)
    g_side.add_argument("--sample_steps", type=int, default=30)
    g_side.add_argument("--sample_guidance_scale", type=float, default=3.5)
    g_side.add_argument("--sample_seed", type=int, default=42)

    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
