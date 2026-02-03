import argparse

from src.captioning import caption_images

def main():
    parser = argparse.ArgumentParser(description="Generate FLUX fine-tuning captions with xAI Grok vision.")
    parser.add_argument("--image-dir", default="preprocessed_data")
    parser.add_argument("--subject", required=True, help="Name to use as trigger word in captions")
    parser.add_argument("--reference", required=True, help="Reference image filename or path")
    parser.add_argument("--output", default="data.json")
    parser.add_argument("--model", default="grok-4-1-fast-reasoning")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick tests")
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        help="Only caption specific filename(s) (repeatable), e.g. --only selfie.jpeg",
    )
    args = parser.parse_args()

    caption_images(
        args.image_dir,
        args.output,
        model=args.model,
        subject_name=args.subject,
        reference_image=args.reference,
        limit=args.limit,
        only=args.only,
    )

if __name__ == "__main__":
    main()
