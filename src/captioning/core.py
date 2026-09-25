import os
import base64
import json
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

from .caption_prompt import caption_prompt


def _make_client() -> OpenAI:
    """Create the xAI client, reading XAI_API_KEY from the environment, .env.local or .env."""
    env_path = Path(".env.local")
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()

    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise SystemExit("Error: XAI_API_KEY is not set. Export it, or add it to .env.local or .env.")

    return OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")

def _mime_type_for_path(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    return "application/octet-stream"


def encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')



def caption_images(
    image_dir: str,
    output_file: str,
    subject_name: str,
    reference_image: str,
    model: str = "grok-4-1-fast-reasoning",
    limit: Optional[int] = None,
    only: Optional[list[str]] = None,
):
    image_extensions = (".png", ".jpg", ".jpeg", ".webp")
    image_dir_path = Path(image_dir)
    
    if not image_dir_path.exists():
        print(f"Error: Directory {image_dir} does not exist.")
        return

    reference_path = (image_dir_path / reference_image) if not Path(reference_image).exists() else Path(reference_image)
    if not reference_path.exists():
        print(
            f"Error: Reference image not found at {reference_path}. "
            f"Expected '{reference_image}' inside '{image_dir}'."
        )
        return

    # Get all image files and sort them to ensure consistent order
    image_paths = sorted([p for p in image_dir_path.iterdir() if p.suffix.lower() in image_extensions])
    
    if not image_paths:
        print(f"No images found in {image_dir}")
        return

    if only:
        wanted = set(only)
        image_paths = [p for p in image_paths if p.name in wanted or str(p) in wanted]
        
    if limit is not None:
        image_paths = image_paths[: max(0, limit)]

    print(f"Found {len(image_paths)} images. Starting generation with model {model}...")

    client = _make_client()
    data = []
    failed = []
    prompt = caption_prompt(subject_name=subject_name)
    reference_b64 = encode_image(reference_path)
    reference_mime = _mime_type_for_path(reference_path)
    
    for i, image_path in enumerate(image_paths):
        print(f"[{i+1}/{len(image_paths)}] Processing {image_path.name}...")
        
        try:
            target_b64 = encode_image(image_path)
            target_mime = _mime_type_for_path(image_path)
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt + "\nImage 1 (reference selfie):"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{reference_mime};base64,{reference_b64}",
                                    "detail": "high",
                                },
                            },
                            {"type": "text", "text": "Image 2 (caption this image):"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{target_mime};base64,{target_b64}",
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ],
                temperature=0.3,
                max_tokens=200,
            )
            
            description = response.choices[0].message.content.strip()
            
            data.append({
                "id": image_path.name,
                "description": description
            })
            print(f"  Description: {description}")
            
        except Exception as e:
            # Skip the image: a placeholder caption would end up in the training data
            print(f"  Error processing {image_path.name}: {e}")
            failed.append(image_path.name)

    # Save to JSON
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)
    
    print(f"\nSuccessfully generated {output_file} with {len(data)} entries.")
    if failed:
        print(
            f"Skipped {len(failed)} image(s) that could not be captioned: {', '.join(failed)}. "
            "Caption them again with --only <filename> and a different --output file."
        )