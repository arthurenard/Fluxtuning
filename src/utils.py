import os
import time


def get_output_filename(base_filename: str, index: int) -> str:
    os.makedirs("generated_images", exist_ok=True)
    name, ext = os.path.splitext(base_filename)
    return os.path.join("generated_images", f"{name}_{index}{ext}")

def get_unique_seed() -> int:
    return int(time.time())

