def caption_prompt(subject_name: str) -> str:
    return (
        f"You are helping prepare LoRA training captions for black-forest-labs/FLUX.1-dev.\n"
        f"You will always see two images: Image 1 is a reference photo of {subject_name}; Image 2 is the image to caption.\n"
        f"Write one or two natural sentences (no bullet list) that starts with \"{subject_name}\".\n"
        f"Note any distinguishing features of {subject_name} in the reference image and only mention them in captions if they change.\n"
        f"If other people are present:\n"
        f"- Mention them simply (e.g. \"posing with a friend\", \"standing with two people\", \"with a group\").\n"
        f"- Do NOT describe other people's appearance in detail.\n"
        f"- Explicitly state where {subject_name} is in the frame (e.g., \"on the left\", \"in the center\", \"on the right\").\n"
        f"- Focus on {subject_name}.\n"
        f"Describe what {subject_name} is doing, the setting or background, lighting/time of day, framing (selfie, mirror selfie, wide shot, close-up).\n"
        f"Do not add explanations or extra sentences. Output only the caption."
    )
