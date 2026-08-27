from __future__ import annotations

POPE_TEMPLATE = "Is there a {obj} in the image?"
CHAIR_PROMPT = "Please describe this image in detail."


def pope_prompt(obj: str) -> str:
    return POPE_TEMPLATE.format(obj=obj)


def chat_prompt(question: str, with_image: bool = True) -> str:
    image_marker = "<image>\n" if with_image else ""
    return f"USER: {image_marker}{question} ASSISTANT:"