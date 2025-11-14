import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from accelerate import Accelerator
from PIL import Image
from transformers.pipelines import pipeline

from looky.dwpose import DWposeDetector
from looky.pipelines.pipeline_virtual_try_on import VirtualTryOnPipeline


def main():
    category = "upper-body"

    image = Image.open("data/test/person/000001.jpg").convert("RGB")
    garment_image = Image.open("data/test/garment/000001.jpg").convert("RGB")

    device = Accelerator().device

    detector = DWposeDetector(device=device)
    pose_image, keypoints, scores = detector(image)

    generator = pipeline(
        task="agnostic-mask-generation",
        model="./weights/human_parsing",
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    outputs = generator(
        image, category=category, keypoints=keypoints[0], scores=scores[0]
    )
    mask_image = outputs["mask"]

    pipe = VirtualTryOnPipeline.from_pretrained(
        "./weights/virtual_try_on",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    pipe.to(device)

    image = pipe(
        image=image,
        mask_image=mask_image,
        garment_image=garment_image,
        pose_image=pose_image,
        height=1024,
        width=768,
        num_inference_steps=20,
        guidance_scale=2.0,
    ).images[0]

    image.save("image.jpg")


if __name__ == "__main__":
    main()
