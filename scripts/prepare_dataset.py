import gc
import os
from functools import partial
from typing import List

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import Dataset, DatasetDict, Image
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from transformers.feature_extraction_utils import BatchFeature


def compute_embeddings(
    examples,
    image_processor: CLIPImageProcessor,
    image_encoders: List[CLIPVisionModelWithProjection],
) -> BatchFeature:
    images = examples["garment_image"]
    pixel_values = image_processor(images, return_tensors="pt").pixel_values

    with torch.no_grad():
        image_embeds = []
        for image_encoder in image_encoders:
            pixel_values = pixel_values.to(
                device=image_encoder.device, dtype=image_encoder.dtype
            )
            image_embed = image_encoder(pixel_values).image_embeds
            image_embeds.append(image_embed)

    image_embeds = torch.cat(image_embeds, dim=-1)
    image_embeds = image_embeds.float().cpu().numpy()

    return {"garment_image_embeds": image_embeds}


def compute_encodings(
    examples,
    image_processor: VaeImageProcessor,
    mask_processor: VaeImageProcessor,
    vae: AutoencoderKL,
) -> torch.Tensor:
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    image = examples["person_image"]
    image = image_processor.preprocess(image)
    image = image.to(device=vae.device, dtype=vae.dtype)

    with torch.no_grad():
        image_latents = vae.encode(image).latent_dist.sample()
        image_latents = (
            image_latents - vae.config.shift_factor
        ) * vae.config.scaling_factor

    image_latents = image_latents.float().cpu().numpy()

    garment_image = examples["garment_image"]
    garment_image = image_processor.preprocess(garment_image)
    garment_image = garment_image.to(device=vae.device, dtype=vae.dtype)

    with torch.no_grad():
        garment_image_latents = vae.encode(garment_image).latent_dist.sample()
        garment_image_latents = (
            garment_image_latents - vae.config.shift_factor
        ) * vae.config.scaling_factor

    garment_image_latents = garment_image_latents.float().cpu().numpy()

    mask_image = examples["mask_image"]
    mask_image = mask_processor.preprocess(mask_image)
    mask_image = mask_image.to(device=vae.device, dtype=vae.dtype)

    height, width = image.shape[-2:]
    mask = F.interpolate(
        mask_image,
        size=(height // vae_scale_factor, width // vae_scale_factor),
    )
    mask = mask.float().cpu().numpy()

    masked_images = image * (mask_image < 0.5)

    with torch.no_grad():
        masked_image_latents = vae.encode(masked_images).latent_dist.sample()
        masked_image_latents = (
            masked_image_latents - vae.config.shift_factor
        ) * vae.config.scaling_factor

    masked_image_latents = masked_image_latents.float().cpu().numpy()

    pose_image = examples["pose_image"]
    pose_image = image_processor.preprocess(pose_image)
    pose_image = pose_image.float().cpu().numpy()

    return {
        "image_latents": image_latents,
        "garment_image_latents": garment_image_latents,
        "masked_image_latents": masked_image_latents,
        "mask": mask,
        "pose_cond": pose_image,
    }


def main():
    device = Accelerator().device

    parquet_path = os.path.join("data", "train", "metadata.parquet")
    train_dataset = Dataset.from_parquet(parquet_path)

    train_dataset = train_dataset.cast_column("person_image", Image())
    train_dataset = train_dataset.cast_column("garment_image", Image())
    train_dataset = train_dataset.cast_column("pose_image", Image())
    train_dataset = train_dataset.cast_column("mask_image", Image())
    train_dataset = train_dataset.cast_column("agnostic_mask_image", Image())

    parquet_path = os.path.join("data", "test", "metadata.parquet")
    test_dataset = Dataset.from_parquet(parquet_path)

    test_dataset = test_dataset.cast_column("person_image", Image())
    test_dataset = test_dataset.cast_column("garment_image", Image())
    test_dataset = test_dataset.cast_column("pose_image", Image())
    test_dataset = test_dataset.cast_column("mask_image", Image())
    test_dataset = test_dataset.cast_column("agnostic_mask_image", Image())

    image_processor = CLIPImageProcessor()
    image_encoder_one = CLIPVisionModelWithProjection.from_pretrained(
        "openai/clip-vit-large-patch14",
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    image_encoder_two = CLIPVisionModelWithProjection.from_pretrained(
        "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        dtype=torch.bfloat16,
        local_files_only=True,
    )

    image_encoder_one.to(device)
    image_encoder_two.to(device)

    image_encoders = [image_encoder_one, image_encoder_two]
    compute_embeddings_fn = partial(
        compute_embeddings,
        image_processor=image_processor,
        image_encoders=image_encoders,
    )

    train_dataset = train_dataset.map(
        compute_embeddings_fn,
        batched=True,
        batch_size=64,
    )
    test_dataset = test_dataset.map(
        compute_embeddings_fn,
        batched=True,
        batch_size=64,
    )

    del image_encoder_one, image_encoder_two
    gc.collect()
    torch.cuda.empty_cache()

    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        subfolder="vae",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    vae.to(device)

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    latent_channels = vae.config.latent_channels

    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
    mask_processor = VaeImageProcessor(
        vae_scale_factor=vae_scale_factor,
        vae_latent_channels=latent_channels,
        do_normalize=False,
        do_convert_grayscale=True,
    )

    compute_encodings_fn = partial(
        compute_encodings,
        image_processor=image_processor,
        mask_processor=mask_processor,
        vae=vae,
    )

    train_dataset = train_dataset.map(
        compute_encodings_fn,
        batched=True,
        batch_size=64,
    )
    test_dataset = test_dataset.map(
        compute_encodings_fn,
        batched=True,
        batch_size=64,
    )

    del vae
    gc.collect()
    torch.cuda.empty_cache()

    dataset = DatasetDict(
        {
            "train": train_dataset,
            "test": test_dataset,
        }
    )
    dataset.save_to_disk("preprocessed_data")


if __name__ == "__main__":
    main()
