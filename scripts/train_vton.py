import argparse
import copy
import math
import os
from typing import Any, Dict, List

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_EVALUATE_OFFLINE"] = "1"

os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_PROJECT"] = "looky"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

import torch
from accelerate import Accelerator
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from datasets import load_from_disk
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from tqdm import tqdm
from transformers import get_scheduler

from looky.models.transformer_garment import GarmentTransformer2DModel
from looky.models.transformer_vton import VtonTransformer2DModel


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
    )

    parser.add_argument(
        "--allow_tf32",
        action="store_true",
    )

    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-8,
    )

    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
    )

    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"],
    )

    parser.add_argument(
        "--logit_mean",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--logit_std",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
    )

    return parser.parse_args()


class EarlyStoppingCallback:
    def __init__(self, threshold=0, patience=5):
        self.threshold = threshold
        self.patience = patience
        self.counter = 0
        self.best_score = float("inf")

    def __call__(self, score: float) -> bool:
        if self.best_score - score > self.threshold:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    garment_image_latents = torch.stack(
        [feature["garment_image_latents"] for feature in features]
    )
    garment_image_embeds = torch.stack(
        [feature["garment_image_embeds"] for feature in features]
    )
    image_latents = torch.stack([feature["image_latents"] for feature in features])
    masked_image_latents = torch.stack(
        [feature["masked_image_latents"] for feature in features]
    )
    mask = torch.stack([feature["mask"] for feature in features])
    pose_cond = torch.stack([feature["pose_cond"] for feature in features])
    person_image = torch.stack([feature["person_image"] for feature in features])

    return {
        "garment_image_latents": garment_image_latents,
        "garment_image_embeds": garment_image_embeds,
        "image_latents": image_latents,
        "masked_image_latents": masked_image_latents,
        "mask": mask,
        "pose_cond": pose_cond,
        "person_image": person_image,
    }


def main():
    args = parse_args()

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=args.output_dir
    )
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    set_seed(args.seed)

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        subfolder="scheduler",
        local_files_only=True,
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        subfolder="vae",
        local_files_only=True,
    )
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    transformer_garment = GarmentTransformer2DModel.from_pretrained(
        "./weights/virtual_try_on",
        subfolder="transformer_garment",
        local_files_only=True,
    )

    transformer_vton = VtonTransformer2DModel.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        subfolder="transformer",
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )

    proj = transformer_vton.pos_embed.proj
    weight = proj.weight.new_empty(1536, 17, 2, 2)
    torch.nn.init.xavier_uniform_(weight)
    weight = torch.cat([proj.weight, weight], dim=1)
    proj.weight = torch.nn.Parameter(weight)

    transformer_garment.requires_grad_(False)
    transformer_vton.requires_grad_(True)
    vae.requires_grad_(False)

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg")
    fid = FrechetInceptionDistance(normalize=True)
    kid = KernelInceptionDistance(normalize=True)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=torch.float32)
    ssim.to(accelerator.device)
    lpips.to(accelerator.device)
    fid.to(accelerator.device)
    kid.to(accelerator.device)

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    dataset = load_from_disk("preprocessed_data")
    dataset.set_format(
        type="torch",
        columns=[
            "garment_image_latents",
            "garment_image_embeds",
            "image_latents",
            "masked_image_latents",
            "mask",
            "pose_cond",
            "person_image",
        ],
    )

    train_dataset = dataset["train"].shuffle(seed=42)
    eval_dataset = dataset["test"].shuffle(seed=42)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.per_device_train_batch_size,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        shuffle=True,
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(
        transformer_vton.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
        eps=args.adam_epsilon,
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=(
            args.max_train_steps
            if overrode_max_train_steps
            else args.max_train_steps * accelerator.num_processes
        ),
    )

    (
        transformer_garment,
        transformer_vton,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        transformer_garment,
        transformer_vton,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    )

    callback = EarlyStoppingCallback()

    if accelerator.is_main_process:
        project_name = "my-project"
        accelerator.init_trackers(project_name, config=vars(args))

    global_step = 0
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)

        return sigma

    for epoch in range(args.num_train_epochs):
        transformer_vton.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer_vton):
                garment_image_embeds = batch["garment_image_embeds"]
                garment_image_embeds = garment_image_embeds.to(
                    device=accelerator.device, dtype=weight_dtype
                )

                model_input = batch["garment_image_latents"]
                model_input = model_input.to(
                    device=accelerator.device, dtype=weight_dtype
                )

                timesteps = torch.zeros(
                    model_input.shape[0],
                    dtype=torch.long,
                    device=accelerator.device,
                )

                with torch.no_grad():
                    _, cache = transformer_garment(
                        hidden_states=model_input,
                        timestep=timesteps,
                        encoder_hidden_states=None,
                        pooled_projections=garment_image_embeds,
                        return_dict=False,
                    )

                image_latents = batch["image_latents"]
                image_latents = image_latents.to(
                    device=accelerator.device, dtype=weight_dtype
                )

                noise = torch.randn_like(image_latents)
                batch_size = image_latents.shape[0]

                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=batch_size,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(
                    device=image_latents.device
                )

                sigmas = get_sigmas(
                    timesteps, n_dim=image_latents.ndim, dtype=image_latents.dtype
                )
                noisy_image_latents = (1.0 - sigmas) * image_latents + sigmas * noise

                masked_image_latents = batch["masked_image_latents"]
                masked_image_latents = masked_image_latents.to(
                    device=accelerator.device, dtype=weight_dtype
                )

                mask = batch["mask"]
                mask = mask.to(device=accelerator.device, dtype=weight_dtype)

                model_input = torch.cat(
                    [noisy_image_latents, masked_image_latents, mask], dim=1
                )

                pose_cond = batch["pose_cond"]
                pose_cond = pose_cond.to(device=accelerator.device, dtype=weight_dtype)

                model_pred = transformer_vton(
                    hidden_states=model_input,
                    timestep=timesteps,
                    encoder_hidden_states=None,
                    pooled_projections=garment_image_embeds,
                    return_dict=False,
                    cache=cache,
                    pose_cond=pose_cond,
                )[0]

                target = image_latents

                model_pred = model_pred * (-sigmas) + noisy_image_latents

                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.weighting_scheme, sigmas=sigmas
                )

                loss = torch.mean(
                    (
                        weighting.float() * (model_pred.float() - target.float()) ** 2
                    ).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        transformer_vton.parameters(), args.max_grad_norm
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if (
                    args.checkpointing_steps is not None
                    and args.checkpointing_steps > 0
                    and global_step % args.checkpointing_steps == 0
                    and accelerator.is_main_process
                ):
                    save_path = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}"
                    )
                    accelerator.save_state(save_path)

            accelerator.log(
                {
                    "epoch": epoch,
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "step": global_step,
                },
                step=global_step,
            )

            if global_step >= args.max_train_steps:
                break

        transformer_vton.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            garment_image_embeds = batch["garment_image_embeds"]
            garment_image_embeds = garment_image_embeds.to(
                device=accelerator.device, dtype=weight_dtype
            )

            model_input = batch["garment_image_latents"]
            model_input = model_input.to(device=accelerator.device, dtype=weight_dtype)

            timesteps = torch.zeros(
                model_input.shape[0],
                dtype=torch.long,
                device=accelerator.device,
            )

            with torch.no_grad():
                cache = transformer_garment(
                    hidden_states=model_input,
                    timestep=timesteps,
                    encoder_hidden_states=None,
                    pooled_projections=garment_image_embeds,
                    return_dict=False,
                )[1]

            image_latents = batch["image_latents"]
            image_latents = image_latents.to(
                device=accelerator.device, dtype=weight_dtype
            )

            noise = torch.randn_like(image_latents)
            batch_size = image_latents.shape[0]

            u = compute_density_for_timestep_sampling(
                weighting_scheme=args.weighting_scheme,
                batch_size=batch_size,
                logit_mean=args.logit_mean,
                logit_std=args.logit_std,
                mode_scale=args.mode_scale,
            )
            indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
            timesteps = noise_scheduler_copy.timesteps[indices].to(
                device=image_latents.device
            )

            sigmas = get_sigmas(
                timesteps, n_dim=image_latents.ndim, dtype=image_latents.dtype
            )
            noisy_image_latents = (1.0 - sigmas) * image_latents + sigmas * noise

            masked_image_latents = batch["masked_image_latents"]
            masked_image_latents = masked_image_latents.to(
                device=accelerator.device, dtype=weight_dtype
            )

            mask = batch["mask"]
            mask = mask.to(device=accelerator.device, dtype=weight_dtype)

            model_input = torch.cat(
                [noisy_image_latents, masked_image_latents, mask], dim=1
            )

            pose_cond = batch["pose_cond"]
            pose_cond = pose_cond.to(device=accelerator.device, dtype=weight_dtype)

            with torch.no_grad():
                model_pred = transformer_vton(
                    hidden_states=model_input,
                    timestep=timesteps,
                    encoder_hidden_states=None,
                    pooled_projections=garment_image_embeds,
                    return_dict=False,
                    cache=cache,
                    pose_cond=pose_cond,
                )[0]

            target = image_latents

            model_pred = model_pred * (-sigmas) + noisy_image_latents

            weighting = compute_loss_weighting_for_sd3(
                weighting_scheme=args.weighting_scheme, sigmas=sigmas
            )

            loss = torch.mean(
                (
                    weighting.float() * (model_pred.float() - target.float()) ** 2
                ).reshape(target.shape[0], -1),
                1,
            )
            loss = loss.mean()
            losses.append(accelerator.gather_for_metrics(loss))

            preds = model_pred / vae.config.scaling_factor + vae.config.shift_factor
            preds = vae.decode(preds, return_dict=False)[0]
            preds = image_processor.postprocess(preds, output_type="pt")
            preds = accelerator.gather_for_metrics(preds)

            target = batch["person_image"]
            target = target.to(device=accelerator.device, dtype=weight_dtype)
            target = accelerator.gather_for_metrics(target)

            if accelerator.is_main_process:
                ssim.update(preds, target)
                lpips.update(preds, target)

                fid.update(target, real=True)
                fid.update(preds, real=False)

                kid.update(target, real=True)
                kid.update(preds, real=False)

        losses = torch.cat(losses)
        eval_loss = torch.mean(losses)

        if accelerator.is_main_process:
            eval_ssim = ssim.compute().item()
            eval_lpips = lpips.compute().item()
            eval_fid = fid.compute().item()
            eval_kid = kid.compute().item()

            accelerator.log(
                {
                    "epoch": epoch,
                    "eval_loss": eval_loss,
                    "eval_ssim": eval_ssim,
                    "eval_lpips": eval_lpips,
                    "eval_fid": eval_fid,
                    "eval_kid": eval_kid,
                    "step": global_step,
                },
                step=global_step,
            )

            ssim.reset()
            lpips.reset()
            fid.reset()
            kid.reset()

        if callback(eval_loss.item()):
            accelerator.set_trigger()

        if accelerator.check_trigger():
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer_vton = accelerator.unwrap_model(transformer_vton)
        transformer_vton.save_pretrained(args.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main()
