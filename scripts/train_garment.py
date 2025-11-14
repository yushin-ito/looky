import argparse
import math
import os
from typing import Dict, List

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_EVALUATE_OFFLINE"] = "1"

os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_PROJECT"] = "looky"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    BatchFeature,
    get_scheduler,
)

from looky.models.transformer_garment import GarmentTransformer2DModel


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
        default=8,
    )

    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
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
        default="constant",
    )

    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=500,
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


def collate_fn(features: List[BatchFeature]) -> Dict[str, torch.Tensor]:
    garment_image_latents = [feature["garment_image_latents"] for feature in features]
    garment_image_embeds = [feature["garment_image_embeds"] for feature in features]

    return {
        "garment_image_latents": torch.stack(garment_image_latents),
        "garment_image_embeds": torch.stack(garment_image_embeds),
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

    transformer = GarmentTransformer2DModel.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        subfolder="transformer",
        local_files_only=True,
    )

    transformer.requires_grad_()

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    dataset = load_from_disk("preprocessed_data")
    dataset.set_format(
        type="torch",
        columns=[
            "garment_image_latents",
            "garment_image_embeds",
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
        transformer.parameters(),
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

    transformer, optimizer, train_dataloader, eval_dataloader, lr_scheduler = (
        accelerator.prepare(
            transformer, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )
    )
    callback = EarlyStoppingCallback()

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        project_name = "my-project"
        accelerator.init_trackers(project_name, config=vars(args))

    global_step = 0

    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )

    for epoch in range(args.num_train_epochs):
        transformer.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                image_embeds = batch["garment_image_embeds"]
                image_embeds = image_embeds.to(
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

                model_pred, _ = transformer(
                    hidden_states=model_input,
                    timestep=timesteps,
                    pooled_projections=image_embeds,
                    return_dict=False,
                )

                loss = F.mse_loss(model_pred.float(), model_input.float())

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        transformer.parameters(), args.max_grad_norm
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if (
                        args.checkpointing_steps is not None
                        and args.checkpointing_steps > 0
                        and global_step % args.checkpointing_steps == 0
                    ):
                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)

            accelerator.log(
                {
                    "epoch": epoch,
                    "loss": loss.detach().item(),
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    "step": global_step,
                },
                step=global_step,
            )

            if global_step >= args.max_train_steps:
                break

        transformer.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            image_embeds = batch["garment_image_embeds"]
            image_embeds = image_embeds.to(
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
                model_pred, _ = transformer(
                    hidden_states=model_input,
                    timestep=timesteps,
                    pooled_projections=image_embeds,
                    return_dict=False,
                )

            loss = F.mse_loss(model_pred.float(), model_input.float())
            losses.append(accelerator.gather_for_metrics(loss))

        losses = torch.cat(losses)
        eval_loss = torch.mean(losses)

        accelerator.log(
            {
                "epoch": epoch,
                "eval_loss": eval_loss,
                "step": global_step,
            },
            step=global_step,
        )

        if accelerator.is_main_process:
            if callback(eval_loss.item()):
                accelerator.set_trigger()

        if accelerator.check_trigger():
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = accelerator.unwrap_model(transformer)
        transformer.save_pretrained(args.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main()
