import math
import os
from dataclasses import asdict
from itertools import islice

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.model import DecoderTransformer, RoFormer
from core.utils import get_autocast_context, get_device
from data.setup_data import get_data_loader_kwargs, get_data_loaders


def scale_gradients(model, scale):
    if scale == 1.0:
        return

    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(scale)


def optimiser_step(model, optimiser, scheduler, max_grad_norm=None):
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimiser.step()
    optimiser.zero_grad(set_to_none=True)
    scheduler.step()


def train_step(
    model,
    batch,
    optimiser,
    scheduler,
    device,
    ignore_index=-100,
    max_grad_norm=None,
    gradient_accumulation_steps=1,
    should_step=True,
    use_checkpoint=False,
):
    model.train()

    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    with get_autocast_context(device):
        logits = model(input_ids, use_checkpoint=use_checkpoint)
        loss = F.cross_entropy(
            input=logits.view(-1, logits.size(-1)),
            target=labels.view(-1),
            ignore_index=ignore_index,
        )

    loss = loss / gradient_accumulation_steps
    loss.backward()

    if should_step:
        optimiser_step(model, optimiser, scheduler, max_grad_norm=max_grad_norm)

    return loss.item() * gradient_accumulation_steps


def validate_step(model, val_loader, device, ignore_index=-100):
    model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        with tqdm(val_loader, leave=True) as progress_bar:
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                logits = model(input_ids)
                loss = F.cross_entropy(
                    input=logits.view(-1, logits.size(-1)),
                    target=labels.view(-1),
                    ignore_index=ignore_index,
                )

                total_val_loss += loss.item()
                progress_bar.set_postfix(batch_loss=f"{loss:.4f}")

    return total_val_loss / len(val_loader)


def get_checkpoint_path(config, epoch, global_step):
    filename = f"{config.checkpoint_prefix}_epoch_{epoch:03d}_step_{global_step:06d}.pt"
    return os.path.join(config.checkpoint_dir, filename)


def load_checkpoint(config, model, optimiser, scheduler, device):
    if not config.resume_training:
        return 0, 0, 0, scheduler

    if config.checkpoint_path is None:
        raise ValueError("checkpoint_path must be set when resume_training is True")

    checkpoint = torch.load(
        config.checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
    if "scheduler_state_dict" in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint["epoch"]
    global_step = checkpoint["global_step"]
    batch_offset = checkpoint.get("batch_offset", 0)

    tqdm.write(
        f"Resuming from {config.checkpoint_path}: "
        f"completed_epoch={start_epoch}, global_step={global_step}, "
        f"batch_offset={batch_offset}"
    )

    return start_epoch, global_step, batch_offset, scheduler


def save_checkpoint(
    config,
    model,
    optimiser,
    scheduler,
    completed_epoch,
    global_step,
    avg_train_loss,
    avg_val_loss,
    batch_offset=0,
):
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    checkpoint_path = get_checkpoint_path(config, completed_epoch, global_step)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimiser_state_dict": optimiser.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "epoch": completed_epoch,
            "global_step": global_step,
            "batch_offset": batch_offset,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "config": asdict(config),
        },
        checkpoint_path,
    )

    return checkpoint_path


def train(config, device=None):
    device = device or get_device()
    torch.set_float32_matmul_precision("high")
    train_loader, val_loader = get_data_loaders(config=config)

    if config.model_type == "roformer":
        model = RoFormer(config=config)
    else:
        model = DecoderTransformer(config=config)
    model.to(device)

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    optimiser.zero_grad(set_to_none=True)

    steps_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation_steps)
    total_steps = max(1, config.epochs * steps_per_epoch)
    warmup_steps = min(config.warmup_steps, max(total_steps - 1, 0))
    cosine_steps = max(1, total_steps - warmup_steps)

    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser,
        T_max=cosine_steps,
        eta_min=config.min_lr,
    )
    if warmup_steps > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimiser,
            start_factor=1e-6,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimiser, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )
    else:
        scheduler = cosine

    start_epoch, global_step, batch_offset, scheduler = load_checkpoint(
        config, model, optimiser, scheduler, device
    )

    if config.resume_training and batch_offset == 0 and len(train_loader) > 0:
        batch_offset = global_step % len(train_loader)
        if batch_offset > 0:
            tqdm.write(f"Inferred batch_offset={batch_offset} from global_step")

    if start_epoch >= config.epochs:
        tqdm.write(
            f"Checkpoint already completed {start_epoch} epoch(s); "
            f"target is {config.epochs}. Nothing to train."
        )
        return

    current_epoch = start_epoch
    total_loss = 0.0
    batches_trained_this_epoch = 0
    batches_trained_this_run = 0
    accumulation_step = 0

    try:
        for epoch in range(start_epoch, config.epochs):
            current_epoch = epoch
            total_loss = 0.0
            batches_trained_this_epoch = batch_offset if epoch == start_epoch else 0
            batches_trained_this_run = 0
            accumulation_step = 0

            generator = torch.Generator()
            generator.manual_seed(config.seed + epoch)
            epoch_train_loader = DataLoader(
                train_loader.dataset,
                batch_size=config.batch_size,
                shuffle=True,
                generator=generator,
                **get_data_loader_kwargs(config),
            )
            batches = islice(epoch_train_loader, batches_trained_this_epoch, None)

            with tqdm(
                desc=f"Epoch {epoch + 1}",
                initial=batches_trained_this_epoch,
                total=len(epoch_train_loader),
                leave=True,
            ) as progress_bar:
                for batch in batches:
                    accumulation_step += 1
                    should_step = accumulation_step == config.gradient_accumulation_steps
                    batch_loss = train_step(
                        model,
                        batch,
                        optimiser,
                        scheduler,
                        device,
                        ignore_index=config.ignore_index,
                        max_grad_norm=config.max_grad_norm,
                        gradient_accumulation_steps=config.gradient_accumulation_steps,
                        should_step=should_step,
                        use_checkpoint=config.activation_checkpointing,
                    )
                    total_loss += batch_loss
                    batches_trained_this_epoch += 1
                    batches_trained_this_run += 1
                    global_step += 1

                    if should_step:
                        accumulation_step = 0

                    progress_bar.set_postfix(
                        batch_loss=f"{batch_loss:.4f}",
                        global_step=global_step,
                    )
                    progress_bar.update(1)

            if accumulation_step > 0:
                scale_gradients(
                    model,
                    config.gradient_accumulation_steps / accumulation_step,
                )
                optimiser_step(
                    model,
                    optimiser,
                    scheduler,
                    max_grad_norm=config.max_grad_norm,
                )
                accumulation_step = 0

            avg_train_loss = total_loss / max(batches_trained_this_run, 1)
            batch_offset = 0
            train_perplexity = math.exp(avg_train_loss)
            avg_val_loss = validate_step(
                model,
                val_loader,
                device,
                ignore_index=config.ignore_index,
            )
            val_perplexity = math.exp(avg_val_loss)

            if device.type == "mps":
                torch.mps.empty_cache()

            tqdm.write(
                f"Epoch {epoch + 1}: "
                f"train_loss={avg_train_loss:.4f}, train_ppl={train_perplexity:.2f}, "
                f"val_loss={avg_val_loss:.4f}, val_ppl={val_perplexity:.2f}"
            )

            checkpoint_path = save_checkpoint(
                config,
                model,
                optimiser,
                scheduler,
                completed_epoch=epoch + 1,
                global_step=global_step,
                avg_train_loss=avg_train_loss,
                avg_val_loss=avg_val_loss,
                batch_offset=0,
            )
            tqdm.write(f"Saved checkpoint to {checkpoint_path}")
    except KeyboardInterrupt:
        if accumulation_step > 0:
            scale_gradients(
                model,
                config.gradient_accumulation_steps / accumulation_step,
            )
            optimiser_step(
                model,
                optimiser,
                scheduler,
                max_grad_norm=config.max_grad_norm,
            )
            accumulation_step = 0

        avg_train_loss = (
            total_loss / batches_trained_this_run
            if batches_trained_this_run > 0
            else None
        )
        completed_epoch = (
            current_epoch + 1
            if batches_trained_this_epoch == len(train_loader)
            else current_epoch
        )

        checkpoint_path = save_checkpoint(
            config,
            model,
            optimiser,
            scheduler,
            completed_epoch=completed_epoch,
            global_step=global_step,
            avg_train_loss=avg_train_loss,
            avg_val_loss=None,
            batch_offset=batches_trained_this_epoch,
        )
        tqdm.write(
            "Keyboard interrupt received. "
            f"Weight checkpoint safely saved to {checkpoint_path}"
        )
        raise
