import os
from dataclasses import asdict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from core.model import DecoderTransformer
from core.utils import get_device
from data.setup_data import get_data_loaders


def train_step(model, batch, optimiser, device, ignore_index=-100, max_grad_norm=None):
    model.train()

    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    with torch.autocast(device_type="mps", dtype=torch.bfloat16):
        logits = model(input_ids)
        loss = F.cross_entropy(
            input=logits.view(-1, logits.size(-1)),
            target=labels.view(-1),
            ignore_index=ignore_index,
        )

    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimiser.step()

    return loss.item()


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


def load_checkpoint(config, model, optimiser, device):
    if not config.resume_training:
        return 0, 0

    if config.checkpoint_path is None:
        raise ValueError("checkpoint_path must be set when resume_training is True")

    checkpoint = torch.load(
        config.checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    optimiser.load_state_dict(checkpoint["optimiser_state_dict"])

    start_epoch = checkpoint["epoch"]
    global_step = checkpoint["global_step"]

    tqdm.write(
        f"Resuming from {config.checkpoint_path}: "
        f"completed_epoch={start_epoch}, global_step={global_step}"
    )

    return start_epoch, global_step


def save_checkpoint(
    config,
    model,
    optimiser,
    completed_epoch,
    global_step,
    avg_train_loss,
    avg_val_loss,
):
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    checkpoint_path = get_checkpoint_path(config, completed_epoch, global_step)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimiser_state_dict": optimiser.state_dict(),
            "epoch": completed_epoch,
            "global_step": global_step,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "config": asdict(config),
        },
        checkpoint_path,
    )

    return checkpoint_path


def train(config, device=None):
    device = device or get_device()
    train_loader, val_loader = get_data_loaders(config=config)

    model = DecoderTransformer(config=config)
    model.to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    start_epoch, global_step = load_checkpoint(config, model, optimiser, device)

    if start_epoch >= config.epochs:
        tqdm.write(
            f"Checkpoint already completed {start_epoch} epoch(s); "
            f"target is {config.epochs}. Nothing to train."
        )
        return

    current_epoch = start_epoch
    total_loss = 0.0
    batches_trained_this_epoch = 0

    try:
        for epoch in range(start_epoch, config.epochs):
            current_epoch = epoch
            total_loss = 0.0
            batches_trained_this_epoch = 0

            with tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=True) as progress_bar:
                for batch in progress_bar:
                    batch_loss = train_step(
                        model,
                        batch,
                        optimiser,
                        device,
                        ignore_index=config.ignore_index,
                        max_grad_norm=config.max_grad_norm,
                    )
                    total_loss += batch_loss
                    batches_trained_this_epoch += 1
                    global_step += 1
                    progress_bar.set_postfix(
                        batch_loss=f"{batch_loss:.4f}",
                        global_step=global_step,
                    )

            avg_train_loss = total_loss / len(train_loader)
            avg_val_loss = validate_step(
                model,
                val_loader,
                device,
                ignore_index=config.ignore_index,
            )

            tqdm.write(
                f"Epoch {epoch + 1}: "
                f"train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}"
            )

            checkpoint_path = save_checkpoint(
                config,
                model,
                optimiser,
                completed_epoch=epoch + 1,
                global_step=global_step,
                avg_train_loss=avg_train_loss,
                avg_val_loss=avg_val_loss,
            )
            tqdm.write(f"Saved checkpoint to {checkpoint_path}")
    except KeyboardInterrupt:
        avg_train_loss = (
            total_loss / batches_trained_this_epoch
            if batches_trained_this_epoch > 0
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
            completed_epoch=completed_epoch,
            global_step=global_step,
            avg_train_loss=avg_train_loss,
            avg_val_loss=None,
        )
        tqdm.write(f"Keyboard interrupt received. Saved checkpoint to {checkpoint_path}")
        raise
