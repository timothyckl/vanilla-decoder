import os

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


def train(config, device=None):
    device = device or get_device()
    train_loader, val_loader = get_data_loaders(config=config)

    model = DecoderTransformer(config=config)
    model.to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.epochs):
        total_loss = 0

        with tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=True) as progress_bar:
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
                progress_bar.set_postfix(batch_loss=f"{batch_loss:.4f}")

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = validate_step(
            model,
            val_loader,
            device,
            ignore_index=config.ignore_index,
        )

        tqdm.write(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

        os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
        torch.save(model.state_dict(), config.checkpoint_path)
