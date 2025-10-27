import torch
import torch.nn.functional as F
from tqdm import tqdm

from model import DecoderTransformer
from setup_data import get_data_loaders

def train_step(model, batch, optimiser, device):
    model.train()

    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    logits = model(input_ids)
    loss = F.cross_entropy(
        input=logits.view(-1, logits.size(-1)), target=labels.view(-1)
    )

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    return loss.item()


def validate_step(model, val_loader, device):
    model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        with tqdm(val_loader, leave=True) as progress_bar:
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                logits = model(input_ids)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
                )

                total_val_loss += loss.item()
                progress_bar.set_postfix(batch_loss=f"{loss:.4f}")

    return total_val_loss / len(val_loader)


def train(config):
    device = torch.device("mps")
    train_loader, val_loader = get_data_loaders(config=config)

    model = DecoderTransformer(config=config)
    model.to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.epochs):
        total_loss = 0

        with tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=True) as progress_bar:
            for batch in progress_bar:
                batch_loss = train_step(model, batch, optimiser, device)
                total_loss += batch_loss
                progress_bar.set_postfix(batch_loss=f"{batch_loss:.4f}")

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = validate_step(model, val_loader, device)

        tqdm.write(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

        torch.save(model.state_dict(), "./weights/new_model_weights.pt")
