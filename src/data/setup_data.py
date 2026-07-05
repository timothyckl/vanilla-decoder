import os

from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast


def _dataset_paths(block_size):
    return (
        f"./data/tinystories_tokenized_train_block_{block_size}",
        f"./data/tinystories_tokenized_val_block_{block_size}",
    )


def group_texts(examples, block_size):
    block_length = block_size + 1
    concatenated_ids = sum(examples["input_ids"], [])
    total_length = (len(concatenated_ids) // block_length) * block_length

    input_blocks = []
    label_blocks = []

    for i in range(0, total_length, block_length):
        block = concatenated_ids[i : i + block_length]
        input_blocks.append(block[:-1])
        label_blocks.append(block[1:])

    return {"input_ids": input_blocks, "labels": label_blocks}


def get_data_loader_kwargs(config):
    kwargs = {
        "num_workers": config.num_workers,
        "pin_memory": False,
    }

    if config.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = config.prefetch_factor

    return kwargs


def get_data_loaders(config):
    train_path, val_path = _dataset_paths(config.block_size)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_batch(batch):
        tokenized = tokenizer(
            batch["text"],
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )
        tokenized["input_ids"] = [
            input_ids + [tokenizer.eos_token_id]
            for input_ids in tokenized["input_ids"]
        ]
        return tokenized

    # Check if tokenised datasets exist for this context length, otherwise create them.
    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        print("Tokenized datasets not found. Loading and tokenizing raw data...")

        os.makedirs("./data", exist_ok=True)
        ds = load_dataset("roneneldan/TinyStories")

        tokenized = ds.map(
            tokenize_batch,
            batched=True,
            remove_columns=ds["train"].column_names,
        )

        lm_dataset = tokenized.map(
            lambda examples: group_texts(examples, block_size=config.block_size),
            batched=True,
        )

        lm_dataset.set_format(type="torch", columns=["input_ids", "labels"])
        lm_dataset["train"].save_to_disk(train_path)
        lm_dataset["validation"].save_to_disk(val_path)
        print("Datasets tokenized and saved to disk.")

    train_dataset = load_from_disk(train_path)
    val_dataset = load_from_disk(val_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        **get_data_loader_kwargs(config),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        **get_data_loader_kwargs(config),
    )

    return train_loader, val_loader
