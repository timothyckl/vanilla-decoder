import os

from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast

BLOCK_SIZE: int = 64


def group_texts(examples):
    # Concatenate all fields (e.g., input_ids and attention_mask)
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated["input_ids"])

    # Drop the remainder to make clean blocks
    total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE

    # Slice each field consistently
    result = {
        k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
        for k, t in concatenated.items()
    }

    input_blocks = []
    label_blocks = []

    for block in result["input_ids"]:
        input_blocks.append(block[:-1])  # all except last token
        label_blocks.append(block[1:])  # all except first token

    result["input_ids"] = input_blocks
    result["labels"] = label_blocks

    return result


def get_data_loaders(config):
    train_path = "./data/tinystories_tokenized_train"
    val_path = "./data/tinystories_tokenized_val"

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_batch(batch):
        # truncates each example to at most block_size tokens
        return tokenizer(batch["text"], truncation=True, max_length=BLOCK_SIZE)

    # Check if tokenised datasets exist, otherwise create them
    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        print("Tokenized datasets not found. Loading and tokenizing raw data...")

        # Load raw dataset
        ds = load_dataset("roneneldan/TinyStories")

        tokenized = ds.map(
            tokenize_batch,
            batched=True,
            remove_columns=ds[
                "train"
            ].column_names,  # remove original columns if you want
        )

        lm_dataset = tokenized.map(
            group_texts,
            batched=True,
        )

        lm_dataset.set_format(type="torch", columns=["input_ids", "labels"])
        lm_dataset["train"].save_to_disk("./data/tinystories_tokenized_train")
        lm_dataset["validation"].save_to_disk("./data/tinystories_tokenized_val")
        print("Datasets tokenized and saved to disk.")

    # Load from disk
    train_dataset = load_from_disk(train_path)
    val_dataset = load_from_disk(val_path)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, val_loader
