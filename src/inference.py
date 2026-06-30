import torch
from transformers import GPT2TokenizerFast

from config import Config
from core.model import DecoderTransformer
from core.utils import get_device


def sample_next_token(logits, temperature=1.0, top_k=50):
    next_token_logits = logits[:, -1, :]

    if temperature <= 0:
        return torch.argmax(next_token_logits, dim=-1, keepdim=True)

    next_token_logits = next_token_logits / temperature
    if top_k is not None:
        k = min(top_k, next_token_logits.size(-1))
        top_k_values, _ = torch.topk(next_token_logits, k)
        next_token_logits = next_token_logits.masked_fill(
            next_token_logits < top_k_values[:, [-1]],
            -torch.inf,
        )
    probs = torch.softmax(next_token_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def stream_text(
    model,
    tokenizer,
    prompt,
    block_size,
    max_new_tokens=100,
    device="cpu",
    temperature=1.0,
    top_k=50,
):
    model.eval()
    model.reset_cache()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    model_input = input_ids[:, -block_size:]
    logits = model(model_input, use_cache=True)

    avail_new_tokens = block_size - model_input.size(1)
    max_new_tokens = min(max_new_tokens, avail_new_tokens)
    previous_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    for _ in range(max_new_tokens):
        next_token = sample_next_token(logits, temperature=temperature, top_k=top_k)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        if (
            tokenizer.eos_token_id is not None
            and next_token.item() == tokenizer.eos_token_id
        ):
            break

        current_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        chunk = current_text[len(previous_text) :]
        previous_text = current_text

        if chunk:
            yield chunk

        logits = model(next_token, use_cache=True)


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt,
    block_size,
    max_new_tokens=100,
    device="cpu",
    temperature=1.0,
    top_k=50,
):
    return prompt + "".join(
        stream_text(
            model,
            tokenizer,
            prompt,
            block_size,
            max_new_tokens=max_new_tokens,
            device=device,
            temperature=temperature,
            top_k=top_k,
        )
    )


if __name__ == "__main__":
    cfg = Config()
    device = get_device()

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    if cfg.checkpoint_path is None:
        raise ValueError("checkpoint_path must be set before running inference")

    checkpoint = torch.load(cfg.checkpoint_path, map_location=device, weights_only=True)
    model = DecoderTransformer(config=cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    prompt = "Little Timmy."
    print(prompt, end="", flush=True)
    for chunk in stream_text(
        model,
        tokenizer,
        prompt,
        block_size=cfg.block_size,
        max_new_tokens=100,
        device=device,
        temperature=0.7,
        top_k=50,
    ):
        print(chunk, end="", flush=True)
    print()
