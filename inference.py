import torch
from transformers import GPT2TokenizerFast

from config import Config
from model import DecoderTransformer


@torch.no_grad()
def generate_text(
    model, tokenizer, prompt, block_size, max_new_tokens=50, device="cpu"
):
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    # print(f"prompt tensor shape: {input_ids.shape}")

    for _ in range(max_new_tokens):
        if input_ids.size(1) > block_size:
            input_ids = input_ids[:, -block_size:]

        logits = model(input_ids)  # [1, seq_len, vocab_size]
        next_token_logits = logits[:, -1, :]

        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # append predicted token
        input_ids = torch.cat([input_ids, next_token], dim=1)

    # decode to text
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text


if __name__ == "__main__":
    # ensure that config is same as during training!
    cfg = Config(
        block_size=63, batch_size=16, num_heads=4, num_layers=4, learning_rate=1e-3
    )
    device = torch.device("mps")

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # load model weights and move to mps device
    state_dict = torch.load("./weights/model_weights.pt")
    model = DecoderTransformer(config=cfg)
    model.load_state_dict(state_dict)
    model.to(device)

    prompt = "A long time ago,"
    generated = generate_text(
        model,
        tokenizer,
        prompt,
        block_size=cfg.block_size,
        max_new_tokens=8,
        device=device,
    )
    print(generated)
