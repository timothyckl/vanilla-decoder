# Vanilla Decoder Transformer

This project is an educational implementation of the Transformer inspired by the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

Unlike the full encoder–decoder architecture described in the paper, this project builds a decoder-only transformer as it is standard architecture for causal language modeling and text generation tasks, as used in models like GPT.

The motivation for this project was to:

1. Re-familiarise myself with PyTorch after not using it for some time
2. Gain a deeper understanding of how attention is computed and how the model internals interact during training and inference
3. Build an intuition for autoregressive text generation and the structure of small-scale transformer models

The model was trained from scratch on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset which is a lightweight dataset designed for training small language models that generate simple narratives.

## Features

- [x] Rotary positional embeddings
- [ ] RMSNorm
- [ ] SwiGLU Activation 
- [ ] Weight tying
- [ ] Full bias removal
- [ ] QK normalisation
- [ ] Grouped Query Attention / Multi-Query Attention
- [ ] AdamW / Muon optimisers

### Resources Used

- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) - tokeniser setup, training loop, model scaling
- [Decoder-Only Transformers: The Workhorse of Generative LLMs](https://cameronrwolfe.substack.com/p/decoder-only-transformers-the-workhorse) - conceptual overview of why decoder-only architectures dominate modern LLMs
- [Learn PyTorch for Deep Learning: Zero to Mastery book](https://www.learnpytorch.io/) - refresh my understanding of modern PyTorch syntax and best practices

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/timothyckl/vanilla-decoder.git
```

### 2. Set up virtual environment

```bash
python3 -m venv .venv
```

### 3. Install dependencies 

```bash
pip3 install -r requirements.txt
```

## Training

```bash
python3 src/train.py
```

Checkpoints are saved at the end of each epoch using the completed epoch number:

```text
weights/checkpoint_epoch_001.pt
```

To resume training, set `resume_training = True` and `checkpoint_path` in `src/config.py`. The `epochs` value is the total target epoch count, so resuming from `checkpoint_epoch_002.pt` with `epochs = 5` continues from epoch 3 through epoch 5.

## Inference

Given the prompt `A long time ago,`, the model is able to produce the following output even though the dataset used is somewhat simple and training was only on a single epoch.

Set `checkpoint_path` in `src/config.py` to the checkpoint you want to use, then run:

```bash
python3 src/inference.py
```

```bash
"A long time ago, there was a little girl named Lily."
```
