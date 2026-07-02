import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim=2, seq_len=6):
        super().__init__()
        pe = torch.zeros(seq_len, embed_dim)  # [seq_len, embed_dim]
        pos = torch.arange(start=0, end=seq_len, step=1).float().unsqueeze(1)
        embed_idx = torch.arange(start=0, end=embed_dim, step=2).float()
        div_term = torch.exp(embed_idx * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term[: pe[:, 1::2].size(1)])

        self.register_buffer("pe", pe)

    def forward(self, token_embeddings, start_pos=0):
        seq_len = token_embeddings.size(1)
        pe = self.pe.unsqueeze(dim=0)  # [1, seq_len, embed_dim]
        return token_embeddings + pe[:, start_pos : start_pos + seq_len, :]


class RotaryEmbedding(nn.Module):
    def __init__(self, d_h, seq_len=128, base=10000):
        super().__init__()

        self.d_h = d_h  # full head dimensions
        self.num_pairs = d_h // 2
        self.pos = torch.arange(start=0, end=seq_len, step=1).float().unsqueeze(1)
        
        # create frequences for d_h // 2 pairs
        two_i = torch.arange(0, self.num_pairs).float()
        freqs = 1.0 / (base ** (two_i / self.num_pairs))

        # combine positions and frequencies into angles [seq_len, d_h // 2]
        angles = self.pos.matmul(freqs.unsqueeze(0))
        cos, sin = angles.cos(), angles.sin()

        # store cos and sin as buffers
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

        # NOTE: to broadcast cleanly over batches and heads,
        # we make cos and sin [1, 1, seq_len, d_h // 2],
        # which broadcasts to [batch_size, num_heads, seq_len, d_h // 2]

    def forward(self, x, start_pos=0):
        # derive seq_len from Q or K.
        # NOTE: -2 is the sequence length
        seq_len = x.size(-2)

        cos = self.cos[start_pos: start_pos + seq_len]  # [seq_len, d_h // 2]
        sin = self.sin[start_pos: start_pos + seq_len]  # [seq_len, d_h // 2]

        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, d_h // 2]
        sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, d_h // 2]

        # apply rotations to Q or K
        x_even = x[...,0::2]  # [batch_size, num_heads, seq_len, d_h // 2]
        x_odd = x[...,1::2]   # [batch_size, num_heads, seq_len, d_h // 2]

        x_rot_even = x_even * cos - x_odd * sin 
        x_rot_odd = x_even * sin + x_odd * cos 

        out = torch.empty_like(x)
        out[...,0::2] = x_rot_even # [batch_size, num_heads, seq_len, d_h // 2]
        out[...,1::2] = x_rot_odd # [batch_size, num_heads, seq_len, d_h // 2]

        return out


class AttentionHead(nn.Module):
    def __init__(self, embed_dim=2) -> None:
        super().__init__()
        self.w_q = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)
        self.w_k = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)
        self.w_v = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)

        self.row_dim = 1
        self.col_dim = 2

        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)

    def reset_cache(self):
        self.cache_k = None
        self.cache_v = None

    def forward(self, encoded_embeddings, mask=None, use_cache=False):
        q = self.w_q(encoded_embeddings)
        k = self.w_k(encoded_embeddings)
        v = self.w_v(encoded_embeddings)

        if use_cache:
            if self.cache_k is not None and self.cache_v is not None:
                # append new keys/values to the cached tensors
                k = torch.cat([self.cache_k, k], dim=self.row_dim)
                v = torch.cat([self.cache_v, v], dim=self.row_dim)

            self.cache_k = k
            self.cache_v = v

        k_T = k.transpose(dim0=self.row_dim, dim1=self.col_dim)

        sim = torch.matmul(q, k_T)
        scaled_sim = sim / (k.size(self.col_dim) ** 0.5)

        if mask is not None:
            scaled_sim = scaled_sim.masked_fill(mask=mask, value=-torch.inf)

        att_pct = F.softmax(scaled_sim, dim=self.col_dim)
        att_scr = torch.matmul(att_pct, v)

        return att_scr


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim=2,
        num_heads=1,
        dropout=0.0,
        seq_len=128,
        use_rope=False,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_h = embed_dim // num_heads  # dimensions per head
        self.rope = RotaryEmbedding(d_h=self.d_h, seq_len=seq_len) if use_rope else None

        self.w_q = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)
        self.w_k = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)
        self.w_v = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.w_o = nn.Linear(in_features=embed_dim, out_features=embed_dim)

        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)

    def reset_cache(self):
        self.cache_k = None
        self.cache_v = None

    def forward(self, encoded_embeddings, mask=None, use_cache=False):
        batch_size, seq_len, _ = encoded_embeddings.size()

        # [batch_size, seq_len, embed_dim]
        q = self.w_q(encoded_embeddings)
        k = self.w_k(encoded_embeddings)
        v = self.w_v(encoded_embeddings)

        # split q, k, v into heads, shape: [batch_size, num_heads, seq_len, d_h]
        q = q.view(batch_size, seq_len, self.num_heads, self.d_h).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_h).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_h).transpose(1, 2)

        if self.rope is not None:
            cached_seq_len = self.cache_k.size(2) if use_cache and self.cache_k is not None else 0
            q = self.rope(q, start_pos=cached_seq_len)
            k = self.rope(k, start_pos=cached_seq_len)

        if use_cache:
            if self.cache_k is not None and self.cache_v is not None:
                k = torch.cat([self.cache_k, k], dim=2)  # dim = 2 is seq_len here.
                v = torch.cat([self.cache_v, v], dim=2)
            self.cache_k = k
            self.cache_v = v

        q_len = q.size(-2)  
        k_len = k.size(-2) 

        # mask during training/prefill, not during decode
        is_causal = mask is not None and (q_len == k_len)

        att_scr = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=is_causal,
        )

        # concatenate outputs per head
        out = (
            att_scr.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )

        # linear projection
        return self.resid_dropout(self.w_o(out))


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, seq_len=128, use_rope=False):
        super().__init__()
        self.attn_heads = MultiHeadAttention(
            embed_dim, num_heads, dropout, seq_len=seq_len, use_rope=use_rope
        )
        self.pw_ffn = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=4 * embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=4 * embed_dim, out_features=embed_dim),
            nn.Dropout(dropout),
        )
        self.norm_a = nn.LayerNorm(normalized_shape=embed_dim)
        self.norm_b = nn.LayerNorm(normalized_shape=embed_dim)

    def forward(self, encoded_embeddings, mask=None, use_cache=False):
        # prenorm embeddings
        attn_out = encoded_embeddings + self.attn_heads(
            self.norm_a(encoded_embeddings), mask, use_cache=use_cache
        )
        ffn_out = attn_out + self.pw_ffn(self.norm_b(attn_out))

        return ffn_out


class DecoderTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.we = nn.Embedding(
            num_embeddings=config.vocab_size, embedding_dim=config.embed_dim
        )
        self.pe = PositionalEncoding(
            embed_dim=config.embed_dim, seq_len=config.block_size
        )
        self.layers = nn.ModuleList(
            [
                Block(config.embed_dim, config.num_heads, config.dropout)
                for _ in range(config.num_layers)
            ]
        )
        self.lm_head = nn.Linear(
            in_features=config.embed_dim, out_features=config.vocab_size
        )

    def reset_cache(self):
        for layer in self.layers:
            layer.attn_heads.reset_cache()

    def forward(self, input_ids, use_cache=False):
        cached_seq_len = 0

        if use_cache and self.layers[0].attn_heads.cache_k is not None:
            cached_seq_len = self.layers[0].attn_heads.cache_k.size(2)

        word_embedding = self.we(input_ids)
        position_encoding = self.pe(word_embedding, start_pos=cached_seq_len)

        seq_len = input_ids.size(1)
        bool_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        bool_mask = bool_mask.unsqueeze(0).unsqueeze(
            0
        )  # [batch_size, num_heads, seq_len, seq_len]

        x = position_encoding
        for layer in self.layers:
            x = layer(x, bool_mask, use_cache=use_cache)

        logits = self.lm_head(x)

        return logits


class RoFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.we = nn.Embedding(
            num_embeddings=config.vocab_size, embedding_dim=config.embed_dim
        )
        self.layers = nn.ModuleList(
            [
                Block(
                    config.embed_dim,
                    config.num_heads,
                    config.dropout,
                    seq_len=config.block_size,
                    use_rope=True,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.lm_head = nn.Linear(
            in_features=config.embed_dim, out_features=config.vocab_size
        )

    def reset_cache(self):
        for layer in self.layers:
            layer.attn_heads.reset_cache()

    def forward(self, input_ids, use_cache=False):
        word_embedding = self.we(input_ids)

        seq_len = input_ids.size(1)
        bool_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        bool_mask = bool_mask.unsqueeze(0).unsqueeze(
            0
        )  # [batch_size, num_heads, seq_len, seq_len]

        x = word_embedding
        for layer in self.layers:
            x = layer(x, bool_mask, use_cache=use_cache)

        logits = self.lm_head(x)

        return logits
