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

    def forward(self, token_embeddings):
        seq_len = token_embeddings.size(1)
        pe = self.pe.unsqueeze(dim=0)  # [1, seq_len, embed_dim]
        return token_embeddings + pe[:, :seq_len, :]


class AttentionHead(nn.Module):
    def __init__(self, embed_dim=2) -> None:
        super().__init__()
        self.w_q = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)
        self.w_k = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)
        self.w_v = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)

        self.row_dim = 1
        self.col_dim = 2

    def forward(self, encoded_embeddings, mask=None):
        q = self.w_q(encoded_embeddings)
        k = self.w_k(encoded_embeddings)
        v = self.w_v(encoded_embeddings)

        k_T = k.transpose(dim0=self.row_dim, dim1=self.col_dim)

        sim = torch.matmul(q, k_T)
        scaled_sim = sim / torch.tensor(k.size(self.col_dim) ** 0.5)
        if mask is not None:
            scaled_sim = scaled_sim.masked_fill(mask=mask, value=-torch.inf)

        att_pct = F.softmax(scaled_sim, dim=self.col_dim)
        att_scr = torch.matmul(att_pct, v)

        return att_scr


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=2, num_heads=1, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_h = embed_dim // num_heads  # dimensions per head

        self.w_q = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)
        self.w_k = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)
        self.w_v = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.w_o = nn.Linear(in_features=embed_dim, out_features=embed_dim)

    def forward(self, encoded_embeddings, mask=None):
        batch_size, seq_len, _ = encoded_embeddings.size()

        # [batch_size, seq_len, embed_dim]
        q = self.w_q(encoded_embeddings)
        k = self.w_k(encoded_embeddings)
        v = self.w_v(encoded_embeddings)

        # split q, k, v into heads, shape: [batch_size, num_heads, seq_len, d_h]
        q = q.view(batch_size, seq_len, self.num_heads, self.d_h).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_h).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_h).transpose(1, 2)

        # scaled dot product attention per head
        k_T = k.transpose(dim0=-2, dim1=-1)
        sim = torch.matmul(q, k_T)
        scaled_sim = sim / (self.d_h**0.5)  # [batch_size, num_heads, seq_len, seq_len]

        if mask is not None:
            scaled_sim = scaled_sim.masked_fill(mask=mask, value=-torch.inf)

        att_pct = self.attn_dropout(F.softmax(scaled_sim, dim=-1))
        att_scr = torch.matmul(att_pct, v)  # [batch_size, num_heads, seq_len, d_h]

        # concatenate outputs per head
        out = (
            att_scr.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )

        # linear projection
        return self.resid_dropout(self.w_o(out))


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.attn_heads = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.pw_ffn = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=4 * embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=4 * embed_dim, out_features=embed_dim),
            nn.Dropout(dropout),
        )
        self.norm_a = nn.LayerNorm(normalized_shape=embed_dim)
        self.norm_b = nn.LayerNorm(normalized_shape=embed_dim)

    def forward(self, encoded_embeddings, mask=None):
        # prenorm embeddings
        attn_out = encoded_embeddings + self.attn_heads(self.norm_a(encoded_embeddings), mask)
        ffn_out = attn_out + self.pw_ffn(self.norm_b(attn_out))

        return ffn_out


class DecoderTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.we = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embed_dim)
        self.pe = PositionalEncoding(embed_dim=config.embed_dim, seq_len=config.block_size)
        self.layers = nn.ModuleList([
            Block(config.embed_dim, config.num_heads, config.dropout)
            for _ in range(config.num_layers)
        ])
        self.lm_head = nn.Linear(in_features=config.embed_dim, out_features=config.vocab_size)

    def forward(self, input_ids):
        word_embedding = self.we(input_ids)
        position_encoding = self.pe(word_embedding)

        seq_len = input_ids.size(1)
        bool_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        bool_mask = bool_mask.unsqueeze(0).unsqueeze(0)  # [batch_size, num_heads, seq_len, seq_len]

        x = position_encoding
        for layer in self.layers:
            x = layer(x, bool_mask)

        logits = self.lm_head(x)

        return logits
