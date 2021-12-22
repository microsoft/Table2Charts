import math

import torch
import torch.nn as nn

from .config import ModelConfig


class InputEmbedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        embed_hidden = config.hidden - config.data_len
        self.embedding_compress = nn.Linear(config.embed_len, embed_hidden) if config.embed_len > 0 else None
        self.cat_embeds = nn.ModuleList([nn.Embedding(cat_num, embed_hidden) for cat_num in config.num_categories])
        self.token_type_embed = nn.Embedding(config.num_token_type, embed_hidden, padding_idx=0)
        self.segment_embed = nn.Embedding(config.num_segment_type, embed_hidden, padding_idx=0)
        self.position_embed = (PositionalEmbedding(d_model=embed_hidden) if config.positional else None)
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, token_types, segments, semantic_embeds, categories):
        x = self.token_type_embed(token_types) + self.segment_embed(segments)
        if self.embedding_compress is not None:
            x += self.embedding_compress(semantic_embeds)
        if self.position_embed is not None:
            x += self.position_embed(segments)
        if len(self.cat_embeds) > 0:
            for cat, embed in zip(categories.chunk(categories.size(-1), -1), self.cat_embeds):
                x += embed(cat.squeeze(dim=-1))
        return self.dropout(x)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        odd_len = d_model - div_term.size(-1)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:odd_len])

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
