import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalWordEmbedding(nn.Module):
    def __init__(self, d_model, n_unique_tokens, max_len=5000):
        super(PositionalWordEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

        self.embedding = nn.Embedding(num_embeddings=n_unique_tokens, embedding_dim=d_model)

    def forward(self, x):
        # Convert x to int tensor if it's not already
        if x.dtype != torch.int64:
            x = x.long()  # Convert to int tensor
        return (self.pe[:, :x.size(1)] + self.embedding(x)).float()
