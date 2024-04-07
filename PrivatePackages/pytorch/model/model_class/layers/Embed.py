import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalWordEmbedding(nn.Module):
    def __init__(self, d_model, n_unique_tokens, max_len=5000, train_embedding = True):
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
        self.embedding.weight.requires_grad = train_embedding

    def forward(self, x):

        return (self.pe[:, :x.size(1)] + self.embedding(x))


class WordEmbedding(nn.Module):
    def __init__(self, d_model, n_unique_tokens, max_len=5000, train_embedding = True):
        super(WordEmbedding, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=n_unique_tokens, embedding_dim=d_model)
        self.embedding.weight.requires_grad = train_embedding

    def forward(self, x):

        return self.embedding(x)