import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_class.layers.SelfAttention_Family import FullAttention, AttentionLayer
from model.model_class.layers.Embed import PositionalWordEmbedding
from model.model_class.layers.modules import *
from model.model_class.layers.Transformer_EncDec import Encoder, EncoderLayer
import numpy as np
from model.model_class.__template__ import *


class W2V(W2V_Model):

    class Model(nn.Module):



        def __init__(self, configs):
            super().__init__() 

            self.configs = configs

            self.embed = nn.Embedding(num_embeddings=self.configs.n_unique_tokens, embedding_dim=self.configs.d_model)

            self.linear = nn.Linear(in_features = self.configs.d_model, out_features = self.configs.n_unique_tokens)

        def forward(self, X, mask):

            dim_1_mask = []
            for i in range(self.configs.batch_size):
                dim_1_mask.extend([i for _ in range(self.configs.k)])
            dim_2_mask = []
            for i in range(self.configs.batch_size):
                dim_2_mask.extend(mask[i])
            
            embed = self.embed(X)

            prediction = self.linear(embed)

            predictions = predictions[dim_1_mask, dim_2_mask]

            predictions = predictions.reshape(self.configs.batch_size, -1)

            return predictionk