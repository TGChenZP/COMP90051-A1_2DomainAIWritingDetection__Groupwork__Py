import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_class.layers.SelfAttention_Family import FullAttention, AttentionLayer
from model.model_class.layers.Embed import PositionalWordEmbedding
from model.model_class.layers.modules import *
from model.model_class.layers.Transformer_EncDec import Encoder, EncoderLayer
import numpy as np


class W2V(W2V_Model):

    class Model(nn.Module):

        def __init__(self, configs):
            super().__init__() 

            self.configs = configs

            self.embed = nn.Embedding(num_embeddings=self.configs.n_unique_tokens, embedding_dim=self.configs.d_model)

            self.linear = nn.Linear(in_features = self.configs.d_model, out_features = self.configs.n_unique_tokens)

        def forward(self, X, mask):

            dim_1_mask = []
            for i in range(X.shape[0]):
                dim_1_mask.extend([i for _ in range(self.configs.k)])
            dim_2_mask = mask.view(-1)
            
            dim_1_mask = torch.LongTensor(np.array(dim_1_mask)).to(self.configs.device)
            
            embed = self.embed(X)

            prediction = self.linear(embed)

            prediction = prediction[dim_1_mask, dim_2_mask]

            

            prediction = prediction.reshape(X.shape[0], -1)

            return prediction