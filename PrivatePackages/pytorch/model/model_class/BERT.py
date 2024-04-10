import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_class.layers.SelfAttention_Family import FullAttention, AttentionLayer
from model.model_class.layers.Embed import PositionalWordEmbedding
from model.model_class.layers.modules import *
from model.model_class.layers.Transformer_EncDec import Encoder, EncoderLayer
import numpy as np
from model.model_class.__template__ import *


class BERT(ClassificationModel):

    class Model(nn.Module):
        """
        Vanilla Transformer
        with O(L^2) complexity
        Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf

        Adjusted to be encoder only, with MLP as decoder; and takes either 
        flatten-projected last-layer embeddings, or cls, or last position as the 
        representation of the embedding of encoder
        """

        def __init__(self, configs):
            super().__init__() 

            self.configs = configs

            assert configs.d_model % configs.n_heads == 0, "d_model must be a multiple of n_heads"

            # Embedding
            self.embedding = PositionalWordEmbedding(configs.d_model, configs.n_unique_tokens, configs.seq_len, train_embedding=configs.train_embedding)
            # Encoder
            self.encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            FullAttention(configs.mask_flag, None, attention_dropout=configs.dropout,
                                        output_attention=False), configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    ) for l in range(configs.e_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model)
            )

            # Decode

            if self.configs.res_learning:
                self.mlp = nn.ModuleList([ResLayer(self.configs, self.configs.d_model, self.configs.d_model, self.configs.activation,
                                                    self.configs.dropout) for _ in range(self.configs.n_mlp_layers)])
            else:
                self.mlp = nn.ModuleList([LinearLayer(self.configs, self.configs.d_model, self.configs.d_model, self.configs.activation,
                                                        self.configs.dropout) for _ in range(self.configs.n_mlp_layers)])

            self.out = nn.Linear(configs.d_model, configs.d_output)
            self.softmax = nn.Softmax(dim=1)


        def forecast(self, x):
            # Embedding

            enc_out = self.embedding(x)

            enc_out, attns = self.encoder(enc_out, attn_mask=None)

            return enc_out


        def forward(self, x, mask=False):
            
            if mask: # pretraining
                x = self.forecast(x)

                x = x[:, mask, :] # cls

                for layer in self.mlp:
                    x = layer(x)

                y = self.out(x)
                
                return y  

            else:

                x = self.forecast(x)

                x = x[:, 0, :] # cls

                for layer in self.mlp:
                    x = layer(x)

                y = self.softmax(self.out(x))
                
                return y  


class BERT_DANN(DANN_Model):

    class Model(nn.Module):
        """
        Vanilla Transformer
        with O(L^2) complexity
        Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf

        Adjusted to be encoder only, with MLP as decoder; and takes either 
        flatten-projected last-layer embeddings, or cls, or last position as the 
        representation of the embedding of encoder
        """

        def __init__(self, configs):
            super().__init__() 

            self.configs = configs

            assert configs.d_model % configs.n_heads == 0, "d_model must be a multiple of n_heads"

            # Embedding
            self.embedding = PositionalWordEmbedding(configs.d_model, configs.n_unique_tokens, configs.seq_len, train_embedding=configs.train_embedding)
            # Encoder
            self.encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            FullAttention(configs.mask_flag, None, attention_dropout=configs.dropout,
                                        output_attention=False), configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    ) for l in range(configs.e_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model)
            )

            # Decode


            self.gradient_reverse = GradientReversalLayer()

            if self.configs.res_learning:
                self.mlp_clf = nn.ModuleList([ResLayer(self.configs, self.configs.d_model, self.configs.d_model, self.configs.activation,
                                                    self.configs.dropout) for _ in range(self.configs.n_mlp_clf_layers)])
            else:
                self.mlp_clf = nn.ModuleList([LinearLayer(self.configs, self.configs.d_model, self.configs.d_model, self.configs.activation,
                                                    self.configs.dropout) for _ in range(self.configs.n_mlp_clf_layers)])
                
            if self.configs.res_learning:
                self.mlp_dom = nn.ModuleList([ResLayer(self.configs, self.configs.d_model, self.configs.d_model, self.configs.activation,
                                                    self.configs.dropout) for _ in range(self.configs.n_mlp_dom_layers)])
            else:
                self.mlp_dom = nn.ModuleList([LinearLayer(self.configs, self.configs.d_model, self.configs.d_model, self.configs.activation,
                                                    self.configs.dropout) for _ in range(self.configs.n_mlp_dom_layers)])

            self.out_dom = nn.Linear(self.configs.d_model, self.configs.d_output)
            self.out_clf = nn.Linear(self.configs.d_model, self.configs.d_output)
            self.softmax = nn.Softmax(dim=1)


        def forecast(self, x):
            # Embedding

            enc_out = self.embedding(x)

            enc_out, attns = self.encoder(enc_out, attn_mask=None)

            return enc_out

    

        def forward(self, x, mask=False):
            

            x = self.forecast(x)

            x = x[:, 0, :] # cls

            for layer in self.mlp_clf:
                x = layer(x)

            rev_x = self.gradient_reverse(x)

            for layer in self.mlp_dom:
                rev_x = layer(rev_x)

            y = self.softmax(self.out_clf(x))

            dom = self.softmax(self.out_dom(rev_x))

            return y, dom