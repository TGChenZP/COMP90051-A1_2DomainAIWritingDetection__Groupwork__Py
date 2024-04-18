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

        def __init__(self, configs):
            super().__init__() 

            self.configs = configs

            assert configs.d_model % configs.n_heads == 0, "d_model must be a multiple of n_heads"

            configs.split_domain = False

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
                self.mlp = nn.ModuleList([ResLayer(self.configs, self.configs.d_model+self.configs.d_extra_decoder_features, \
                                                   self.configs.d_model+self.configs.d_extra_decoder_features, self.configs.activation,
                                                    self.configs.dropout) for _ in range(self.configs.n_mlp_layers)])
            else:
                self.mlp = nn.ModuleList([LinearLayer(self.configs, self.configs.d_model+self.configs.d_extra_decoder_features, \
                                                      self.configs.d_model+self.configs.d_extra_decoder_features, self.configs.activation,
                                                        self.configs.dropout) for _ in range(self.configs.n_mlp_layers)])

            self.out = nn.Linear(configs.d_model+self.configs.d_extra_decoder_features, configs.d_output)
            self.softmax = nn.Softmax(dim=1)

            self.pretrain_out = nn.Linear(configs.d_model, configs.n_unique_tokens)


        def forecast(self, x):
            # Embedding

            enc_out = self.embedding(x)

            enc_out, attns = self.encoder(enc_out, attn_mask=None)

            return enc_out


        def forward(self, x, mask=False, aux = None):
            
            if type(mask) != bool: # pretraining
                x = self.forecast(x)

                x = x[range(x.shape[0]), mask, :] # cls

                for layer in self.mlp:
                    x = layer(x)

                y = self.pretrain_out(x)
                
                return y  

            else:
                
                if aux is not None:

                    x = self.forecast(x)

                    x = x[:, 0, :] # cls

                    # concat aux and x
                    x = torch.cat([x, aux], dim=1)

                    for layer in self.mlp:
                        x = layer(x)
                    
                    y = self.out(x)
                
                else:

                    x = self.forecast(x)

                    x = x[:, 0, :] # cls

                    for layer in self.mlp:
                        x = layer(x)

                    y = self.out(x)
                    
                    return y  
                    


class BERT_DANN(DANN_Model):

    class Model(nn.Module):

        def __init__(self, configs):
            super().__init__() 

            self.configs = configs

            assert configs.d_model % configs.n_heads == 0, "d_model must be a multiple of n_heads"

            configs.split_domain = False

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
                self.mlp_clf = nn.ModuleList([ResLayer(self.configs, self.configs.d_model+self.configs.d_extra_decoder_features, \
                                                       self.configs.d_model+self.configs.d_extra_decoder_features, self.configs.activation,
                                                    self.configs.dropout) for _ in range(self.configs.n_mlp_clf_layers)])
            else:
                self.mlp_clf = nn.ModuleList([LinearLayer(self.configs, self.configs.d_model+self.configs.d_extra_decoder_features, \
                                                          self.configs.d_model+self.configs.d_extra_decoder_features, self.configs.activation,
                                                    self.configs.dropout) for _ in range(self.configs.n_mlp_clf_layers)])
                
            if self.configs.res_learning:
                self.mlp_dom = nn.ModuleList([ResLayer(self.configs, self.configs.d_model, self.configs.d_model, self.configs.activation,
                                                    self.configs.dropout) for _ in range(self.configs.n_mlp_dom_layers)])
            else:
                self.mlp_dom = nn.ModuleList([LinearLayer(self.configs, self.configs.d_model, self.configs.d_model, self.configs.activation,
                                                    self.configs.dropout) for _ in range(self.configs.n_mlp_dom_layers)])

            self.out_dom = nn.Linear(self.configs.d_model, self.configs.d_output)
            self.out_clf = nn.Linear(self.configs.d_model+self.configs.d_extra_decoder_features, self.configs.d_output)
            self.softmax = nn.Softmax(dim=1)


        def forecast(self, x):
            # Embedding

            enc_out = self.embedding(x)

            enc_out, attns = self.encoder(enc_out, attn_mask=None)

            return enc_out

    

        def forward(self, x, mask=False, aux = None):
            
            if type(mask) != bool: # pretraining
                x = self.forecast(x)

                x = x[range(x.shape[0]), mask, :] # cls

                for layer in self.mlp:
                    x = layer(x)

                y = self.pretrain_out(x)
                
                return y 

            else:

                if aux is not None:

                    x = self.forecast(x)

                    x = x[:, 0, :]

                    rev_x = self.gradient_reverse(x)

                    x = torch.cat([x, aux], dim=1)

                    for layer in self.mlp_clf:
                        x = layer(x)

                    for layer in self.mlp_dom:
                        rev_x = layer(rev_x)
                    
                    y = self.out_clf(x)

                    dom = self.out_dom(rev_x)

                    return y, dom

                else:
                    x = self.forecast(x)

                    x = x[:, 0, :] # cls

                    rev_x = self.gradient_reverse(x)

                    for layer in self.mlp_clf:
                        x = layer(x)

                    for layer in self.mlp_dom:
                        rev_x = layer(rev_x)

                    y = self.out_clf(x)

                    dom = self.out_dom(rev_x)

                    return y, dom
            

class BERT_DCE_DANN(DCE_DANNModel):

    class Model(nn.Module):

        def __init__(self, configs):
            super().__init__() 

            self.configs = configs

            assert configs.d_model % configs.n_heads == 0, "d_model must be a multiple of n_heads"

            configs.split_domain = False

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
                self.mlp_clf = nn.ModuleList([ResLayer(self.configs, self.configs.d_model+self.configs.d_extra_decoder_features, \
                                                       self.configs.d_model+self.configs.d_extra_decoder_features, self.configs.activation,
                                                    self.configs.dropout) for _ in range(self.configs.n_mlp_clf_layers)])
            else:
                self.mlp_clf = nn.ModuleList([LinearLayer(self.configs, self.configs.d_model+self.configs.d_extra_decoder_features, \
                                                           self.configs.d_model+self.configs.d_extra_decoder_features, self.configs.activation,
                                                    self.configs.dropout) for _ in range(self.configs.n_mlp_clf_layers)])
                
            if self.configs.res_learning:
                self.mlp_dom = nn.ModuleList([ResLayer(self.configs, self.configs.d_model, self.configs.d_model, self.configs.activation,
                                                    self.configs.dropout) for _ in range(self.configs.n_mlp_dom_layers)])
            else:
                self.mlp_dom = nn.ModuleList([LinearLayer(self.configs, self.configs.d_model, self.configs.d_model, self.configs.activation,
                                                    self.configs.dropout) for _ in range(self.configs.n_mlp_dom_layers)])

            self.out_dom = nn.Linear(self.configs.d_model, self.configs.d_output)
            self.out_clf = nn.Linear(self.configs.d_model+self.configs.d_extra_decoder_features, self.configs.d_output)
            self.softmax = nn.Softmax(dim=1)


        def forecast(self, x):
            # Embedding

            enc_out = self.embedding(x)

            enc_out, attns = self.encoder(enc_out, attn_mask=None)

            return enc_out


        def forward(self, x, mask=False, aux = None):
            
            if type(mask) != bool: # pretraining
                x = self.forecast(x)

                x = x[range(x.shape[0]), mask, :] # cls

                for layer in self.mlp:
                    x = layer(x)

                y = self.pretrain_out(x)
                
                return y 

            else:

                if aux is not None:
                        
                    x = self.forecast(x)

                    x = x[:, 0, :]

                    rev_x = self.gradient_reverse(x)

                    x = torch.cat([x, aux], dim=1)

                    for layer in self.mlp_clf:
                        x = layer(x)

                    for layer in self.mlp_dom:
                        rev_x = layer(rev_x)
                    
                    y = self.out_clf(x)

                    dom = self.out_dom(rev_x)

                    return y, dom
                
                else:
                    x = self.forecast(x)

                    x = x[:, 0, :] # cls

                    rev_x = self.gradient_reverse(x)

                    for layer in self.mlp_clf:
                        x = layer(x)

                    for layer in self.mlp_dom:
                        rev_x = layer(rev_x)

                    y = self.out_clf(x)

                    dom = self.out_dom(rev_x)

                    return y, dom
            

class BERT_Hinge(HingeModel):

    class Model(nn.Module):


        def __init__(self, configs):
            super().__init__() 

            self.configs = configs

            assert configs.d_model % configs.n_heads == 0, "d_model must be a multiple of n_heads"

            configs.split_domain = False

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
                self.mlp = nn.ModuleList([ResLayer(self.configs, self.configs.d_model+self.configs.d_extra_decoder_features, \
                                                   self.configs.d_model+self.configs.d_extra_decoder_features, self.configs.activation,
                                                    self.configs.dropout) for _ in range(self.configs.n_mlp_layers)])
            else:
                self.mlp = nn.ModuleList([LinearLayer(self.configs, self.configs.d_model, self.configs.d_model, self.configs.activation,
                                                        self.configs.dropout) for _ in range(self.configs.n_mlp_layers)])

            self.out = nn.Linear(configs.d_model+self.configs.d_extra_decoder_features, configs.d_output)
            self.softmax = nn.Softmax(dim=1)


        def forecast(self, x):
            # Embedding

            enc_out = self.embedding(x)

            enc_out, attns = self.encoder(enc_out, attn_mask=None)

            return enc_out


        def forward(self, x, mask=False, aux = None):
            
            if type(mask) != bool: # pretraining
                x = self.forecast(x)

                x = x[range(x.shape[0]), mask, :] # cls

                for layer in self.mlp:
                    x = layer(x)

                y = self.pretrain_out(x)
                
                return y  

            else:

                x = self.forecast(x)

                x = x[:, 0, :] # cls

                for layer in self.mlp:
                    x = layer(x)

                y = self.out(x)
                
                return y  
            

class BERT_DoubleDecoder(ClassificationModel):

    class Model(nn.Module):

        def __init__(self, configs):
            super().__init__() 

            self.configs = configs

            assert configs.d_model % configs.n_heads == 0, "d_model must be a multiple of n_heads"

            configs.split_domain = True

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
                self.mlp1 = nn.ModuleList([ResLayer(self.configs, self.configs.d_model+self.configs.d_extra_decoder_features, \
                                                   self.configs.d_model+self.configs.d_extra_decoder_features, self.configs.activation,
                                                    self.configs.dropout) for _ in range(self.configs.n_mlp_layers)])
                self.mlp2 = nn.ModuleList([ResLayer(self.configs, self.configs.d_model+self.configs.d_extra_decoder_features, \
                                                    self.configs.d_model+self.configs.d_extra_decoder_features, self.configs.activation,
                                                     self.configs.dropout) for _ in range(self.configs.n_mlp_layers)])
            else:
                self.mlp2 = nn.ModuleList([LinearLayer(self.configs, self.configs.d_model+self.configs.d_extra_decoder_features, \
                                                      self.configs.d_model+self.configs.d_extra_decoder_features, self.configs.activation,
                                                        self.configs.dropout) for _ in range(self.configs.n_mlp_layers)])
                self.mlp1 = nn.ModuleList([LinearLayer(self.configs, self.configs.d_model+self.configs.d_extra_decoder_features, \
                                                        self.configs.d_model+self.configs.d_extra_decoder_features, self.configs.activation,
                                                            self.configs.dropout) for _ in range(self.configs.n_mlp_layers)])

            self.out1 = nn.Linear(configs.d_model+self.configs.d_extra_decoder_features, configs.d_output)
            self.out2 = nn.Linear(configs.d_model+self.configs.d_extra_decoder_features, configs.d_output)
            self.softmax = nn.Softmax(dim=1)

            self.pretrain_out = nn.Linear(configs.d_model, configs.n_unique_tokens)


        def forecast(self, x):
            # Embedding

            enc_out = self.embedding(x)

            enc_out, attns = self.encoder(enc_out, attn_mask=None)

            return enc_out


        def forward(self, x, mask=False, aux = None, domain = None):
            
            if type(mask) != bool: # pretraining
                x = self.forecast(x)

                x = x[range(x.shape[0]), mask, :] # cls

                for layer in self.mlp:
                    x = layer(x)

                y = self.pretrain_out(x)
                
                return y  

            else:
                # since we are splitting our input by domain at some point, need object to get back together
                y = torch.zeros(x.shape[0], self.configs.d_output).to(x.device) 
                
                # get masks that indexes the data based on domain ownership
                mask1 = torch.LongTensor([i for i in range(x.shape[0]) if domain[i,1] == 0]).to(x.device)
                mask2 = torch.LongTensor([i for i in range(x.shape[0]) if domain[i,1] == 1]).to(x.device)

                if aux is not None:

                    x = self.forecast(x)

                    x = x[:, 0, :] # cls

                    # concat aux and x
                    x = torch.cat([x, aux], dim=1)

                    x1 = x[mask1, :]
                    x2 = x[mask2, :]

                    for layer in self.mlp1:
                        x1 = layer(x1)
                    
                    for layer in self.mlp2:
                        x2 = layer(x2)

                    y1 = self.out1(x1)
                    y2 = self.out2(x2)

                    y[mask1,:] = y1
                    y[mask2,:] = y2
                    
                    return y

                else:

                    x = self.forecast(x)

                    x = x[:, 0, :] # cls

                    x1 = x[mask1, :]
                    x2 = x[mask2, :]

                    for layer in self.mlp1:
                        x1 = layer(x1)
                    
                    for layer in self.mlp2:
                        x2 = layer(x2)

                    y1 = self.out1(x1)
                    y2 = self.out2(x2)

                    y[mask1,:] = y1
                    y[mask2,:] = y2
                    
                    return y



class BERT_DANN_DoubleDecoder(DANN_Model):

    class Model(nn.Module):

        def __init__(self, configs):
            super().__init__() 

            self.configs = configs

            assert configs.d_model % configs.n_heads == 0, "d_model must be a multiple of n_heads"

            configs.split_domain = True

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
                self.mlp_clf1 = nn.ModuleList([ResLayer(self.configs, self.configs.d_model+self.configs.d_extra_decoder_features, \
                                                       self.configs.d_model+self.configs.d_extra_decoder_features, self.configs.activation,
                                                    self.configs.dropout) for _ in range(self.configs.n_mlp_clf_layers)])

                self.mlp_clf2 = nn.ModuleList([ResLayer(self.configs, self.configs.d_model+self.configs.d_extra_decoder_features, \
                                                         self.configs.d_model+self.configs.d_extra_decoder_features, self.configs.activation,
                                                    self.configs.dropout) for _ in range(self.configs.n_mlp_clf_layers)])

            else:
                self.mlp_clf1 = nn.ModuleList([LinearLayer(self.configs, self.configs.d_model+self.configs.d_extra_decoder_features, \
                                                          self.configs.d_model+self.configs.d_extra_decoder_features, self.configs.activation,
                                                    self.configs.dropout) for _ in range(self.configs.n_mlp_clf_layers)])
                self.mlp_clf2 = nn.ModuleList([LinearLayer(self.configs, self.configs.d_model+self.configs.d_extra_decoder_features, \
                                                            self.configs.d_model+self.configs.d_extra_decoder_features, self.configs.activation,
                                                        self.configs.dropout) for _ in range(self.configs.n_mlp_clf_layers)])

            if self.configs.res_learning:
                self.mlp_dom = nn.ModuleList([ResLayer(self.configs, self.configs.d_model, self.configs.d_model, self.configs.activation,
                                                    self.configs.dropout) for _ in range(self.configs.n_mlp_dom_layers)])
            else:
                self.mlp_dom = nn.ModuleList([LinearLayer(self.configs, self.configs.d_model, self.configs.d_model, self.configs.activation,
                                                    self.configs.dropout) for _ in range(self.configs.n_mlp_dom_layers)])

            self.out_dom = nn.Linear(self.configs.d_model, self.configs.d_output)
            self.out_clf1 = nn.Linear(self.configs.d_model+self.configs.d_extra_decoder_features, self.configs.d_output)
            self.out_clf2 = nn.Linear(self.configs.d_model+self.configs.d_extra_decoder_features, self.configs.d_output)
            self.softmax = nn.Softmax(dim=1)

        def forecast(self, x):
            # Embedding

            enc_out = self.embedding(x)

            enc_out, attns = self.encoder(enc_out, attn_mask=None)

            return enc_out


        def forward(self, x, mask=False, aux = None, domain = None):
            
            if type(mask) != bool: # pretraining
                x = self.forecast(x)

                x = x[range(x.shape[0]), mask, :] # cls

                for layer in self.mlp:
                    x = layer(x)

                y = self.pretrain_out(x)
                
                return y 

            else:
                y = torch.zeros(x.shape[0], self.configs.d_output).to(x.device)
                
                mask1 = torch.LongTensor([i for i in range(x.shape[0]) if domain[i,1] == 0]).to(x.device)
                mask2 = torch.LongTensor([i for i in range(x.shape[0]) if domain[i,1] == 1]).to(x.device)

                if aux is not None:

                    x = self.forecast(x)

                    x = x[:, 0, :] # cls

                    rev_x = self.gradient_reverse(x)

                    x = torch.cat([x, aux], dim=1)

                    x1 = x[mask1, :]
                    x2 = x[mask2, :]

                    for layer in self.mlp_clf1:
                        x1 = layer(x1)
                    
                    for layer in self.mlp_clf2:
                        x2 = layer(x2)

                    for layer in self.mlp_dom:
                        rev_x = layer(rev_x)

                    y1 = self.out_clf1(x1)
                    y2 = self.out_clf2(x2)

                    y[mask1,:] = y1
                    y[mask2,:] = y2

                    dom = self.out_dom(rev_x)

                    return y, dom
                
                else:

                    x = self.forecast(x)

                    x = x[:, 0, :] # cls

                    rev_x = self.gradient_reverse(x)

                    x1 = x[mask1, :]
                    x2 = x[mask2, :]

                    for layer in self.mlp_clf1:
                        x1 = layer(x1)

                    for layer in self.mlp_clf2:
                        x2 = layer(x2)

                    for layer in self.mlp_dom:
                        rev_x = layer(rev_x)

                    y1 = self.out_clf1(x1)
                    y2 = self.out_clf2(x2)

                    y[mask1,:] = y1
                    y[mask2,:] = y2

                    dom = self.out_dom(rev_x)

                    return y, dom
        

class BERT_DCE_DANN_DoubleDecoder(DCE_DANNModel):

    class Model(nn.Module):

        def __init__(self, configs):
            super().__init__() 

            self.configs = configs

            assert configs.d_model % configs.n_heads == 0, "d_model must be a multiple of n_heads"

            configs.split_domain = True

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
                self.mlp_clf1 = nn.ModuleList([ResLayer(self.configs, self.configs.d_model+self.configs.d_extra_decoder_features, \
                                                       self.configs.d_model+self.configs.d_extra_decoder_features, self.configs.activation,
                                                    self.configs.dropout) for _ in range(self.configs.n_mlp_clf_layers)])
                self.mlp_clf2 = nn.ModuleList([ResLayer(self.configs, self.configs.d_model+self.configs.d_extra_decoder_features, \
                                                         self.configs.d_model+self.configs.d_extra_decoder_features, self.configs.activation,
                                                    self.configs.dropout) for _ in range(self.configs.n_mlp_clf_layers)])
            else:
                self.mlp_clf1 = nn.ModuleList([LinearLayer(self.configs, self.configs.d_model+self.configs.d_extra_decoder_features, \
                                                           self.configs.d_model+self.configs.d_extra_decoder_features, self.configs.activation,
                                                    self.configs.dropout) for _ in range(self.configs.n_mlp_clf_layers)])
                self.mlp_clf2 = nn.ModuleList([LinearLayer(self.configs, self.configs.d_model+self.configs.d_extra_decoder_features, \
                                                              self.configs.d_model+self.configs.d_extra_decoder_features, self.configs.activation,
                                                     self.configs.dropout) for _ in range(self.configs.n_mlp_clf_layers)])

            if self.configs.res_learning:
                self.mlp_dom = nn.ModuleList([ResLayer(self.configs, self.configs.d_model, self.configs.d_model, self.configs.activation,
                                                    self.configs.dropout) for _ in range(self.configs.n_mlp_dom_layers)])
            else:
                self.mlp_dom = nn.ModuleList([LinearLayer(self.configs, self.configs.d_model, self.configs.d_model, self.configs.activation,
                                                    self.configs.dropout) for _ in range(self.configs.n_mlp_dom_layers)])

            self.out_dom = nn.Linear(self.configs.d_model, self.configs.d_output)
            self.out_clf1 = nn.Linear(self.configs.d_model+self.configs.d_extra_decoder_features, self.configs.d_output)
            self.out_clf2 = nn.Linear(self.configs.d_model+self.configs.d_extra_decoder_features, self.configs.d_output)
            self.softmax = nn.Softmax(dim=1)


        def forecast(self, x):
            # Embedding

            enc_out = self.embedding(x)

            enc_out, attns = self.encoder(enc_out, attn_mask=None)

            return enc_out

    

        def forward(self, x, mask=False, aux = None, domain = None):
            
            if type(mask) != bool: # pretraining
                x = self.forecast(x)

                x = x[range(x.shape[0]), mask, :] # cls

                for layer in self.mlp:
                    x = layer(x)

                y = self.pretrain_out(x)
                
                return y 

            else:
                y = torch.zeros(x.shape[0], self.configs.d_output).to(x.device)

                mask1 = torch.LongTensor([i for i in range(x.shape[0]) if domain[i,1] == 0]).to(x.device)
                mask2 = torch.LongTensor([i for i in range(x.shape[0]) if domain[i,1] == 1]).to(x.device)

                if aux is not None:

                    x = self.forecast(x)

                    x = x[:, 0, :] # cls

                    rev_x = self.gradient_reverse(x)

                    x = torch.cat([x, aux], dim=1)

                    x1 = x[mask1, :]
                    x2 = x[mask2, :]

                    for layer in self.mlp_clf1:
                        x1 = layer(x1)
                    
                    for layer in self.mlp_clf2:
                        x2 = layer(x2)

                    for layer in self.mlp_dom:
                        rev_x = layer(rev_x)

                    y1 = self.out_clf1(x1)
                    y2 = self.out_clf2(x2)

                    y[mask1,:] = y1
                    y[mask2,:] = y2

                    dom = self.out_dom(rev_x)

                    return y, dom

                else:

                    x = self.forecast(x)

                    x = x[:, 0, :] # cls

                    rev_x = self.gradient_reverse(x)

                    x1 = x[mask1, :]
                    x2 = x[mask2, :]

                    for layer in self.mlp_clf1:
                        x1 = layer(x1)
                    
                    for layer in self.mlp_clf2:
                        x2 = layer(x2)

                    for layer in self.mlp_dom:
                        rev_x = layer(rev_x)

                    y1 = self.out_clf1(x1)
                    y2 = self.out_clf2(x2)

                    y[mask1,:] = y1
                    y[mask2,:] = y2

                    dom = self.out_dom(rev_x)

                    return y, dom