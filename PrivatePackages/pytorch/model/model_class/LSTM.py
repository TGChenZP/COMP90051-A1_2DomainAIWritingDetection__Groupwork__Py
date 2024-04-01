from model.model_class.layers.modules import *
from model.model_class.__template__ import *
from model.model_class.layers.SelfAttention_Family import FullAttention, AttentionLayer
from model.model_class.layers.Embed import PositionalWordEmbedding, WordEmbedding


class LSTM(ClassificationModel):
    class Model(nn.Module):
        def __init__(self, configs):
            super().__init__()
            self.configs = configs

            torch.manual_seed(self.configs.random_state)

            self.embedding = WordEmbedding(configs.d_model, configs.n_unique_tokens, configs.seq_len)

            self.lstm = nn.LSTM(
                input_size = self.configs.d_model,
                hidden_size = self.configs.d_model,
                num_layers = self.configs.n_recurrent_layers,
                batch_first=True,
                dropout= self.configs.dropout,
                bidirectional= self.configs.bidirectional
                )
            
            self.mha = AttentionLayer(attention = FullAttention(mask_flag=self.configs.mask_flag, scale=None, attention_dropout=self.configs.dropout, output_attention=False), 
                                      d_model = self.configs.d_model, n_heads = self.configs.n_heads) if self.configs.n_heads > 0 else None
            self.ffwd = FeedForward(self.configs, self.configs.d_model * 2 if self.configs.bidirectional else self.configs.d_model, self.configs.activation, self.configs.dropout) if self.configs.n_heads > 0 else None
            
            self.flatten = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.configs.d_model * self.configs.seq_len * 2 if self.configs.bidirectional else self.configs.d_model * self.configs.seq_len, 
                          self.configs.d_model * 2 if self.configs.bidirectional else self.configs.d_model),
            ) if self.configs.flatten else None

            if self.configs.res_learning:
                self.mlp = nn.ModuleList([ResLayer(self.configs, self.configs.d_model * 2 if self.configs.bidirectional else self.configs.d_model, 
                                                   self.configs.d_model * 2 if self.configs.bidirectional else self.configs.d_model, self.configs.activation,
                                                   self.configs.dropout) for _ in range(self.configs.n_mlp_layers)])
            else:
                self.mlp = nn.ModuleList([LinearLayer(self.configs, self.configs.d_model * 2 if self.configs.bidirectional else self.configs.d_model, 
                                                      self.configs.d_model * 2 if self.configs.bidirectional else self.configs.d_model,
                                                      self.configs.activation, self.configs.dropout) for _ in range(self.configs.n_mlp_layers)])

            self.out = nn.Linear(self.configs.d_model * 2 if self.configs.bidirectional else self.configs.d_model, self.configs.d_output)
            self.softmax = nn.Softmax(dim=1)

            # initialise parameters
            for name, param in self.lstm.named_parameters():
                if 'weight' in name:
                    init.normal_(param.data, mean=0, std=0.01)
                elif 'bias' in name:
                    init.constant_(param.data, 0)

            if self.flatten is not None:
                for name, param in self.flatten.named_parameters():
                    if 'weight' in name:
                        init.normal_(param.data, mean=0, std=0.01)
                    elif 'bias' in name:
                        init.constant_(param.data, 0)

            for layer in self.mlp:
                for name, param in layer.named_parameters():
                    if 'weight' in name:
                        init.normal_(param.data, mean=0, std=0.01)
                    elif 'bias' in name:
                        init.constant_(param.data, 0)
        
            init.normal_(self.out.weight.data, mean=0, std=0.01)
            init.constant_(self.out.bias.data, 0)

        def forward(self, x):
            
            x = self.embedding(x)

            x, (_, _) = self.lstm(x)

            x_attn, attns = self.mha(x, x, x, attn_mask = None) if self.configs.n_heads > 0 else (None, None)
            x = self.ffwd(x, x_attn) if self.configs.n_heads > 0 else x

            x = self.flatten(x) if self.flatten else x[:, -1, :]

            for layer in self.mlp:
                x = layer(x)

            y = self.softmax(self.out(x))

            return y