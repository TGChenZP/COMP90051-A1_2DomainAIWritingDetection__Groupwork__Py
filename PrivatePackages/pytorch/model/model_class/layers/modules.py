from environment import *
from model.model_class.layers.Masking import TriangularCausalMask
from math import sqrt

class LinearLayer(nn.Module):
    """ Dense Layer: Linear -> ReLU -> Dropout """
    def __init__(self, config, input_hidden_dim, output_hidden_dim, activation, dropout=0):
        super().__init__()
        self.config = config

        torch.manual_seed(self.config.random_state)

        self.layer = nn.Sequential(
            nn.Linear(input_hidden_dim, output_hidden_dim),
            activation,
            nn.Dropout(dropout)
        )

        init.normal_(self.layer[0].weight, mean=0, std=0.01)
        init.constant_(self.layer[0].bias, 0)
        
    def forward(self, x):
        return self.layer(x)
    
    
class ResLayer(nn.Module):
    """ Dense Residual Layer: x + Linear -> ReLU -> Dropout """
    def __init__(self, config, input_hidden_dim, output_hidden_dim, activation, dropout=0):
        super().__init__()
        self.config = config

        torch.manual_seed(self.config.random_state)

        self.layer = nn.Sequential(
            nn.Linear(input_hidden_dim, input_hidden_dim),
            activation,
            nn.Linear(input_hidden_dim, output_hidden_dim)
        )
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        for module in self.layer.modules():
            if isinstance(module, nn.Linear):
                init.normal_(module.weight, mean=0, std=0.01)
                init.constant_(module.bias, 0)
        
    def forward(self, x):
        return self.dropout(self.activation(x + self.layer(x)))

class FeedForward(nn.Module):
    """ Feed Forward layer of a transformer - to aggregate add attention outputs to the input """
    def __init__(self, config, d_model, activation, dropout=0):
        super().__init__()
        self.config = config

        torch.manual_seed(self.config.random_state)

        self.linear1 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model, d_model)
        self.activation = activation
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        init.normal_(self.linear1.weight, mean=0, std=0.01)
        init.normal_(self.linear2.weight, mean=0, std=0.01)
    
    def forward(self, x, x_attn):
        
        norm_x_add_attn = self.norm1(x + x_attn)
        x_add_attn = self.dropout(self.linear1(self.activation(self.linear2(norm_x_add_attn))))

        return self.norm2(norm_x_add_attn + x_add_attn)
    

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lambda_):
        # Save context for backward pass
        ctx.lambda_ = lambda_
        return input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse the gradient during backward pass
        grad_input = -ctx.lambda_ * grad_output.clone()
        return grad_input, None