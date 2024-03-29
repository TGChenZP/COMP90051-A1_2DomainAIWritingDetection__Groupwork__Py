from model_class.__template__ import TorchToSklearn_Model
from model_class.Modules import *

class MLP(TorchToSklearn_Model):

    """ MLP Classifier or Regressor that can be used as a sklearn model """

    class Model(nn.Module):
        def __init__(self, CFG):
            super().__init__()
            self.CFG = CFG

            torch.manual_seed(self.CFG.random_state)

            if self.CFG.hidden_dim == 0: # hidden_dim = 0 => shrink the hidden layer size
                step_size = (self.CFG.input_dim - self.CFG.output_dim) // (self.CFG.hidden_layers - 1)
                if step_size == 0:
                    step_size = 1

                self.hidden_layer_size = [self.CFG.input_dim]
                self.hidden_layer_size += [
                    self.CFG.input_dim - i * step_size if self.CFG.input_dim - i * step_size > self.CFG.output_dim else self.CFG.output_dim
                            for i in range(1, self.CFG.hidden_layers)]

            else: # hidden_dim != 0 => use the same hidden layer size; note hidden_layers is the number of hidden layers including the output layer but excluding the input layer
                self.hidden_layer_size = [self.CFG.hidden_dim for _ in range(self.CFG.hidden_layers)]

            
            layers = []
            
            # Input layer
            layers.append(LinearLayer(self.CFG, self.CFG.input_dim, self.hidden_layer_size[0], self.CFG.dropout))
            if self.CFG.batchnorm:
                layers.append(nn.BatchNorm1d(self.hidden_layer_size[0]))
            layers.append(nn.ReLU())

            # Hidden layers
            for i in range(self.CFG.hidden_layers-1):
                layers.append(LinearLayer(self.CFG, self.hidden_layer_size[i], self.hidden_layer_size[i+1], self.CFG.dropout))
                if self.CFG.batchnorm:
                    layers.append(nn.BatchNorm1d(self.hidden_layer_size[i+1]))
                layers.append(nn.ReLU())

            # Output layer
            layers.append(LinearLayer(self.CFG, self.hidden_layer_size[-1], self.CFG.output_dim))

            self.full_model = nn.Sequential(*layers)
        
        def forward(self, X):

            y = self.full_model(X)

            if self.CFG.mode == 'Classification': # if classification, apply softmax to get the probability vector
                y = torch.softmax(y, dim=1)

            return y

    def __init__(self, 
                input_dim, 
                output_dim, 
                hidden_layers, 
                hidden_dim, 
                dropout,
                mode,
                batch_size,
                epochs,
                loss,
                TabularDataFactory,
                TabularDataset,
                lr=1e-3,
                random_state=42,
                grad_clip=False,
                batchnorm=False,
                rootpath = './',
                name="MLP"):

        class CFG:
            def __init__(self, input_dim, output_dim, hidden_layers, hidden_dim, dropout, mode, batch_size, epochs, lr, random_state, grad_clip, batchnorm, loss, TabularDataFactory, TabularDataset, rootpath, name):
                self.input_dim = input_dim
                self.output_dim = output_dim
                self.hidden_layers = hidden_layers
                self.hidden_dim = hidden_dim
                self.dropout = dropout
                self.mode = mode
                self.batch_size = batch_size
                self.epochs = epochs
                self.lr = lr
                self.random_state = random_state
                self.grad_clip = grad_clip
                self.batchnorm = batchnorm
                self.loss = loss
                self.TabularDataFactory = TabularDataFactory
                self.TabularDataset = TabularDataset
                self.rootpath = rootpath
                self.name = name


        self.CFG = CFG(input_dim, output_dim, hidden_layers, hidden_dim, dropout, mode, batch_size, epochs, lr, random_state, grad_clip, batchnorm, loss, TabularDataFactory, TabularDataset, rootpath, name)
        super().__init__(self.CFG, name=self.CFG.name)