from environment import *

class WeightedBinaryCrossEntropyLoss(nn.Module):
    def __init__(self, positive_weight, negative_weight):
        super(WeightedBinaryCrossEntropyLoss, self).__init__()
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight

    def forward(self, y_true, y_pred):
        # Clip predicted values to prevent log(0) and log(1)
        epsilon = 1e-7
        y_pred = torch.clamp(y_pred, epsilon, 1. - epsilon)

        # Compute binary cross-entropy loss
        bce_loss = - (y_true * torch.log(y_pred) * self.positive_weight + (1 - y_true) * torch.log(1 - y_pred) * self.negative_weight)

        bce_loss = - (y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred)) * y_true*self.positive_weight + \
                     - (y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred)) * (1 - y_true)*self.negative_weight


        return torch.mean(bce_loss)