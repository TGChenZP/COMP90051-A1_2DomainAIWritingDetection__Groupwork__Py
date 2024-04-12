from environment import *
    

class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, predictions, targets):
        # Extract the positive class prediction and target
        positive_predictions = predictions  # Assuming the positive class is index 1
        positive_targets = targets  # Assuming the positive class is index 1

        # Convert targets from {0, 1} to {-1, 1}
        targets = 2 * positive_targets - 1

        # Compute hinge loss element-wise
        hinge_loss = torch.max(torch.zeros_like(positive_predictions), 1 - positive_predictions * targets)

        # Compute mean over all dimensions except batch dimension
        loss = hinge_loss.mean()

        return loss