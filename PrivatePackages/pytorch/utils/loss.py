from environment import *
    

class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, predictions, targets):
        # Extract the positive class prediction and target

        # Compute hinge loss element-wise
        hinge_loss = torch.max(0*targets, 1 - predictions * targets)

        # Compute mean over all dimensions except batch dimension
        loss = hinge_loss.mean()

        return loss