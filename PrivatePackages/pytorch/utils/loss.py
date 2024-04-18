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
    

class ThresholdCrossEntropyLoss(nn.Module):
    def __init__(self, threshold, neg_weight=1, pos_weight=1):
        super(ThresholdCrossEntropyLoss, self).__init__()
        self.neg_weight = neg_weight
        self.pos_weight = pos_weight
        self.threshold = threshold

    def forward(self, pred, target):
        p = F.softmax(pred, dim=1)  # Apply softmax along the classes dimension

        pos_loss = -(target[:, 1] * (torch.log(p[:, 1] + 1e-7) + np.log(self.threshold))) * self.pos_weight
        neg_loss = -(target[:, 0] * (torch.log(1 - p[:, 1] + 1e-7) + np.log(self.threshold))) * self.neg_weight

        loss = torch.max(pos_loss + neg_loss, torch.zeros_like(pos_loss))  # Cap loss at 0
        loss = torch.mean(loss)  # Calculate mean loss across the batch
        return loss
