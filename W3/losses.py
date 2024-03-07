import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Takes embeddings of two samples and a target label == 1 if
    samples are from the same class and label == 0 otherwise
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9
    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1) # squared distances
        losses = 0.5 * (target.float() * distances +
        (1.0 - target).float() * F.relu(self.margin
        - (distances + self.eps).sqrt()).pow(2))
        # sqrt() of a tiny number may be negative!
        return losses.mean() if size_average else losses.sum()


class TripletLoss(nn.Module):
    """
    Takes embeddings of an anchor sample, a positive sample
    and a negative sample
    """
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1) # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1) # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()