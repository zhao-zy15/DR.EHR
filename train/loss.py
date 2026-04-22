import torch
import torch.nn as nn


class MultiSimilarityLoss(nn.Module):
    def __init__(self, threshold = 0.5, epsilon = 0.1, scale_pos = 1.0, scale_neg = 1.0):
        super(MultiSimilarityLoss, self).__init__()
        self.threshold = threshold
        self.epsilon = epsilon
        self.scale_pos = scale_pos
        self.scale_neg = scale_neg


    def forward(self, pos_sim, neg_sim):
        max_neg = torch.max(neg_sim, dim = 1).values
        min_pos = torch.min(pos_sim, dim = 1).values
        pos_mask = (pos_sim - self.epsilon) < max_neg.unsqueeze(1)
        neg_mask = (neg_sim + self.epsilon) > min_pos.unsqueeze(1)

        pos_loss = 1.0 / self.scale_pos * torch.log(
                1 + torch.sum(torch.exp(-self.scale_pos * (pos_sim * pos_mask - self.threshold))))
        neg_loss = 1.0 / self.scale_neg * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (neg_sim * neg_mask - self.threshold))))

        loss = (pos_loss + neg_loss) / pos_sim.shape[0]
        return loss

