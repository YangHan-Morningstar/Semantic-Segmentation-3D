import torch.nn as nn
import torch


class SquareDiceLoss(nn.Module):
    def __init__(self, num_class, epsilon=1e-5):
        super(SquareDiceLoss, self).__init__()
        self.num_class = num_class
        self.epsilon = epsilon

    def forward(self, score, target):
        loss = 0
        for class_index in range(1, self.num_class):
            pre, y = score[:, class_index, :, :, :], (target == class_index).float()
            intersect = torch.sum(pre * y)
            pre_sum = torch.sum(torch.pow(pre, 2))
            y_sum = torch.sum(torch.pow(y, 2))
            loss += (2 * (intersect + self.epsilon)) / (pre_sum + y_sum + self.epsilon)
        loss = (self.num_class - 1) - loss

        return loss


class DiceLoss(nn.Module):
    def __init__(self, num_class, epsilon=1e-5):
        super(DiceLoss, self).__init__()
        self.num_class = num_class
        self.epsilon = epsilon

    def forward(self, score, target):
        loss = 0
        for class_index in range(1, self.num_class):
            pre, y = score[:, class_index, :, :, :], (target == class_index).float()
            intersect = torch.sum(pre * y)
            pre_sum = torch.sum(pre)
            y_sum = torch.sum(y)
            loss += (2 * (intersect + self.epsilon)) / (pre_sum + y_sum + self.epsilon)
        loss = (self.num_class - 1) - loss

        return loss
