import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    '''
    -a * (1-p_t)^2 * log(p_t)
    '''

    def __init__(self, class_channel, alpha=None, gamma=2, reduction=True):
        super(FocalLoss, self).__init__()
        self.alpha = torch.ones(class_channel, requires_grad=True).cuda() if alpha is None else alpha
        self.gamma = gamma
        self.class_channel = class_channel
        self.reduction = reduction

    def forward(self, pre, tar):
        '''
        :param pre: (batch_size, class_channel, ...)
        :param tar: (batch_size, ...)
        :return: focal loss
        '''

        b, c = pre.size()[0], pre.size()[1]
        pre = pre.view(b, c, -1).transpose(1, 2).contiguous().view(-1, self.class_channel)
        tar = tar.view(-1, 1).long()

        pre_log_soft = torch.log_softmax(pre, dim=-1).gather(dim=1, index=tar).view(-1)
        pre_soft = pre_log_soft.exp()
        ones_tensor = torch.ones_like(pre_soft).cuda()
        focal_factor = torch.pow((ones_tensor - pre_soft), self.gamma)

        alpha = self.alpha.gather(dim=0, index=tar.view(-1))

        focal_loss = -1 * alpha * focal_factor * pre_log_soft

        return torch.mean(focal_loss) if self.reduction else torch.sum(focal_loss)
