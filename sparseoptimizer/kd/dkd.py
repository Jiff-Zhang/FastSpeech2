import torch
import torch.nn.functional as F


def dkd_loss(stu_logits, tea_logits, labels, alpha=.5, beta=2., temperature=1., reduction='batchmean'):
    """
    Decoupled Knowledge Distillation Loss, refer to: https://arxiv.org/abs/2203.08679(CVPR 2022)
    """

    '''i. generate masks'''
    gt_mask = torch.zeros_like(stu_logits).scatter_(-1, labels.unsqueeze(-1), 1.)
    ngt_mask = 1. - gt_mask

    '''ii. logits -> probability'''
    stu_prob = F.softmax(stu_logits / temperature, dim=-1)
    tea_prob = F.softmax(tea_logits / temperature, dim=-1)

    '''iii. sum of probability for target & non-target classes'''
    stu_prob_gt, stu_prob_ngt_sum = (stu_prob * gt_mask).sum(-1, keepdim=True), (stu_prob * ngt_mask).sum(-1, keepdim=True)
    tea_prob_gt, tea_prob_ngt_sum = (tea_prob * gt_mask).sum(-1, keepdim=True), (tea_prob * ngt_mask).sum(-1, keepdim=True)

    '''iv. kd loss of target class'''
    tckd_loss = F.kl_div(
        torch.cat([stu_prob_gt, stu_prob_ngt_sum], dim=-1).log(),
        torch.cat([tea_prob_gt, tea_prob_ngt_sum], dim=-1),
        reduction=reduction
    ) * temperature ** 2

    '''v. individual probability for non-target classes'''
    stu_prob_ngt = F.log_softmax(stu_logits / temperature - 1000. * gt_mask, dim=-1)
    tea_prob_ngt = F.softmax(tea_logits / temperature - 1000. * gt_mask, dim=-1)

    '''vi. kd loss of each non-target class'''
    nckd_loss = F.kl_div(stu_prob_ngt, tea_prob_ngt, reduction=reduction) * temperature ** 2

    '''vii. dkd loss: weighted sum of tckd & nckd loss'''
    return alpha * tckd_loss + beta * nckd_loss
