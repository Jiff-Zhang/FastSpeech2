import torch
import torch.nn.functional as F

from sparseoptimizer.utils.loss import soft_ce_loss


def seq_cls_loss(
    sparse_outputs, dense_outputs,
    begin_layer=0, end_layer=-1, norm_hs=False,
    logit_weight=1., hs_weight=1., attn_weight=1., **kwargs
):
    sparse_hs, sparse_attns, sparse_logits = sparse_outputs.hidden_states, \
        sparse_outputs.attentions, sparse_outputs.logits
    dense_hs, dense_attns, dense_logits = dense_outputs.hidden_states, \
        dense_outputs.attentions, dense_outputs.logits

    # Logits loss(ce)
    logit_loss = soft_ce_loss(sparse_logits, dense_logits)

    if end_layer == -1:
        end_layer = len(sparse_hs)

    # Hidden states loss(mse)
    hs_loss = 0.
    for sparse_layer_hs, dense_layer_hs in \
        zip(sparse_hs[begin_layer:end_layer], dense_hs[begin_layer:end_layer]):
        # Normalize hidden states, usually for pre-norm model
        if norm_hs:
            eps = 1e-6

            sparse_hs_var = sparse_layer_hs.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
            dense_hs_var = dense_layer_hs.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)

            sparse_layer_hs = sparse_layer_hs * torch.rsqrt(sparse_hs_var + eps)
            dense_layer_hs = dense_layer_hs * torch.rsqrt(dense_hs_var + eps)

        hs_loss = hs_loss + F.mse_loss(sparse_layer_hs, dense_layer_hs)

    # Attentions loss(mse)
    attn_loss = 0.
    for sparse_layer_attn, dense_layer_attn in \
        zip(sparse_attns[begin_layer:end_layer], dense_attns[begin_layer:end_layer]):
        attn_loss = attn_loss + F.mse_loss(sparse_layer_attn, dense_layer_attn)

    return logit_weight * logit_loss + hs_weight * hs_loss + attn_weight * attn_loss
