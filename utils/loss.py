from torch import tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.architecture import get_constr_out, softmax_constr_out

class HMLLoss(nn.Module):
    def __init__(
            self,
            R,
    ):
        super(HMLLoss, self).__init__()
        self.R = R

    def forward(self, probs, labels):
        loss, prediction_probs = mc_loss(probs, labels, self.R)

        return loss, prediction_probs

def mc_loss(probs, labels, R):
    constr_output = get_constr_out(probs, R)
    train_output = labels * probs
    train_output = get_constr_out(train_output, R)
    train_output = (1-labels) * constr_output + labels * train_output
    loss = F.binary_cross_entropy(
        train_output,
        labels
    )

    prediction_probs = constr_output.detach().cpu()

    return loss, prediction_probs

class SinglePathLoss(nn.Module):
    def __init__(self, R, leaf_index, mode="path", eps=1e-7):
        super().__init__()
        self.R = R
        self.register_buffer(
            "leaf_index", torch.as_tensor(leaf_index, dtype=torch.long)
        )
        assert mode in ("path", "leaf")
        self.mode = mode
        self.eps = eps

    def forward(self, logits, labels):
        coherent = softmax_constr_out(logits, self.R, self.leaf_index)

        if self.mode == "leaf":
            desc_marked = labels @ self.R.t()
            is_true_endpoint = (labels > 0) & (desc_marked.round() == 1)
            leaf_tgt = is_true_endpoint.index_select(1, self.leaf_index).float()
            leaf_probs = coherent.index_select(1, self.leaf_index)
            loss = -(leaf_tgt * torch.log(leaf_probs.clamp_min(self.eps))).sum(1).mean()
        else:
            loss = -(labels * torch.log(coherent.clamp_min(self.eps))).sum(1).mean()

        prediction_probs = coherent.detach().cpu()
        return loss, prediction_probs
