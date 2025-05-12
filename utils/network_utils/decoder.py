import torch
import torch.nn.functional as F

def soft_argmax(prob, bins, axis):
    idxs = torch.arange(prob.shape[axis], device=prob.device).float()
    return (prob * idxs).sum(dim=axis)

def decode_simcc(pred_x, pred_y, input_size, bins, return_score=False):
    B, K, W = pred_x.shape
    _, _, H = pred_y.shape
    prob_x = F.softmax(pred_x, dim=2)
    prob_y = F.softmax(pred_y, dim=2)

    idx_x = soft_argmax(prob_x, bins, axis=2)
    idx_y = soft_argmax(prob_y, bins, axis=2)

    x = idx_x / bins * 4
    y = idx_y / bins * 4
    coords = torch.stack([x, y], dim=2)

    if return_score:
        score_x = torch.max(prob_x, dim=2)[0]
        score_y = torch.max(prob_y, dim=2)[0]
        conf = (score_x + score_y) / 2.0
        return coords, conf
    return coords
