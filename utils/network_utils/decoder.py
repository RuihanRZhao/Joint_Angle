import torch
import torch.nn.functional as F

def soft_argmax_1d(prob: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Soft-argmax 实现，用于亚像素坐标预测
    输入: [B, K, C]
    输出: [B, K]
    """
    idxs = torch.arange(prob.shape[dim], device=prob.device).float()
    return torch.sum(prob * idxs, dim=dim)

def decode_simcc(pred_x: torch.Tensor, pred_y: torch.Tensor, input_size: tuple, bins: int, return_score: bool = False, downsample: int = 4):
    """
    解码 SimCC 表达为图像坐标：
    - 使用 softmax + soft-argmax
    - 支持可选返回置信度
    参数:
        pred_x: [B, K, Cx]
        pred_y: [B, K, Cy]
        input_size: (W, H)
        bins: 每像素分类数量（通常为4）
        return_score: 是否返回置信度
        downsample: 模型下采样倍数（默认4）
    返回:
        coords: [B, K, 2]
        scores: [B, K]（可选）
    """
    B, K, Cx = pred_x.shape
    _, _, Cy = pred_y.shape

    # 归一化为概率分布
    prob_x = F.softmax(pred_x, dim=2)
    prob_y = F.softmax(pred_y, dim=2)

    # soft-argmax 得到亚像素位置
    idx_x = soft_argmax_1d(prob_x, dim=2)  # [B, K]
    idx_y = soft_argmax_1d(prob_y, dim=2)  # [B, K]

    # 还原回像素坐标
    x = idx_x / bins * downsample
    y = idx_y / bins * downsample
    coords = torch.stack([x, y], dim=2)  # [B, K, 2]

    if return_score:
        score_x = prob_x.max(dim=2)[0]  # [B, K]
        score_y = prob_y.max(dim=2)[0]
        score = (score_x + score_y) / 2.0
        return coords, score

    return coords
