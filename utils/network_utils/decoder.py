import torch
import torch.nn.functional as F

def decode_simcc(pred_x, pred_y, input_size, bins, return_score=False):
    """
    SimCC 解码函数：
    - 输入 logits，softmax 得到概率分布；
    - 用 argmax 得到 index，再还原坐标。
    """
    B, K, W = pred_x.shape  # W = width * bins
    _, _, H = pred_y.shape

    # softmax 得到概率分布
    prob_x = F.softmax(pred_x, dim=2)  # [B,K,W]
    prob_y = F.softmax(pred_y, dim=2)  # [B,K,H]

    # argmax 得到索引
    idx_x = torch.argmax(prob_x, dim=2).float()  # [B,K]
    idx_y = torch.argmax(prob_y, dim=2).float()  # [B,K]

    # 还原到特征图坐标（除以 bins）
    x = idx_x / bins * 4  # 4是模型下采样比例
    y = idx_y / bins * 4

    # [B, K, 2]
    coords = torch.stack([x, y], dim=2)

    if return_score:
        # 置信度为最大概率
        score_x = torch.max(prob_x, dim=2)[0]
        score_y = torch.max(prob_y, dim=2)[0]
        conf = (score_x + score_y) / 2.0  # 平均作为关键点置信度
        return coords, conf
    else:
        return coords
