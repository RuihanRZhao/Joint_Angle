import torch
import torch.nn.functional as F

def decode_simcc(pred_x, pred_y, input_size, bins):
    """
    解码 SimCC 分类输出为关键点坐标。

    Args:
        pred_x: Tensor[B, K, Cx]，SimCC x 分类 logits
        pred_y: Tensor[B, K, Cy]，SimCC y 分类 logits
        input_size: tuple (W, H) 原图尺寸
        bins: int，每像素细分的分类数量（例如 4）

    Returns:
        coords: Tensor[B, K, 2]，浮点像素坐标
    """
    B, K, Cx = pred_x.shape
    _, _, Cy = pred_y.shape
    W, H = input_size

    # softmax 归一化
    prob_x = F.softmax(pred_x, dim=2)  # [B, K, Cx]
    prob_y = F.softmax(pred_y, dim=2)  # [B, K, Cy]

    # 生成 index grid
    index_x = torch.arange(Cx, device=pred_x.device).float().view(1, 1, -1)  # [1,1,Cx]
    index_y = torch.arange(Cy, device=pred_y.device).float().view(1, 1, -1)

    # 期望坐标位置
    x = (prob_x * index_x).sum(dim=2) / bins * (W / (Cx / bins))
    y = (prob_y * index_y).sum(dim=2) / bins * (H / (Cy / bins))

    coords = torch.stack([x, y], dim=2)  # [B, K, 2]
    return coords
