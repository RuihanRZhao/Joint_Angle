import torch

def decode_heatmap(heatmap):
    """
    将模型预测的关键点热图解码为关键点坐标和置信度。
    参数:
    - heatmap: 预测热图张量，形状 [B, num_joints, H, W]
    返回:
    - coords: 张量 [B, num_joints, 2]，每个关键点在热图上的 (x, y) 像素坐标
    - conf: 张量 [B, num_joints]，每个关键点对应的置信度分数 (热图最大值)
    注: 输出坐标为热图坐标系下的位置。若需转换回原图坐标，应乘以相应缩放因子(例如输入/输出尺寸比)并加上偏移。
    """
    assert heatmap.dim() == 4, "Heatmap tensor must be 4-dimensional [B, joints, H, W]"
    B, J, H, W = heatmap.shape
    # 将每个关节通道展平成一维，在每个通道上找到最大值及其索引
    vals, idx = torch.max(heatmap.view(B, J, -1), dim=2)  # vals: [B, J], idx: [B, J]
    # 将一维索引转换回二维坐标 (idx_y, idx_x)
    idx_y = idx // W
    idx_x = idx % W
    coords = torch.stack([idx_x, idx_y], dim=2).float()  # [B, J, 2], 顺序为 (x, y)
    conf = vals.float()  # [B, J] 对应的最大值作为置信度
    # （可选）若需次像素级精细化坐标，可根据邻域像素插值，这里省略。
    return coords, conf
