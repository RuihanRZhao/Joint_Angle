import torch

def keypoints_to_heatmaps(keypoints, output_size, sigma):
    """
    将单张图像的 COCO 格式关键点转换为热图表示。
    Args:
        keypoints: 形状 (17, 3) 的关键点坐标和可见性，其中每一行为 (x, y, v)。
        output_size: 输出热图大小，可以是(int H, int W)元组或单个int（正方形边长）。
        sigma: 高斯分布的标准差（像素）。
    Returns:
        heatmaps: torch.Tensor，形状 (17, H, W)，每个通道为对应关键点的高斯热图。
    """
    # 解析输出尺寸
    if isinstance(output_size, int):
        H = W = output_size
    else:
        H, W = output_size
    # 将关键点数据转换为张量 (在原设备上)，数据类型为浮点
    if isinstance(keypoints, torch.Tensor):
        device = keypoints.device
        kps = keypoints.to(dtype=torch.float32)
    else:
        device = torch.device('cpu')
        kps = torch.as_tensor(keypoints, dtype=torch.float32, device=device)
    # 如果只有两列坐标，没有提供可见性，则默认都可见
    if kps.shape[-1] == 2:
        vis = torch.ones(kps.shape[0], device=device)
        kps = torch.cat([kps, vis.unsqueeze(1)], dim=1)
    # 利用 batch 接口生成 (1,17,H,W) 热图，再去除批维度
    heatmaps = batch_keypoints_to_heatmaps(kps.unsqueeze(0), (H, W), sigma)
    return heatmaps[0]  # 返回 (17,H,W)

def batch_keypoints_to_heatmaps(keypoints_batch, output_size, sigma):
    """
    将一批图像的关键点转换为热图。
    Args:
        keypoints_batch: 形状 (B, 17, 3) 的关键点张量或数组。
        output_size: 输出热图大小 (H, W) 或单个整数（正方形）。
        sigma: 高斯标准差。
    Returns:
        heatmaps_batch: torch.Tensor，形状 (B, 17, H, W) 的热图张量。
    """
    # 解析输出尺寸
    if isinstance(output_size, int):
        H = W = output_size
    else:
        H, W = output_size
    # 转换关键点批次为张量
    if isinstance(keypoints_batch, torch.Tensor):
        device = keypoints_batch.device
        kps = keypoints_batch.to(dtype=torch.float32)
    else:
        device = torch.device('cpu')
        kps = torch.as_tensor(keypoints_batch, dtype=torch.float32, device=device)
    B, J, _ = kps.shape  # B批大小, J=17关键点个数
    # 若关键点不包含可见性列，则补上一列全1（表示全部可见）
    if kps.shape[-1] == 2:
        vis = torch.ones(B, J, device=device)
        kps = torch.cat([kps, vis.unsqueeze(2)], dim=2)
    # 拆分坐标和可见性
    x = kps[..., 0]  # (B, J)
    y = kps[..., 1]  # (B, J)
    v = kps[..., 2]  # (B, J)
    # 创建输出热图张量，初始化为0
    heatmaps = torch.zeros((B, J, H, W), device=device, dtype=torch.float32)
    if B == 0:
        return heatmaps  # 空批次直接返回
    # 构造坐标网格 (H, W)
    yy = torch.arange(H, device=device, dtype=torch.float32).view(H, 1)
    xx = torch.arange(W, device=device, dtype=torch.float32).view(1, W)
    # 利用广播计算每个关节高斯：exp(-((X-x)^2 + (Y-y)^2) / (2*sigma^2))
    # 在这里，我们将 x, y 张量扩展为形状 (B, J, H, W)，与坐标网格进行运算
    x = x.view(B, J, 1, 1)
    y = y.view(B, J, 1, 1)
    # (B,J,H,W) = ((1,1,H,W) - (B,J,1,1))^2 + similarly for Y
    dist_sq = (xx - x)**2 + (yy - y)**2
    gauss = torch.exp(-dist_sq / (2 * (sigma ** 2)))
    # 将不可见的关键点对应的高斯图置零
    mask = (v > 0).view(B, J, 1, 1).to(dtype=torch.float32)
    heatmaps = gauss * mask
    return heatmaps

def heatmaps_to_keypoints(heatmaps):
    """
    将单张图片的热图解码为 COCO 格式关键点。
    Args:
        heatmaps: shape (17, H, W) 的热图张量或数组。
    Returns:
        keypoints: torch.Tensor，形状 (17, 3)，每行 (x, y, confidence)。
    """
    # 确保输入为 4维张量 (1,17,H,W)
    if isinstance(heatmaps, torch.Tensor):
        device = heatmaps.device
        hm = heatmaps.to(dtype=torch.float32)
    else:
        device = torch.device('cpu')
        hm = torch.as_tensor(heatmaps, dtype=torch.float32, device=device)
    if hm.dim() == 3:
        hm = hm.unsqueeze(0)
    keypoints_batch = batch_heatmaps_to_keypoints(hm)  # (1,17,3)
    return keypoints_batch[0]

def batch_heatmaps_to_keypoints(heatmaps_batch):
    """
    将一批热图张量解码为关键点坐标。
    Args:
        heatmaps_batch: shape (B, 17, H, W) 的热图张量或数组。
    Returns:
        keypoints_batch: torch.Tensor，形状 (B, 17, 3)，每个关键点为 (x, y, confidence)。
    """
    # 转换输入为张量
    if isinstance(heatmaps_batch, torch.Tensor):
        device = heatmaps_batch.device
        hm = heatmaps_batch.to(dtype=torch.float32)
    else:
        device = torch.device('cpu')
        hm = torch.as_tensor(heatmaps_batch, dtype=torch.float32, device=device)
    B, J, H, W = hm.shape
    # 展平热图的空间维度，方便取最大值
    # heatmaps_flat: (B, J, H*W)
    heatmaps_flat = hm.view(B, J, -1)
    # 找到每个关节热图的最大值以及索引
    max_vals, max_idxs = torch.max(heatmaps_flat, dim=2)  # (B, J)

    print(max_vals.shape)
    print(max_idxs.shape)

    # 计算 x, y 坐标：索引展开回2D平面的位置
    preds = torch.zeros((B, J, 2), device=device, dtype=torch.float32)
    preds[..., 0] = (max_idxs % W).float()  # x = idx mod W
    preds[..., 1] = (max_idxs // W).float()  # y = idx // W
    # 邻域偏移修正，提高亚像素精度
    for n in range(B):
        for j in range(J):
            px = int(preds[n, j, 0].item())
            py = int(preds[n, j, 1].item())
            # 确保在图像内部，避免越界访问邻居
            if 1 < px < W-1 and 1 < py < H-1:
                # 取该点左右和上下两个方向的值差
                diff_x = hm[n, j, py, px+1] - hm[n, j, py, px-1]
                diff_y = hm[n, j, py+1, px] - hm[n, j, py-1, px]
                # 根据差值的符号进行0.25像素的偏移
                preds[n, j, 0] += float(torch.sign(diff_x)) * 0.25
                preds[n, j, 1] += float(torch.sign(diff_y)) * 0.25
    # 将坐标和置信度拼接
    keypoints = torch.cat([preds, max_vals], dim=2)  # (B, J, 3)
    # 对于那些最大值为非正的点，置置信度为0（无检测信号）
    # （一般不会出现这种情况，除非网络未检测出该关节）
    mask = (max_vals > 0).to(dtype=torch.float32)
    keypoints *= mask  # 若 max_val<=0，则对应坐标也清零
    return keypoints