import torch
import numpy as np

def keypoints_to_heatmaps(keypoints, output_size, sigma=2):
    """
    将关键点坐标转换为高斯热图。
    参数:
        keypoints: 形状为 (num_keypoints, 2) 或 (num_keypoints, 3) 的数组，包含关键点原图像像素坐标（以及可选的可见性标志）。
        heatmap_height: 输出热图的高度。
        heatmap_width: 输出热图的宽度。
        sigma: 高斯分布的标准差（热图坐标尺度下）。
    返回:
        heatmaps: numpy数组，形状为 (num_keypoints, heatmap_height, heatmap_width)，每个关键点对应一个热图。
    """
    heatmap_height, heatmap_width = output_size

    num_keypoints = len(keypoints)
    # 将关键点列表转换为浮点数组，以便计算
    keypoints_arr = np.array(keypoints, dtype=float)
    # 初始化热图张量
    heatmaps = np.zeros((num_keypoints, heatmap_height, heatmap_width), dtype=np.float32)
    if num_keypoints == 0:
        return heatmaps  # 无关键点则直接返回空热图

    # **按比例将原图坐标缩放到热图坐标系**（假设已知原图尺寸为 input_height, input_width）
    # 这里假设原图尺寸与热图尺寸的比例是恒定的。例如原图宽度=input_width，热图宽度=heatmap_width。
    # 如果原图尺寸未知，可根据数据集配置或上下文获取。以下示例假设下采样比例为 input_width/heatmap_width。
    input_width = heatmap_width * (1 if heatmap_width == 0 else (keypoints_arr[:, 0].max() / (heatmap_width - 1) if heatmap_width > 1 else 1))
    input_height = heatmap_height * (1 if heatmap_height == 0 else (keypoints_arr[:, 1].max() / (heatmap_height - 1) if heatmap_height > 1 else 1))
    # 计算缩放比例（热图尺寸 / 原图尺寸）
    scale_x = heatmap_width / input_width
    scale_y = heatmap_height / input_height

    radius = int(3 * sigma)
    diameter = 2 * radius + 1
    # 预先计算高斯核(直径x直径)，中心位于 (radius, radius)
    x = np.arange(diameter) - radius
    y = x[:, np.newaxis]
    gaussian_patch = np.exp(-(x**2 + y**2) / (2 * sigma * sigma))

    for i, kp in enumerate(keypoints_arr):
        # 如果提供了可见性标志且关键点不可见，则跳过绘制
        if kp.shape and len(kp) > 2:
            v = kp[2]
            if v < 0.5:
                continue
        x_orig, y_orig = kp[0], kp[1]
        # 跳过无效坐标
        if x_orig < 0 or y_orig < 0:
            continue
        # **将原图坐标按比例缩放到热图坐标**
        mu_x = int(x_orig * scale_x + 0.5)
        mu_y = int(y_orig * scale_y + 0.5)
        # 如果缩放后超出热图范围则跳过
        if mu_x < 0 or mu_x >= heatmap_width or mu_y < 0 or mu_y >= heatmap_height:
            continue

        # 计算高斯放置的边界坐标（ul为左上角，br为右下角的下一个索引）
        ul_x = mu_x - radius
        ul_y = mu_y - radius
        br_x = mu_x + radius + 1
        br_y = mu_y + radius + 1

        # 将边界裁剪到热图范围内
        img_x_min = max(0, ul_x)
        img_y_min = max(0, ul_y)
        img_x_max = min(heatmap_width, br_x)
        img_y_max = min(heatmap_height, br_y)
        # 计算高斯核对应的裁剪区域
        patch_x_min = max(0, -ul_x)
        patch_y_min = max(0, -ul_y)
        patch_x_max = patch_x_min + (img_x_max - img_x_min)
        patch_y_max = patch_y_min + (img_y_max - img_y_min)
        # 将高斯核区域贴到热图上
        heatmaps[i, img_y_min:img_y_max, img_x_min:img_x_max] = np.maximum(
            heatmaps[i, img_y_min:img_y_max, img_x_min:img_x_max],
            gaussian_patch[patch_y_min:patch_y_max, patch_x_min:patch_x_max]
        )
    return heatmaps

def batch_keypoints_to_heatmaps(batch_keypoints, heatmap_height, heatmap_width, sigma=2):
    """
    将一批样本的关键点集合转换为对应的高斯热图批次。
    参数:
        batch_keypoints: 可迭代对象，每个元素是一个样本的关键点数组（形状为 (num_keypoints, 2) 或 (num_keypoints, 3)）。
        heatmap_height: 输出热图的高度。
        heatmap_width: 输出热图的宽度。
        sigma: 高斯分布的标准差。
    返回:
        batch_heatmaps: numpy数组，形状为 (batch_size, num_keypoints, heatmap_height, heatmap_width)。
    """
    heatmaps_list = []
    for keypoints in batch_keypoints:
        heatmaps = keypoints_to_heatmaps(keypoints, heatmap_height, heatmap_width, sigma)
        heatmaps_list.append(heatmaps)
    # 将列表转换为批次张量
    batch_heatmaps = np.stack(heatmaps_list, axis=0)
    return batch_heatmaps

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
    max_vals = max_vals.unsqueeze(2)  # 现在形状是 (B, J, 1)
    keypoints = torch.cat([preds, max_vals], dim=2)  # (B, J, 3)
    # 对于那些最大值为非正的点，置置信度为0（无检测信号）
    # （一般不会出现这种情况，除非网络未检测出该关节）
    mask = (max_vals > 0).to(dtype=torch.float32)
    keypoints *= mask  # 若 max_val<=0，则对应坐标也清零
    return keypoints