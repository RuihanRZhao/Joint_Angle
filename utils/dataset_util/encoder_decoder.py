import torch
import numpy as np

def keypoints_to_heatmaps(keypoints, input_size, output_size, sigma=2):
    """
    将关键点(像素坐标)转换为高斯热图。
    参数:
        keypoints: ndarray (num_joints, 2) 或 (num_joints, 3)，像素坐标 (x, y) 及可选可见性 v。
        input_size: (input_h, input_w)，模型输入裁剪图的尺寸（像素）。
        output_size: (out_h, out_w)，要生成的热图尺寸。
        sigma: 高斯标准差（热图坐标系下）。
    返回:
        heatmaps: ndarray (num_joints, out_h, out_w)，每个关节一张热图。
    """
    input_h, input_w = input_size
    out_h, out_w = output_size
    num_joints = len(keypoints)
    keypoints_arr = np.array(keypoints, dtype=float)
    heatmaps = np.zeros((num_joints, out_h, out_w), dtype=np.float32)
    if num_joints == 0:
        return heatmaps

    # 根据明确的输入/输出尺寸计算缩放比例
    scale_x = out_w / input_w
    scale_y = out_h / input_h

    # 预生成高斯核
    radius = int(3 * sigma)
    diameter = 2 * radius + 1
    xs = np.arange(diameter) - radius
    ys = xs[:, None]
    gaussian = np.exp(-(xs**2 + ys**2) / (2 * sigma * sigma))

    for i, kp in enumerate(keypoints_arr):
        # 可见性检查
        if kp.shape[0] > 2 and kp[2] < 0.5:
            continue
        x, y = kp[0], kp[1]
        if x < 0 or y < 0 or x > input_w or y > input_h:
            continue

        # 缩放到热图坐标
        mu_x = int(x * scale_x + 0.5)
        mu_y = int(y * scale_y + 0.5)
        if not (0 <= mu_x < out_w and 0 <= mu_y < out_h):
            continue

        # 计算高斯贴图范围
        ul_x, ul_y = mu_x - radius, mu_y - radius
        br_x, br_y = mu_x + radius + 1, mu_y + radius + 1

        x0 = max(0, ul_x); y0 = max(0, ul_y)
        x1 = min(out_w, br_x); y1 = min(out_h, br_y)

        gauss_x0 = max(0, -ul_x); gauss_y0 = max(0, -ul_y)
        gauss_x1 = gauss_x0 + (x1 - x0); gauss_y1 = gauss_y0 + (y1 - y0)

        # 合并
        heatmaps[i, y0:y1, x0:x1] = np.maximum(
            heatmaps[i, y0:y1, x0:x1],
            gaussian[gauss_y0:gauss_y1, gauss_x0:gauss_x1]
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