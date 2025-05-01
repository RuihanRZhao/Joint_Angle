import os
import numpy as np
import fiftyone as fo
import fiftyone.zoo as foz


def prepare_coco_dataset(root_dir, split, max_samples=None):
    """
    加载 COCO-2017 数据集并提取样本列表。

    参数:
        root_dir (str): 数据集下载和存储路径
        split (str): 数据拆分，"train" 或 "validation"
        max_samples (int): 最大样本数，None 表示全部

    返回:
        List[Dict]: 样本列表，每个字典包含:
            - image_path (str): 图像文件路径
            - mask (np.ndarray): 合并后二值分割掩码
            - keypoints (np.ndarray): 关键点坐标数组 (N,2)
            - visibility (np.ndarray): 对应的可见性数组 (N,)
    """
    # 配置 FiftyOne 下载目录
    fo.config.dataset_zoo_dir = root_dir
    # 加载 COCO-2017 指定 split，仅保留 'person' 类，并加载分割和关键点标签
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split=split,
        label_types=["detections", "segmentations", "keypoints"],
        classes=["person"],
        max_samples=max_samples,
    )

    samples = []
    for sample in dataset:
        img_path = sample.filepath  # 图像路径
        # 获取人物实例列表（Detection 对象）
        dets = sample.ground_truth.detections if hasattr(sample, "ground_truth") \
            else sample.detections.detections
        if not dets:
            continue  # 若该图像没有检测到人物，则跳过

        # 合并所有实例掩码（逻辑或），生成二值掩码
        combined_mask = None
        all_keypoints = []
        all_visible = []
        for det in dets:
            mask = det.mask  # 二值掩码 (H, W)，True 表示目标像素
            if mask is None:
                continue
            mask = mask.astype(bool)
            if combined_mask is None:
                combined_mask = mask.copy()
            else:
                combined_mask |= mask  # 合并掩码

            # 提取关键点 (x,y,v) 列表，每个目标17个关键点
            if det.keypoints is not None:
                kps = np.array(det.keypoints).reshape(-1, 3)  # (17,3)
                coords = kps[:, :2]
                vis = kps[:, 2]
                # 收集所有关键点及其可见性
                for (x, y), v in zip(coords, vis):
                    all_keypoints.append([float(x), float(y)])
                    all_visible.append(int(v))

        if combined_mask is None:
            # 如果没有任何人物实例，跳过此样本
            continue

        # 转换为 NumPy 数组并保存
        combined_mask = combined_mask.astype(np.uint8)  # 转换为 0/1 掩码
        keypoints_arr = np.array(all_keypoints, dtype=np.float32).reshape(-1, 2)
        visibility_arr = np.array(all_visible, dtype=np.int32)
        samples.append({
            "image_path": img_path,
            "mask": combined_mask,
            "keypoints": keypoints_arr,
            "visibility": visibility_arr,
        })

    return samples

