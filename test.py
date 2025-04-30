# validation_script.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

from torch.utils.data import DataLoader

from datasets.coco import CocoMultiTask, get_transform, collate_fn
from models.student_model import StudentModel
from utils.loss import SegmentationLoss, KeypointsLoss
from utils.visualization import overlay_segmentation, draw_pose, heatmaps_to_keypoints


def validate_pipeline(
        data_root: str,
        ann_file: str,
        model_path: str = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_dir: str = None,
        num_samples: int = 4,
        confidence_threshold: float = 0.3
):
    """端到端验证流程"""
    # 初始化目录
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # 1. 数据加载验证
    dataset = CocoMultiTask(
        root=data_root,
        transform=get_transform(train=False)
    )
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, shuffle=True)

    print("\n=== 数据加载验证 ===")
    print(f"数据集样本数: {len(dataset)}")
    batch = next(iter(loader))
    images, masks, keypoints_list, vis_list = batch
    print(f"图像张量形状: {images.shape} (BxCxHxW)")
    print(f"掩码张量形状: {masks.shape} (Bx1xHxW)")
    print(f"关键点列表长度: {len(keypoints_list)} (batch_size)")
    print(f"首样本关键点形状: {keypoints_list[0].shape} (Nx2)")

    # 2. 可视化原始数据样本
    print("\n=== 原始数据可视化 ===")
    for i in range(min(len(images), num_samples)):
        img_np = images[i].permute(1, 2, 0).cpu().numpy()  # HWC
        mask_np = masks[i][0].cpu().numpy()  # HW

        # 转换关键点坐标到图像尺寸
        h, w = img_np.shape[:2]
        kps = keypoints_list[i].cpu().numpy() * np.array([w, h])

        # 可视化
        vis_img = overlay_segmentation(
            (img_np * 255).astype(np.uint8),
            (mask_np > 0.5).astype(np.uint8),
            alpha=0.3,
            color=(0, 255, 0)
        )
        vis_img = draw_pose(
            vis_img,
            kps,
            color_map='hsv'
        )

        if save_dir:
            save_path = os.path.join(save_dir, f"sample_{i}_gt.jpg")
            cv2.imwrite(save_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
            print(f"保存GT可视化: {save_path}")
        else:
            plt.imshow(vis_img)
            plt.title(f"Sample {i} - Ground Truth")
            plt.show()

    # 3. 模型验证（如果提供模型路径）
    if model_path:
        print("\n=== 模型推理验证 ===")
        model = StudentModel(num_keypoints=17, seg_channels=1).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        with torch.no_grad():
            images = images.to(device)
            seg_logits, pose_logits = model(images)

        # 转换模型输出
        pred_masks = (torch.sigmoid(seg_logits) > 0.5).cpu().numpy()
        pose_heatmaps = torch.sigmoid(pose_logits).cpu()

        # 可视化预测结果
        for i in range(min(len(images), num_samples)):
            # 获取预测结果
            img_np = images[i].permute(1, 2, 0).cpu().numpy()
            pred_mask = pred_masks[i][0]
            keypoints, confidences = heatmaps_to_keypoints(pose_heatmaps[i], threshold=confidence_threshold)

            # 转换坐标到图像尺寸
            h, w = img_np.shape[:2]
            keypoints *= np.array([w / pose_heatmaps.shape[-1], h / pose_heatmaps.shape[-2]])

            # 可视化
            vis_img = overlay_segmentation(
                (img_np * 255).astype(np.uint8),
                pred_mask.astype(np.uint8),
                alpha=0.3,
                color=(0, 0, 255)
            )
            vis_img = draw_pose(
                vis_img,
                keypoints[confidences > confidence_threshold],
                confidences=confidences,
                color_map='confidence'
            )

            if save_dir:
                save_path = os.path.join(save_dir, f"sample_{i}_pred.jpg")
                cv2.imwrite(save_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
                print(f"保存预测可视化: {save_path}")
            else:
                plt.imshow(vis_img)
                plt.title(f"Sample {i} - Predictions")
                plt.show()

        # 4. 损失计算验证
        print("\n=== 损失计算验证 ===")
        seg_loss_fn = SegmentationLoss()
        pose_loss_fn = KeypointsLoss()

        seg_loss = seg_loss_fn(seg_logits, masks.to(device))
        pose_loss = pose_loss_fn(pose_logits, keypoints_list, vis_list)

        print(f"分割损失: {seg_loss.item():.4f}")
        print(f"姿态损失: {pose_loss.item():.4f}")


if __name__ == "__main__":
    datasets_path = "/run/data/"
    # 配置参数
    config = {
        "data_root": f"./{datasets_path}/coco/images/val2017",
        "ann_file": f"./{datasets_path}/coco/annotations/person_keypoints_val2017.json",
        "model_path": "./checkpoints/best_student.pth",  # 可选
        "save_dir": "./validation_results",
        "device": "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        "num_samples": 4,
        "confidence_threshold": 0.4
    }

    validate_pipeline(**config)
