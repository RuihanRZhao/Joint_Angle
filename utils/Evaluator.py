import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch


class SegmentationEvaluator:
    """人体分割mIoU评估"""

    def __init__(self, num_classes=2, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((num_classes, num_classes))

    def _fast_hist(self, pred, label):
        mask = (label != self.ignore_index)
        hist = np.bincount(
            self.num_classes * label[mask].astype(int) + pred[mask],
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        return hist

    def update(self, pred_logits, gt_mask):
        """更新混淆矩阵"""
        pred = torch.sigmoid(pred_logits).squeeze(1).cpu().numpy() > 0.5  # 二值化
        gt = gt_mask.squeeze(1).cpu().numpy().astype(int)

        for p, l in zip(pred, gt):
            self.confusion_matrix += self._fast_hist(p.flatten(), l.flatten())

    def compute_miou(self):
        """计算mIoU"""
        hist = self.confusion_matrix
        iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1e-10)
        miou = np.nanmean(iou[1:])  # 忽略背景类
        return miou


class PoseEvaluator:
    """姿态估计AP评估"""

    def __init__(self, ann_file, keypoints):
        self.coco_gt = COCO(ann_file)
        self.results = []
        self.keypoints = keypoints  # COCO关键点名称列表

    def _convert_to_coco_format(self, image_id, keypoints, scores):
        """将预测结果转换为COCO格式"""
        coco_kps = []
        for kps, score in zip(keypoints, scores):
            # 坐标缩放回原图尺寸
            kps = np.asarray(keypoints, dtype=np.float32)

            x = kps[:, 0] * 4  # 假设输入256x256，输出64x64热图
            y = kps[:, 1] * 4
            v = np.ones(len(kps))  # 可见性设为1

            coco_kp = []
            for k in self.keypoints:
                idx = self.keypoints.index(k)
                coco_kp.extend([x[idx], y[idx], v[idx]])

            coco_kps.append({
                "image_id": int(image_id),
                "category_id": 1,  # COCO人体类别
                "keypoints": coco_kp,
                "score": float(score)
            })
        return coco_kps

    def update(self, batch_preds, batch_gt):
        """更新预测结果"""
        for img_id, heatmaps, scores in batch_preds:
            # 从热图中提取关键点坐标
            keypoints = []
            for hm in heatmaps:
                y, x = np.unravel_index(np.argmax(hm), hm.shape)
                keypoints.append([x, y])

            mean_score = sum(scores) / len(scores)

            self.results.extend(
                self._convert_to_coco_format(img_id, [keypoints], [mean_score])
            )

    def compute_ap(self):
        """计算COCO标准AP"""
        coco_dt = self.coco_gt.loadRes(self.results)
        coco_eval = COCOeval(self.coco_gt, coco_dt, 'keypoints')

        # 设置评估参数
        coco_eval.params.kpt_oks_sigmas = np.array([
            0.026, 0.025, 0.025, 0.035, 0.035,  # 关键点OKS标准差
            0.079, 0.072, 0.062, 0.107, 0.087,
            0.089, 0.068, 0.062, 0.107, 0.087,
            0.087, 0.089
        ])[:len(self.keypoints)]

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats[0]  # AP@[0.5:0.95]


# 使用示例
if __name__ == "__main__":
    # 分割评估
    seg_eval = SegmentationEvaluator(num_classes=2)
    seg_logits = torch.randn(2, 1, 256, 256)  # 模拟模型输出
    seg_gt = torch.randint(0, 2, (2, 1, 256, 256))  # 模拟真实标签
    seg_eval.update(seg_logits, seg_gt)
    print(f"Segmentation mIoU: {seg_eval.compute_miou():.4f}")

    # 姿态评估
    coco_keypoints = ['nose', 'left_shoulder', 'right_shoulder']  # 示例关键点
    pose_eval = PoseEvaluator('path/to/annotations.json', coco_keypoints)

    # 模拟预测结果 (image_id, heatmaps, scores)
    fake_preds = [
        (0, np.random.rand(3, 64, 64), 0.9),
        (1, np.random.rand(3, 64, 64), 0.8)
    ]
    pose_eval.update(fake_preds, None)
    print(f"Pose AP: {pose_eval.compute_ap():.4f}")
