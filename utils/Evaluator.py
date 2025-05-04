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
            kps_arr = np.asarray(kps, dtype=np.float32)
            x = kps_arr[:, 0] * 4
            y = kps_arr[:, 1] * 4
            v = np.ones(len(kps_arr), dtype=np.float32)

            coco_kp = []
            for idx, kp_name in enumerate(self.keypoints):
                if idx < kps_arr.shape[0]:
                    x_val = float(x[idx])
                    y_val = float(y[idx])
                    v_val = float(v[idx])
                else:
                    x_val = y_val = v_val = 0.0
                coco_kp.extend([x_val, y_val, v_val])

            coco_kps.append({
                "image_id": int(image_id),
                "category_id": 1,
                "keypoints": coco_kp,
                "score": float(score)
            })
        return coco_kps

    def update(self, batch_preds, batch_gt):
        """更新预测结果"""
        for img_id, heatmaps, scores in batch_preds:
            # 如果没有检测到任何人，则跳过
            if not scores:
                continue
            keypoints = []
            for hm in heatmaps:
                y, x = np.unravel_index(np.argmax(hm), hm.shape)
                keypoints.append([x, y])

            mean_score = float(sum(scores) / len(scores))

            self.results.extend(
                self._convert_to_coco_format(img_id, [keypoints], [mean_score])
            )

    def compute_ap(self):
        """计算 COCO 标准 AP"""
        # 过滤掉不在 COCO GT 中的 image_id
        valid_ids = set(self.coco_gt.getImgIds())
        filtered_results = [r for r in self.results if r["image_id"] in valid_ids]
        num_invalid = len(self.results) - len(filtered_results)
        if num_invalid > 0:
            print(f"Warning: Skipped {num_invalid} results with invalid image_id")

        # 如果没有有效结果，跳过评估并返回0
        if not filtered_results:
            print("Warning: No valid pose detections available for evaluation.")
            return 0.0

        # 加载并评估
        coco_dt = self.coco_gt.loadRes(filtered_results)
        coco_eval = COCOeval(self.coco_gt, coco_dt, 'keypoints')

        # 设置 OKS sigma
        coco_eval.params.kpt_oks_sigmas = np.array([
            0.026, 0.025, 0.025, 0.035, 0.035,
            0.079, 0.072, 0.062, 0.107, 0.087,
            0.089, 0.068, 0.062, 0.107, 0.087,
            0.087, 0.089
        ])[:len(self.keypoints)]

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats[0]


