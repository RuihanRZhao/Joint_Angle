import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

def compute_miou(model, loader, device, threshold=0.5):
    """
    计算二分类分割的 Mean IoU（mIoU）。

    Args:
        model:    分割+姿态模型，forward 返回 (seg_logits, ...)
        loader:   DataLoader，返回 (imgs, masks, hm_lbl, paf_lbl, kps, vis, paths)
        device:   torch.device
        threshold:float, 将 sigmoid 输出转为二值 mask 的阈值

    Returns:
        float: 平均 IoU
    """
    model.eval()
    iou_list = []
    with torch.no_grad():
        for imgs, masks, *_ in tqdm(loader, desc="MIoU", leave=False):
            imgs = imgs.to(device)
            seg_logits = model(imgs)[0]                # [B,1,H,W]
            seg_prob   = torch.sigmoid(seg_logits).cpu().numpy()
            seg_pred   = (seg_prob > threshold).astype(np.uint8)
            gt_mask    = masks.cpu().numpy().astype(np.uint8)
            for pred, true in zip(seg_pred, gt_mask):
                pred_mask = pred[0]  # [H,W]
                true_mask = true[0]
                inter = np.logical_and(pred_mask, true_mask).sum()
                union = np.logical_or(pred_mask, true_mask).sum()
                if union > 0:
                    iou_list.append(inter / union)
    return float(np.mean(iou_list)) if iou_list else 0.0

def compute_pck_multi(multi_kps_list, gt_kps_list, gt_vis_list, image_sizes, thresh_ratio=0.05):
    """
    计算多人体场景下的 PCK (Percentage of Correct Keypoints)。

    Args:
        multi_kps_list: list of length B, 每项为 list of np.ndarray [K,2] 多人体关键点预测
        gt_kps_list:    list of length B, 每项为 np.ndarray [P, K, 2] 真实关键点
        gt_vis_list:    list of length B, 每项为 np.ndarray [P, K] 可见性
        image_sizes:    list of length B, 每项为 (W, H)
        thresh_ratio:   float, 相对阈值（相对于最大边长）

    Returns:
        dict: {
            'pck': float, 所有 keypoints 平均 PCK 值,
            'per_kpt': np.ndarray[K], 每个关键点的 PCK
        }
    """
    total_correct = 0
    total_visible = 0
    K = gt_kps_list[0].shape[1] if gt_kps_list[0].ndim == 3 else 0
    per_kpt_correct = np.zeros(K, dtype=int)
    per_kpt_total = np.zeros(K, dtype=int)

    for preds, gts, vis, size in zip(multi_kps_list, gt_kps_list, gt_vis_list, image_sizes):
        W, H = size
        max_edge = max(W, H)
        thresh = thresh_ratio * max_edge

        # 匹配方式：对每个 gt 实例，找最近的 pred 实例
        if len(preds) == 0:
            continue
        preds_arr = np.stack(preds, axis=0)  # [P_pred, K, 2]
        for p_idx in range(gts.shape[0]):
            gt = gts[p_idx]  # [K,2]
            v  = vis[p_idx]
            # 计算每个 pred 实例与当前 gt 实例的平均距离
            dists = np.linalg.norm(preds_arr - gt[None,:,:], axis=2)  # [P_pred, K]
            mean_dists = np.mean(dists, axis=1)
            best = np.argmin(mean_dists)
            d_k = dists[best]  # [K]

            # 计算可见关键点的正确数
            for k in range(K):
                if v[k] > 0:
                    per_kpt_total[k] += 1
                    if d_k[k] <= thresh:
                        total_correct += 1
                        per_kpt_correct[k] += 1
                    total_visible += 1

    pck = total_correct / total_visible if total_visible > 0 else 0.0
    per_kpt = per_kpt_correct.astype(float) / np.maximum(1, per_kpt_total)
    return {'pck': pck, 'per_kpt': per_kpt}


def evaluate_coco(coco_gt_json, results_json, iou_type='keypoints'):
    """
    使用 COCO API 评估多人体关键点检测 AP。

    Args:
        coco_gt_json:   str, COCO 格式的 ground-truth JSON 路径
        results_json:   str, 预测结果 JSON 路径，格式为 COCO keypoints 可接受格式
        iou_type:       'keypoints' 或其他 ['segm', 'bbox']

    Returns:
        dict: COCOeval.stats 各指标
    """
    coco_gt = COCO(coco_gt_json)
    coco_dt = coco_gt.loadRes(results_json)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    stats_names = ['AP', 'AP50', 'AP75', 'APm', 'APl', 'AR1', 'AR10', 'AR100']
    stats = {name: float(coco_eval.stats[i]) for i, name in enumerate(stats_names)}
    return stats
