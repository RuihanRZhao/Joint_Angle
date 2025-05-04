import os
import numpy as np
import json
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm, trange
import cv2

def compute_miou(model, loader, device, threshold=0.5):
    """
    计算二分类分割的 Mean IoU（mIoU）。

    Args:
        model:    分割+姿态模型，forward 返回 (seg_logits, ...)
        loader:   DataLoader，返回 (imgs, masks, ...)
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
            # 前向并转为 numpy
            seg_logits = model(imgs)[0]                     # [B,1,Hp,Wp]
            seg_prob   = torch.sigmoid(seg_logits).cpu().numpy()  # [B,1,Hp,Wp]
            seg_pred   = (seg_prob > threshold).astype(np.uint8)
            gt_mask    = masks.cpu().numpy().astype(np.uint8)     # [B,1,Ht,Wt]

            B = seg_pred.shape[0]
            for i in range(B):
                pred_mask  = seg_pred[i, 0]  # (Hp, Wp)
                true_mask  = gt_mask[i, 0]   # (Ht, Wt)
                Ht, Wt     = true_mask.shape

                # —— 关键：先把预测 resize 到 GT 大小 ——
                pred_resized = cv2.resize(
                    pred_mask,
                    (Wt, Ht),
                    interpolation=cv2.INTER_NEAREST
                )

                inter = np.logical_and(pred_resized, true_mask).sum()
                union = np.logical_or(pred_resized, true_mask).sum()
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
    stats_names = ['coco/AP', 'coco/AP50', 'coco/AP75', 'coco/APm', 'coco/APl', 'coco/AR1', 'coco/AR10', 'coco/AR100']
    stats = {name: float(coco_eval.stats[i]) for i, name in enumerate(stats_names)}
    return stats


def validate_all(model, loader, device,
                 coco_gt_json,
                 tmp_results_json,
                 thresh_ratio=0.05):
    """
    同时计算并返回：
      - 分割 Mean IoU （miou）
      - PCK（pck, per_kpt）
      - COCO Keypoints AP/AR（AP, AP50, AP75, APm, APl, AR1, AR10, AR100）
    """
    model.eval()

    # 1) 计算分割 mIoU
    miou = compute_miou(model, loader, device, threshold=0.5)

    # 2) COCO file_name->image_id 映射
    coco_gt = COCO(coco_gt_json)
    fname2id = { os.path.basename(img['file_name']): img['id']
                 for img in coco_gt.loadImgs(coco_gt.getImgIds()) }

    all_preds, all_gt_kps, all_gt_vis, img_sizes, coco_results = [], [], [], [], []

    with torch.no_grad():
        for imgs, masks, hm_lbl, paf_lbl, kps_list, vis_list, paths in loader:
            B, _, H, W = imgs.shape
            imgs = imgs.to(device)
            seg_logits, hm_out, paf_out, multi_kps = model(imgs)

            # — 对齐并收集预测 keypoints —
            for b in range(B):
                pred = multi_kps[b]
                if torch.is_tensor(pred):
                    pred = pred.cpu().numpy()  # [P_pred, K_model, 2]
                # GT 有多少关键点
                K_gt = kps_list[b].shape[1]
                # 如果模型输出通道 != GT 通道，则截断或补零
                # 先把 pred 变为 ndarray，确保有 .shape
                if isinstance(pred, list):
                    pred = np.asarray(pred)  # 变成 shape (P_pred, K_model, 2)
                  # 下面就可以安全用 pred.shape 了
                if pred.shape[1] > K_gt:
                    pred = pred[:, :K_gt, :]
                elif pred.shape[1] < K_gt:
                    pad = np.zeros((pred.shape[0], K_gt - pred.shape[1], 2), dtype=pred.dtype)
                    pred = np.concatenate([pred, pad], axis=1)
                all_preds.append(pred)

                # 收集 GT
                gt_k = kps_list[b]
                gt_v = vis_list[b]
                all_gt_kps.append(gt_k.cpu().numpy() if torch.is_tensor(gt_k) else gt_k)
                all_gt_vis.append(gt_v.cpu().numpy() if torch.is_tensor(gt_v) else gt_v)

                img_sizes.append((W, H))

                # —— 生成 COCO keypoints 预测条目 ——
                fname = os.path.basename(paths[b])
                image_id = fname2id.get(fname)
                if image_id is None:
                    continue
                for person_preds in pred:
                    xs = person_preds[:,0].tolist()
                    ys = person_preds[:,1].tolist()
                    kps_flat = []
                    for x,y in zip(xs, ys):
                        kps_flat += [x, y, 2]
                    coco_results.append({
                        'image_id': image_id,
                        'category_id': 1,
                        'keypoints': kps_flat,
                        'score': 1.0,
                    })

    # 3) 计算 PCK
    pck_stats = compute_pck_multi(
        multi_kps_list=all_preds,
        gt_kps_list=all_gt_kps,
        gt_vis_list=all_gt_vis,
        image_sizes=img_sizes,
        thresh_ratio=thresh_ratio
    )

    # 4) 写文件 & COCOeval
    with open(tmp_results_json, 'w') as f:
        json.dump(coco_results, f)
    ap_stats = evaluate_coco(coco_gt_json, tmp_results_json, iou_type='keypoints')

    # 5) 合并并返回
    return {
        'validate/miou'    : miou,
        'validate/pck'     : pck_stats['pck'],
        'validate/per_kpt' : pck_stats['per_kpt'],
        **ap_stats
    }