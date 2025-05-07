import torch
import torch.nn as nn
import numpy as np
import cv2
from scipy.ndimage import maximum_filter
import wandb
from typing import List, Dict, Tuple, Optional

from utils.visualization import visualize_coco_keypoints
from utils.coco import COCO_PERSON_SKELETON

class PostProcess(nn.Module):
    """
    A nn.Module that takes network outputs (heatmap + PAF) and img_metas,
    and produces COCO‐format detection results + WandB visualization images.
    """
    def __init__(
        self,
        peak_thresh: float = 0.1,
        paf_score_thresh: float = 0.05,
        paf_count_thresh: float = 0.8,
    ):
        super().__init__()
        self.peak_thresh = peak_thresh
        self.paf_score_thresh = paf_score_thresh
        self.paf_count_thresh = paf_count_thresh

    def forward(
        self,
        heat_pred: torch.Tensor,        # [B, K, H, W]
        paf_pred: Optional[torch.Tensor],# [B, 2L, H, W] or None
        img_metas: List[Dict]           # list of dicts with keys:
                                        # 'img_id', 'orig_img', 'orig_h','orig_w', 'gt_anns'
    ) -> Tuple[List[Dict], List[Dict]]:
        B, K, H, W = heat_pred.shape
        device = heat_pred.device

        # 1) local max filter via max-pooling
        pooled = F.max_pool2d(heat_pred, kernel_size=3, stride=1, padding=1)
        mask = (heat_pred == pooled) & (heat_pred > self.peak_thresh)

        results: List[Dict] = []
        pred_ann_list: List[Dict] = []

        for i, meta in enumerate(img_metas):
            img_id = meta['img_id']
            orig_img = meta['orig_img']
            orig_h, orig_w = meta['orig_h'], meta['orig_w']

            # collect peaks per channel
            all_peaks: List[List[Tuple[float, float, float]]] = []
            for k in range(K):
                hmap = heat_pred[i, k]
                coords = mask[i, k].nonzero(as_tuple=False)  # [[y, x], ...]
                peaks = []
                for (y, x) in coords:
                    y_i, x_i = int(y.item()), int(x.item())
                    score = hmap[y_i, x_i].item()
                    dx, dy = 0.0, 0.0
                    if 0 < x_i < W - 1 and 0 < y_i < H - 1:
                        # gradient
                        dx_raw = 0.5 * (hmap[y_i, x_i + 1] - hmap[y_i, x_i - 1])
                        dy_raw = 0.5 * (hmap[y_i + 1, x_i] - hmap[y_i - 1, x_i])
                        dxx = hmap[y_i, x_i + 1] + hmap[y_i, x_i - 1] - 2 * hmap[y_i, x_i]
                        dyy = hmap[y_i + 1, x_i] + hmap[y_i - 1, x_i] - 2 * hmap[y_i, x_i]
                        dx = (dx_raw / (-dxx)) if dxx.abs() > 1e-6 else dx_raw
                        dy = (dy_raw / (-dyy)) if dyy.abs() > 1e-6 else dy_raw
                        dx, dy = dx.item(), dy.item()
                    peaks.append((x_i + dx, y_i + dy, score))
                all_peaks.append(peaks)

            # 2) PAF connections
            connection_candidates = {c: [] for c in range(len(COCO_PERSON_SKELETON))}
            if paf_pred is not None:
                for c, (a, b) in enumerate(COCO_PERSON_SKELETON):
                    candA, candB = all_peaks[a], all_peaks[b]
                    if not candA or not candB:
                        continue
                    paf_x = paf_pred[i, 2 * c]
                    paf_y = paf_pred[i, 2 * c + 1]
                    for ia, (ax, ay, sa) in enumerate(candA):
                        for ib, (bx, by, sb) in enumerate(candB):
                            dx, dy = bx - ax, by - ay
                            norm = (dx * dx + dy * dy) ** 0.5 + 1e-8
                            vx, vy = dx / norm, dy / norm
                            # sample 10 points on the segment
                            xs = torch.linspace(ax, bx, steps=10, device=device)
                            ys = torch.linspace(ay, by, steps=10, device=device)
                            ix = xs.round().long().clamp(0, W - 1)
                            iy = ys.round().long().clamp(0, H - 1)
                            vecs = paf_x[iy, ix] * vx + paf_y[iy, ix] * vy
                            mean_score = float(vecs.mean().item())
                            valid_ratio = float((vecs > self.paf_score_thresh).float().mean().item())
                            if mean_score > 0 and valid_ratio > self.paf_count_thresh:
                                score = mean_score + 0.5 * (sa + sb)
                                connection_candidates[c].append((ia, ib, score))
                    # greedy match
                    connection_candidates[c].sort(key=lambda x: x[2], reverse=True)
                    usedA = set();
                    usedB = set();
                    conns = []
                    for ia, ib, _ in connection_candidates[c]:
                        if ia not in usedA and ib not in usedB:
                            conns.append((ia, ib));
                            usedA.add(ia);
                            usedB.add(ib)
                    connection_candidates[c] = conns

            # 3) assemble persons
            persons: List[Dict[int, int]] = []
            for c, (a, b) in enumerate(COCO_PERSON_SKELETON):
                for ia, ib in connection_candidates[c]:
                    placed = False
                    for p in persons:
                        if (a in p and p[a] == ia) or (b in p and p[b] == ib):
                            p[a] = ia;
                            p[b] = ib;
                            placed = True;
                            break
                    if not placed:
                        persons.append({a: ia, b: ib})
            # add isolated peaks
            for k, peaks in enumerate(all_peaks):
                for idx in range(len(peaks)):
                    if not any(p.get(k) == idx for p in persons):
                        persons.append({k: idx})

            # 4) build COCO results
            for p in persons:
                kps_flat, scores = [], []
                for k in range(K):
                    if k in p:
                        x, y, sc = all_peaks[k][p[k]]
                        x *= orig_w / W;
                        y *= orig_h / H
                    else:
                        x = y = sc = 0.0
                    kps_flat += [float(x), float(y), float(sc)]
                    scores.append(sc)
                results.append({
                    'image_id': img_id,
                    'category_id': 1,
                    'keypoints': kps_flat,
                    'score': float(np.mean(scores)) if scores else 0.0
                })

            # visualization prep
            pred_anns = []
            for p in persons:
                kplist = []
                num_kp = 0
                for k in range(K):
                    if k in p:
                        x, y, sc = all_peaks[k][p[k]]
                        x *= orig_w / W;
                        y *= orig_h / H
                        v = 2 if sc > 0 else 0;
                        num_kp += 1
                    else:
                        x = y = v = 0.0
                    kplist += [x, y, v]
                pred_anns.append({'keypoints': kplist, 'num_keypoints': num_kp})

            pred_ann_list.append({'image_id': img_id, 'pred_anns': pred_anns})

        return results, pred_ann_list
