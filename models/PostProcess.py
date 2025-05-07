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
        heat_np = heat_pred.detach()
        paf_np  = paf_pred.detach() if paf_pred is not None else None

        print(heat_np.shape)

        results: List[Dict] = []
        pred_ann_list: List[Dict] =[]

        # select which img_ids to visualize

        for i, meta in enumerate(img_metas):
            img_id   = meta['img_id']
            orig_img = meta['orig_img']                   # BGR np.ndarray
            orig_h   = meta['orig_h']
            orig_w   = meta['orig_w']

            # 1) peak detection + subpixel refinement
            all_peaks: List[List[Tuple[float,float,float]]] = []
            for k in range(K):
                hmap = heat_np[i, k]
                mask = (hmap == maximum_filter(hmap, size=3)) & (hmap > self.peak_thresh)
                coords = np.argwhere(mask)
                peaks = []
                for (y, x) in coords:
                    score = float(hmap[y, x])
                    dx = dy = 0.0
                    if 0 < x < W-1 and 0 < y < H-1:
                        dx = 0.5*(hmap[y, x+1] - hmap[y, x-1])
                        dy = 0.5*(hmap[y+1, x] - hmap[y-1, x])
                        dxx = hmap[y, x+1] + hmap[y, x-1] - 2*hmap[y, x]
                        dyy = hmap[y+1, x] + hmap[y-1, x] - 2*hmap[y, x]
                        if dxx != 0: dx /= -dxx
                        if dyy != 0: dy /= -dyy
                    peaks.append((x+dx, y+dy, score))
                all_peaks.append(peaks)

            # 2) build PAF connections & greedy select
            connection_candidates = {c: [] for c in range(len(COCO_PERSON_SKELETON))}
            if paf_np is not None:
                for c, (a, b) in enumerate(COCO_PERSON_SKELETON):
                    candA = all_peaks[a]
                    candB = all_peaks[b]
                    if not candA or not candB:
                        continue
                    paf_x = paf_np[i, 2*c]
                    paf_y = paf_np[i, 2*c+1]
                    for ia, (ax,ay,sa) in enumerate(candA):
                        for ib, (bx,by,sb) in enumerate(candB):
                            dx, dy = bx-ax, by-ay
                            norm = np.hypot(dx, dy) + 1e-8
                            vx, vy = dx/norm, dy/norm
                            xs = np.linspace(ax, bx, 10).astype(int)
                            ys = np.linspace(ay, by, 10).astype(int)
                            vec_scores = []
                            for xx, yy in zip(xs, ys):
                                if 0 <= xx < W and 0 <= yy < H:
                                    vec = np.array([paf_x[yy, xx], paf_y[yy, xx]])
                                    vec_scores.append(vec.dot([vx, vy]))
                            if vec_scores:
                                mean_score = float(np.mean(vec_scores))
                                valid_ratio = np.mean(np.array(vec_scores) > self.paf_score_thresh)
                                if mean_score > 0 and valid_ratio > self.paf_count_thresh:
                                    connection_candidates[c].append(
                                        (ia, ib, mean_score + 0.5*(sa+sb))
                                    )
                    # greedy
                    connection_candidates[c].sort(key=lambda x: x[2], reverse=True)
                    usedA, usedB = set(), set()
                    conns: List[Tuple[int,int]] = []
                    for ia, ib, _ in connection_candidates[c]:
                        if ia not in usedA and ib not in usedB:
                            conns.append((ia, ib))
                            usedA.add(ia); usedB.add(ib)
                    connection_candidates[c] = conns

            # 3) assemble persons
            persons: List[Dict[int,int]] = []
            for c, (a,b) in enumerate(COCO_PERSON_SKELETON):
                for ia, ib in connection_candidates[c]:
                    placed = False
                    for p in persons:
                        if (a in p and p[a]==ia) or (b in p and p[b]==ib):
                            p[a]=ia; p[b]=ib; placed = True; break
                    if not placed:
                        persons.append({a:ia, b:ib})
            # isolated peaks
            for k, peaks in enumerate(all_peaks):
                for idx in range(len(peaks)):
                    if not any(p.get(k)==idx for p in persons):
                        persons.append({k: idx})

            # 4) build COCO‐format results
            for p in persons:
                kps_flat, scores = [], []
                for k in range(K):
                    if k in p:
                        x, y, sc = all_peaks[k][p[k]]
                        x *= (orig_w / W); y *= (orig_h / H)
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

            # visualization data
            pred_anns = []
            for p in persons:
                kplist = []; num_kp = 0
                for k in range(K):
                    if k in p:
                        x, y, sc = all_peaks[k][p[k]]
                        x *= (orig_w / W); y *= (orig_h / H)
                        v = 2 if sc>0 else 0; num_kp += 1
                    else:
                        x = y = v = 0.0
                    kplist += [x, y, v]
                pred_anns.append({'keypoints': kplist, 'num_keypoints': num_kp})

            pred_ann_list.append({
                'image_id': img_id,
                'pred_anns': pred_anns,
            })

        return results, pred_ann_list
