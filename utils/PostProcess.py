import torch
import torch.nn as nn
import numpy as np
import cv2
from scipy.ndimage import maximum_filter
import wandb
from typing import List, Dict, Tuple, Optional, Any

import torch.nn.functional as F
from utils.coco import COCO_PERSON_SKELETON


def postprocess(heat: torch.Tensor, paf, H, W, img_id, data) -> tuple[list[Dict], list[Dict], list[Any], Any]:
    all_peaks = []

    results = []

    peak_thresh = 0.1
    for k in range(heat.shape[0]):
        hmap = heat[k]
        # 局部极大 + 阈值
        max_f = maximum_filter(hmap, size=3)
        mask = (hmap == max_f) & (hmap > peak_thresh)
        coords = np.argwhere(mask)  # [[y, x], ...]
        refined_peaks = []
        for (y, x) in coords:
            score = float(hmap[y, x])
            # Taylor expansion for subpixel 修正
            dx = dy = 0.0
            if 0 < x < W - 1 and 0 < y < H - 1:
                dx = 0.5 * (hmap[y, x + 1] - hmap[y, x - 1])
                dy = 0.5 * (hmap[y + 1, x] - hmap[y - 1, x])
                dxx = hmap[y, x + 1] + hmap[y, x - 1] - 2 * hmap[y, x]
                dyy = hmap[y + 1, x] + hmap[y - 1, x] - 2 * hmap[y, x]
                if dxx != 0: dx /= -dxx
                if dyy != 0: dy /= -dyy
            refined_peaks.append((float(x) + dx, float(y) + dy, score))
        all_peaks.append(refined_peaks)

    # 2. 构建 PAF 连接候选，并贪心匹配
    connection_candidates = {c: [] for c in range(len(COCO_PERSON_SKELETON))}
    if paf is not None:
        for c, (a, b) in enumerate(COCO_PERSON_SKELETON):
            candA = all_peaks[a]
            candB = all_peaks[b]
            if not candA or not candB:
                continue
            paf_x = paf[2 * c]
            paf_y = paf[2 * c + 1]
            for ia, (ax, ay, sa) in enumerate(candA):
                for ib, (bx, by, sb) in enumerate(candB):
                    dx, dy = bx - ax, by - ay
                    norm = np.hypot(dx, dy) + 1e-8
                    vx, vy = dx / norm, dy / norm
                    # 沿连线均匀采样
                    num_samples = 10
                    xs = np.linspace(ax, bx, num_samples).astype(int)
                    ys = np.linspace(ay, by, num_samples).astype(int)
                    scores = []
                    for xx, yy in zip(xs, ys):
                        if 0 <= xx < W and 0 <= yy < H:
                            vec = np.array([paf_x[yy, xx], paf_y[yy, xx]])
                            scores.append(vec.dot([vx, vy]))
                    if not scores:
                        continue
                    avg_score = float(np.mean(scores))
                    if avg_score > 0 and np.mean((np.array(scores) > 0.05).astype(float)) > 0.8:
                        total_score = avg_score + 0.5 * (sa + sb)
                        connection_candidates[c].append((ia, ib, total_score))
            # 贪心选取不冲突连接
            connection_candidates[c].sort(key=lambda x: x[2], reverse=True)
            usedA, usedB = set(), set()
            conns = []
            for ia, ib, _ in connection_candidates[c]:
                if ia not in usedA and ib not in usedB:
                    conns.append((ia, ib))
                    usedA.add(ia);
                    usedB.add(ib)
            connection_candidates[c] = conns

    # 3. 组装多人骨架实例
    persons = []
    for c, (a, b) in enumerate(COCO_PERSON_SKELETON):
        for ia, ib in connection_candidates.get(c, []):
            placed = False
            for p in persons:
                if (a in p and p[a] == ia) or (b in p and p[b] == ib):
                    p[a] = ia;
                    p[b] = ib;
                    placed = True;
                    break
            if not placed:
                persons.append({a: ia, b: ib})
    # 孤立关键点也作为单实例
    for k, peaks in enumerate(all_peaks):
        for idx_p in range(len(peaks)):
            if not any(p.get(k) == idx_p for p in persons):
                persons.append({k: idx_p})

    # 4. 转为 COCO 评估格式
    orig = data.loadImgs([img_id])[0]
    for p in persons:
        kps_flat = []
        scores = []
        for k in range(len(all_peaks)):
            if k in p:
                x, y, sc = all_peaks[k][p[k]]
                x *= (orig['width'] / W)
                y *= (orig['height'] / H)
            else:
                x, y, sc = 0.0, 0.0, 0.0
            kps_flat.extend([float(x), float(y), float(sc)])
            scores.append(sc)
        results.append({
            'image_id': img_id,
            'category_id': 1,
            'keypoints': kps_flat,
            'score': float(np.mean(scores))
        })

    return persons, results, all_peaks, orig