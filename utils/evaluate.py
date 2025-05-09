import os
import torch
import torch.nn.functional as F
import numpy as np
import random
import cv2
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval
import wandb
from utils.coco import COCO_PERSON_SKELETON
from utils.visualization import visualize_coco_keypoints

def evaluate(model, val_loader, device, vis_ids=None):
    """
    在 COCO 验证集上评估模型，返回 mean AP、AP50 以及用于 W&B 的可视化图像列表。
    已优化：峰值检测和 PAF 骨架连接均在 GPU 上加速。
    """
    model.eval()
    coco_gt = val_loader.dataset.coco
    results = []
    img_ids = []
    vis_list = []

    # 可视化图像ID集合
    if vis_ids is not None:
        vis_set = set(vis_ids) & set(val_loader.dataset.img_ids)
    else:
        n_vis = getattr(wandb.config, 'n_vis', 3) if wandb.run is not None else 3
        vis_set = set(random.sample(val_loader.dataset.img_ids,
                                    min(n_vis, len(val_loader.dataset.img_ids))))

    idx_offset = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Evaluating", unit="batch")):
            imgs, _, _, _, _, joint_coords, n_person = batch
            imgs = imgs.to(device)

            # 模型预测
            hm_refine, paf_refine, hm_init, paf_init = model(imgs)
            heatmaps_pred = hm_refine  # [B, K, H, W]
            pafs_pred = paf_refine     # [B, 2*C, H, W]

            B, K, H, W = heatmaps_pred.shape

            # GPU 峰值检测 (3x3 max-pool)
            pooled = F.max_pool2d(heatmaps_pred, kernel_size=3, stride=1, padding=1)
            peak_mask = (heatmaps_pred == pooled) & (heatmaps_pred > 0.1)

            for i in range(B):
                img_id = val_loader.dataset.img_ids[idx_offset + i]
                img_ids.append(img_id)

                # 提取当前图像的峰值
                single_mask = peak_mask[i]      # [K, H, W]
                single_heat = heatmaps_pred[i]  # [K, H, W]
                all_peaks = []
                for k in range(K):
                    coords = torch.nonzero(single_mask[k], as_tuple=False)  # [N,2] (y,x)
                    if coords.numel() == 0:
                        all_peaks.append([])
                        continue
                    scores = single_heat[k][coords[:,0], coords[:,1]]
                    # 将 y,x 转为 x,y 并结合分数
                    xy = coords[:, [1,0]].float()
                    peaks_k = torch.cat([xy, scores.unsqueeze(1)], dim=1)  # [N,3]
                    all_peaks.append(peaks_k.cpu().tolist())

                # GPU 加速 PAF 连接匹配
                paf_tensor = pafs_pred[i]  # [2*C, H, W]
                connection_candidates = {c: [] for c in range(len(COCO_PERSON_SKELETON))}
                for c, (kp_a, kp_b) in enumerate(COCO_PERSON_SKELETON):
                    candA = all_peaks[kp_a]
                    candB = all_peaks[kp_b]
                    if not candA or not candB:
                        continue
                    ptsA = torch.tensor(candA, device=device)  # [NA,3]
                    ptsB = torch.tensor(candB, device=device)  # [NB,3]
                    posA = ptsA[:, :2]  # [NA,2]
                    posB = ptsB[:, :2]  # [NB,2]
                    scoreA = ptsA[:, 2]
                    scoreB = ptsB[:, 2]

                    # 计算向量与方向
                    diff = posB.unsqueeze(0) - posA.unsqueeze(1)  # [NA,NB,2]
                    norm = torch.norm(diff, dim=2, keepdim=True)  # [NA,NB,1]
                    dir_vec = diff / (norm + 1e-8)                # [NA,NB,2]

                    # 沿连线均匀采样 S 点
                    S = 10
                    t_vals = torch.linspace(0, 1, S, device=device).view(1,1,S,1)
                    sample_pts = posA.unsqueeze(1).unsqueeze(2) + diff.unsqueeze(2) * t_vals  # [NA,NB,S,2]
                    sample_xy = sample_pts.round().long()  # [NA,NB,S,2]
                    xs = sample_xy[:,:,:,0].clamp(0, W-1)
                    ys = sample_xy[:,:,:,1].clamp(0, H-1)

                    # 获取 PAF 向量
                    paf_x = paf_tensor[2*c].view(-1)
                    paf_y = paf_tensor[2*c+1].view(-1)
                    idx_flat = ys * W + xs  # [NA,NB,S]
                    paf_vals_x = paf_x[idx_flat]
                    paf_vals_y = paf_y[idx_flat]

                    # 计算点积
                    vx = dir_vec[:,:,0].unsqueeze(2)  # [NA,NB,1]
                    vy = dir_vec[:,:,1].unsqueeze(2)
                    dot = paf_vals_x * vx + paf_vals_y * vy  # [NA,NB,S]

                    avg_dot = dot.mean(dim=2)  # [NA,NB]
                    pass_ratio = (dot > 0.05).float().mean(dim=2)  # [NA,NB]

                    # 筛选有效连接
                    valid = (avg_dot > 0) & (pass_ratio >= 0.8)
                    idx_pairs = valid.nonzero(as_tuple=False)  # [M,2]
                    for ia_ib in idx_pairs:
                        ia, ib = int(ia_ib[0]), int(ia_ib[1])
                        total_score = float(avg_dot[ia,ib].item() + 0.5 * (scoreA[ia] + scoreB[ib]))
                        connection_candidates[c].append((ia, ib, total_score))

                    # 贪心选择非冲突连接
                    connection_candidates[c].sort(key=lambda x: x[2], reverse=True)
                    usedA, usedB = set(), set()
                    conns = []
                    for ia, ib, sc in connection_candidates[c]:
                        if ia not in usedA and ib not in usedB:
                            usedA.add(ia); usedB.add(ib)
                            conns.append((ia, ib))
                    connection_candidates[c] = conns

                # 组装多人骨架
                persons = []
                # 合并连接
                for c, (kp_a, kp_b) in enumerate(COCO_PERSON_SKELETON):
                    for ia, ib in connection_candidates[c]:
                        added = False
                        for person in persons:
                            if (kp_a in person and person[kp_a] == ia) or \
                               (kp_b in person and person[kp_b] == ib):
                                person[kp_a] = ia
                                person[kp_b] = ib
                                added = True
                                break
                        if not added:
                            persons.append({kp_a: ia, kp_b: ib})
                # 孤立点
                for k, peaks in enumerate(all_peaks):
                    for pi in range(len(peaks)):
                        if not any(person.get(k) == pi for person in persons):
                            persons.append({k: pi})

                # 转换为 COCO 结果格式
                orig_info = coco_gt.loadImgs([img_id])[0]
                orig_h, orig_w = orig_info['height'], orig_info['width']
                for person in persons:
                    kps, scores = [], []
                    for k in range(K):
                        if k in person:
                            x, y, sc = all_peaks[k][person[k]]
                            x = x * (orig_w / W)
                            y = y * (orig_h / H)
                            v = 2
                        else:
                            x, y, sc, v = 0.0, 0.0, 0.0, 0
                        kps.extend([x, y, v])
                        scores.append(sc)
                    res_score = float(np.mean([s for s in scores if s > 0] or [0]))
                    results.append({
                        'image_id': img_id,
                        'category_id': 1,
                        'keypoints': kps,
                        'score': res_score
                    })
                # 5. 可视化：将GT和预测关键点骨架叠加绘制在图像上
                if img_id in vis_set:
                    img_info = orig_info
                    img_path = os.path.join(val_loader.dataset.img_dir, img_info['file_name'])
                    orig_img = cv2.imread(img_path)
                    if orig_img is None:
                        orig_img = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
                    # 将原图缩放到网络输入大小用于显示
                    inp_h, inp_w = val_loader.dataset.img_size  # 网络输入尺寸 (h, w)
                    # 绘制GT关键点（绿色）
                    gt_ann_ids = coco_gt.getAnnIds(imgIds=[img_id], catIds=[1], iscrowd=None)
                    gt_anns = coco_gt.loadAnns(gt_ann_ids)
                    vis_img = visualize_coco_keypoints(orig_img, gt_anns, COCO_PERSON_SKELETON,
                                                       output_size=val_loader.dataset.img_size,
                                                       point_color=(0, 255, 0), line_color=(0, 255, 0))
                    # 绘制预测关键点（红色）
                    pred_anns = []
                    for person in persons:
                        kp_list = []
                        num_visible = 0
                        for k in range(len(all_peaks)):
                            if k in person:
                                x_pred, y_pred, score_val = all_peaks[k][person[k]]
                                # 转换到原图尺度再映射到显示尺寸
                                x_disp = x_pred * (orig_w / W) * (inp_w / orig_w)
                                y_disp = y_pred * (orig_h / H) * (inp_h / orig_h)
                                v = 2 if score_val > 0 else 0
                                num_visible += (1 if v == 2 else 0)
                            else:
                                x_disp, y_disp, v = 0.0, 0.0, 0
                            kp_list.extend([float(x_disp), float(y_disp), float(v)])
                        pred_anns.append({'keypoints': kp_list, 'num_keypoints': num_visible})
                    vis_img = visualize_coco_keypoints(
                        vis_img, pred_anns, COCO_PERSON_SKELETON,
                        output_size=(inp_h, inp_w),
                        point_color=(0, 0, 255), line_color=(0, 0, 255)
                    )

                    # BGR 转 RGB 以便构造 WandB 图像
                    vis_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
                    vis_list.append(wandb.Image(vis_rgb, caption=f"Image {img_id} – GT (green) vs Pred (red)"))
            idx_offset += B

    # COCO 评估
    if len(results) == 0:
        return 0.0, 0.0, vis_list
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='keypoints')
    coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    mean_ap = coco_eval.stats[0]
    ap50 = coco_eval.stats[1]

    model.train()
    return mean_ap, ap50, vis_list