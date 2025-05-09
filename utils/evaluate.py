import os
import cv2
import numpy as np
import random
import torch
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval
import wandb
from utils.coco import COCO_PERSON_SKELETON
from utils.visualization import visualize_coco_keypoints

def evaluate(model, val_loader, device, vis_ids=None):
    """
    在COCO验证集上评估模型，返回 mean AP、AP50 以及用于W&B的可视化图像列表。
    参数:
      model: 已训练的姿态模型 (建议传入 eval 模式的模型)。
      val_loader: 验证集 DataLoader。
      device: 运行设备 ('cuda' 或 'cpu')。
      vis_ids: （可选）要可视化的图像ID列表。如果提供，则仅对这些ID生成可视化结果；
               若为 None，则随机选择 config.n_vis 张图用于可视化。
    返回:
      mean_ap: 平均AP (OKS 0.50:0.95)
      ap50: AP@0.5 (OKS=0.50)
      vis_list: 包含 wandb.Image 的列表，可用于日志可视化。
    """
    model.eval()
    coco_gt = val_loader.dataset.coco  # COCO真值数据
    results = []
    img_ids = []
    vis_list = []
    # 确定需要可视化的图像ID集合
    if vis_ids is not None:
        # 取用户指定ID与当前验证集的交集
        vis_set = set(vis_ids) & set(val_loader.dataset.img_ids)
        n_vis = len(vis_set)
    else:
        # 未指定则从配置中获取n_vis数量（默认为3）
        n_vis = getattr(wandb.config, 'n_vis', 3) if wandb.run is not None else 3
        # 随机抽取 n_vis 张不同的图像用于可视化
        vis_set = set(random.sample(val_loader.dataset.img_ids, min(n_vis, len(val_loader.dataset.img_ids))))
    idx_offset = 0
    with torch.no_grad():
        for batch_idx, (imgs, _, _, _) in enumerate(tqdm(val_loader, desc="Evaluating", unit="batch")):
            imgs = imgs.to(device)
            # 模型预测
            #   兼容不同输出格式（支持精细化输出）
            hm_refine, paf_refine, hm_init, paf_init = model(imgs)
            heatmaps_pred, pafs_pred = hm_refine, paf_refine

            B = heatmaps_pred.shape[0]  # batch中图片数
            H, W = heatmaps_pred.shape[2], heatmaps_pred.shape[3]
            # 遍历 batch 内每张图片
            for i in range(B):
                img_id = val_loader.dataset.img_ids[idx_offset + i]
                img_ids.append(img_id)
                # 提取当前图片的预测 heatmap 和 PAF 并转为 numpy
                heat = heatmaps_pred[i].cpu().numpy()  # [K, H, W]
                paf = pafs_pred[i].cpu().numpy() if pafs_pred is not None else None  # [2*C, H, W] 或 None
                # 1. 从 heatmap 提取关键点峰值坐标（含亚像素精细化）
                all_peaks = []  # 每个关键点类型的峰值列表
                peak_thresh = 0.1  # 峰值阈值
                for k in range(heat.shape[0]):
                    hmap = heat[k]
                    # 使用3x3膨胀找到局部最大值
                    max_map = (hmap == cv2.dilate(hmap, np.ones((3,3), np.uint8)))
                    peak_mask = (hmap > peak_thresh) & max_map
                    # 获取峰值坐标
                    coords = np.argwhere(peak_mask)  # [[y, x], ...]
                    refined_peaks = []
                    for (y, x) in coords:
                        score_val = float(hmap[y, x])
                        # 若峰值不在边界，则进行亚像素偏移修正（二阶插值）
                        dx = dy = 0.0
                        if 0 < x < W-1 and 0 < y < H-1:
                            # 计算梯度
                            ddx = 0.5 * (hmap[y, x+1] - hmap[y, x-1])
                            ddy = 0.5 * (hmap[y+1, x] - hmap[y-1, x])
                            ddxx = hmap[y, x+1] + hmap[y, x-1] - 2 * hmap[y, x]
                            ddyy = hmap[y+1, x] + hmap[y-1, x] - 2 * hmap[y, x]
                            # 近似峰值偏移 (仅在二阶导数不为零时)
                            if ddxx != 0:
                                dx = -ddx / (ddxx + 1e-6)
                            if ddyy != 0:
                                dy = -ddy / (ddyy + 1e-6)
                        refined_peaks.append((float(x) + dx, float(y) + dy, score_val))
                    all_peaks.append(refined_peaks)
                # 2. 基于 PAF 将关键点配对成肢体连接
                connection_candidates = {c: [] for c in range(len(COCO_PERSON_SKELETON))}
                if paf is not None:
                    # 遍历每根骨骼连接定义
                    for c, (kp_a, kp_b) in enumerate(COCO_PERSON_SKELETON):
                        candA = all_peaks[kp_a]
                        candB = all_peaks[kp_b]
                        if not candA or not candB:
                            continue  # 某一端无检测到峰值
                        # 取出对应PAF的两个通道（x和y方向）
                        paf_x = paf[2*c]
                        paf_y = paf[2*c + 1]
                        for ia, (ax, ay, score_a) in enumerate(candA):
                            for ib, (bx, by, score_b) in enumerate(candB):
                                # 计算从A到B的向量
                                dx, dy = bx - ax, by - ay
                                norm = np.linalg.norm([dx, dy]) + 1e-8
                                vx, vy = dx / norm, dy / norm
                                # 沿A->B连线均匀采样10个点
                                xs = np.linspace(ax, bx, num=10).astype(np.int32)
                                ys = np.linspace(ay, by, num=10).astype(np.int32)
                                paf_scores = []
                                for (px, py) in zip(xs, ys):
                                    if 0 <= px < W and 0 <= py < H:
                                        paf_vec = np.array([paf_x[py, px], paf_y[py, px]])
                                        paf_scores.append(paf_vec.dot([vx, vy]))
                                if len(paf_scores) == 0:
                                    continue
                                # 计算PAF匹配得分：要求至少80%采样点的点积为正且平均值为正
                                avg_paf_score = float(np.mean(paf_scores))
                                if avg_paf_score > 0 and (np.mean(np.array(paf_scores) > 0.05) >= 0.8):
                                    total_score = avg_paf_score + 0.5 * (score_a + score_b)
                                    connection_candidates[c].append((ia, ib, total_score))
                        # 按得分从高到低排序候选连接
                        connection_candidates[c].sort(key=lambda x: x[2], reverse=True)
                        # 贪心选择非冲突连接对
                        usedA, usedB = set(), set()
                        conns = []
                        for ia, ib, score in connection_candidates[c]:
                            if ia not in usedA and ib not in usedB:
                                usedA.add(ia)
                                usedB.add(ib)
                                conns.append((ia, ib))
                        connection_candidates[c] = conns
                # 3. 组装多人骨架：根据连接结果将关键点合并成若干person实例
                persons = []
                # 先根据 connection_candidates 合并成初步 person 列表
                for c, (kp_a, kp_b) in enumerate(COCO_PERSON_SKELETON):
                    for (ia, ib) in connection_candidates[c]:
                        placed = False
                        for person in persons:
                            # 如果当前 person 已经有此连接的一端，则将另一端加入
                            if (kp_a in person and person[kp_a] == ia) or (kp_b in person and person[kp_b] == ib):
                                person[kp_a] = ia
                                person[kp_b] = ib
                                placed = True
                                break
                        if not placed:
                            # 创建新 person
                            persons.append({kp_a: ia, kp_b: ib})
                # 将仍未连接到任何 person 的孤立关键点各自作为一个 person
                for k, peaks in enumerate(all_peaks):
                    for idx_p in range(len(peaks)):
                        # 若该峰值未被任何已有 person 使用，则新建 person，仅含此关键点
                        if not any(person.get(k) == idx_p for person in persons):
                            persons.append({k: idx_p})
                # 4. 转换每个 person 为 COCO 格式结果
                orig_img_info = coco_gt.loadImgs([img_id])[0]
                orig_h, orig_w = orig_img_info['height'], orig_img_info['width']
                for person in persons:
                    # person 是一个字典: 关键点类型 -> all_peaks 中的索引
                    kps = []
                    valid_scores = []
                    for k in range(len(all_peaks)):
                        if k in person:
                            x_pred, y_pred, score_val = all_peaks[k][person[k]]
                            # 将坐标从特征图尺度映射回原图尺度
                            x_pred *= (orig_w / W)
                            y_pred *= (orig_h / H)
                            v = 2  # 标记该关键点有预测
                        else:
                            x_pred, y_pred, score_val = 0.0, 0.0, 0.0
                            v = 0  # 未检测到关键点
                        kps.extend([float(x_pred), float(y_pred), float(v)])
                        valid_scores.append(score_val)
                    person_score = float(np.mean([s for s in valid_scores if s > 0] or [0]))
                    results.append({
                        "image_id": img_id,
                        "category_id": 1,  # COCO人类类别ID为1
                        "keypoints": kps,  # 长度51的关键点列表 [x1,y1,v1,...,x17,y17,v17]
                        "score": person_score
                    })
                # 5. 可视化：将GT和预测关键点骨架叠加绘制在图像上
                if img_id in vis_set:
                    img_info = orig_img_info
                    img_path = os.path.join(val_loader.dataset.img_dir, img_info['file_name'])
                    orig_img = cv2.imread(img_path)
                    if orig_img is None:
                        orig_img = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
                    # 将原图缩放到网络输入大小用于显示
                    inp_h, inp_w = val_loader.dataset.img_size  # 网络输入尺寸 (h, w)
                    vis_img = cv2.resize(orig_img, (inp_w, inp_h))
                    # 绘制GT关键点（绿色）
                    gt_ann_ids = coco_gt.getAnnIds(imgIds=[img_id], catIds=[1], iscrowd=None)
                    gt_anns = coco_gt.loadAnns(gt_ann_ids)
                    vis_img = visualize_coco_keypoints(vis_img, gt_anns, COCO_PERSON_SKELETON,
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
                    vis_img = visualize_coco_keypoints(vis_img, pred_anns, COCO_PERSON_SKELETON,
                                                       output_size=val_loader.dataset.img_size,
                                                       point_color=(0, 0, 255), line_color=(0, 0, 255))
                    # BGR 转 RGB 以便构造 WandB 图像
                    vis_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
                    vis_list.append(wandb.Image(vis_rgb, caption=f"Image {img_id} – GT (green) vs Pred (red)"))
            idx_offset += B
    # 6. 计算COCO评估指标（如果有检测结果）
    if len(results) == 0:
        # 若无任何检测结果，返回0
        return 0.0, 0.0, vis_list
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='keypoints')
    coco_eval.params.imgIds = img_ids  # 仅评价处理过的图像
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    mean_ap = coco_eval.stats[0]  # 平均AP (OKS 0.50:0.95)
    ap50 = coco_eval.stats[1]     # AP@0.5 (OKS=0.50)


    model.train()

    return mean_ap, ap50, vis_list
