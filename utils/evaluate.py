import os
import random
import numpy as np
import torch
import cv2
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval
import wandb
from utils.visualization import visualize_coco_keypoints
from .coco import COCO_PERSON_SKELETON

def evaluate(model, val_loader, device, vis_ids=None):
    """
    在COCO验证集上评估模型，返回 mean AP、AP50（或准确率）以及用于W&B的可视化图像列表。
    参数:
        model: 已训练的姿态模型 (eval模式)。
        val_loader: 验证集DataLoader。
        device: 运行设备 (cuda 或 cpu)。
        vis_ids: （可选）指定要可视化的图像ID列表。如果提供，则仅对这些ID生成可视化结果。
                 若为None，则随机选取 config.n_vis 张图用于可视化。
    返回:
        mean_ap: 平均AP (OKS=0.50:0.95)
        ap50: AP@0.5 或准确率
        vis_list: 包含wandb.Image的列表，可用于日志可视化。
    """
    model.eval()
    coco_gt = val_loader.dataset.coco  # COCO真值
    results = []
    img_ids = []
    vis_list = []

    # 确定要可视化的图片IDs集合
    if vis_ids is not None:
        # 只使用用户指定的vis_ids与当前val数据集的交集
        vis_set = set(vis_ids) & set(val_loader.dataset.img_ids)
        n_vis = len(vis_set)
    else:
        # 从配置中获取n_vis数量（若未设置则默认3）
        n_vis = getattr(wandb.config, 'n_vis', 3) if wandb.run is not None else 3
        # 随机抽取n_vis张不同的图用于可视化
        vis_set = set(random.sample(val_loader.dataset.img_ids, min(n_vis, len(val_loader.dataset.img_ids))))
    # 遍历验证数据集
    idx_offset = 0
    with torch.no_grad():
        for batch_idx, (imgs, _, _) in enumerate(tqdm(val_loader, desc="Evaluating", unit="batch", leave=False, total=len(val_loader))):
            imgs = imgs.to(device)
            # 模型预测
            output = model(imgs)
            # 兼容不同输出格式（支持精细化输出）
            heatmaps_pred = output[0]  # 精细化heatmap
            pafs_pred = output[1]      # 精细化PAF

            B = heatmaps_pred.shape[0]  # 当前batch图片数
            H, W = heatmaps_pred.shape[2], heatmaps_pred.shape[3]
            # 遍历batch内每张图片
            for i in range(B):
                img_id = val_loader.dataset.img_ids[idx_offset + i]
                img_ids.append(img_id)
                # 提取当前图片的预测heatmap和PAF到CPU numpy
                heat = heatmaps_pred[i].cpu().numpy()           # shape: [K, H, W]
                paf = pafs_pred[i].cpu().numpy() if pafs_pred is not None else None  # shape: [2*L, H, W] 或 None

                # 1. 从heatmap提取峰值坐标并精细化（亚像素修正）
                all_peaks = []  # 每个关键点类型的所有峰值
                peak_thresh = 0.1  # 热力图峰值阈值
                for k in range(heat.shape[0]):
                    hmap = heat[k]
                    # 使用3x3最大滤波找局部极大值
                    max_map = (hmap == cv2.dilate(hmap, np.ones((3,3), np.uint8)))  # True/False矩阵
                    peak_mask = (hmap > peak_thresh) & max_map
                    # 获取峰值坐标
                    coords = np.argwhere(peak_mask)  # [[y, x], ...]
                    refined_peaks = []
                    for (y, x) in coords:
                        score_val = float(hmap[y, x])
                        # 亚像素精细化：使用二阶泰勒展开近似峰值的偏移
                        dx = dy = 0.0
                        if 0 < x < W-1 and 0 < y < H-1:
                            # 计算偏导数
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

                # 2. 基于PAF进行关键点配对，连接肢体
                connection_candidates = {c: [] for c in range(len(COCO_PERSON_SKELETON))}
                if paf is not None:
                    # 遍历每根肢体连接
                    for c, (kp_a, kp_b) in enumerate(COCO_PERSON_SKELETON):
                        candA = all_peaks[kp_a]
                        candB = all_peaks[kp_b]
                        if not candA or not candB:
                            continue  # 若某一端无峰值则跳过
                        # 取出对应PAF的两个通道（x和y方向）
                        paf_x = paf[2*c]
                        paf_y = paf[2*c + 1]
                        for ia, (ax, ay, score_a) in enumerate(candA):
                            for ib, (bx, by, score_b) in enumerate(candB):
                                # 计算从A到B的向量
                                dx, dy = bx - ax, by - ay
                                norm = np.linalg.norm([dx, dy]) + 1e-8
                                vx, vy = dx / norm, dy / norm
                                # 沿A->B连线均匀采样10个点，计算PAF向量与连线方向的点积
                                xs = np.linspace(ax, bx, num=10).astype(np.int32)
                                ys = np.linspace(ay, by, num=10).astype(np.int32)
                                paf_scores = []
                                for (px, py) in zip(xs, ys):
                                    if 0 <= px < W and 0 <= py < H:
                                        paf_vec = np.array([paf_x[py, px], paf_y[py, px]])
                                        paf_scores.append(paf_vec.dot([vx, vy]))
                                if len(paf_scores) == 0:
                                    continue
                                # 计算PAF匹配得分：平均PAF点积，要求至少80%采样点的点积为正且平均值为正
                                avg_paf_score = float(np.mean(paf_scores))
                                if avg_paf_score > 0 and (np.mean(np.array(paf_scores) > 0.05) > 0.8):
                                    total_score = avg_paf_score + 0.5 * (score_a + score_b)
                                    connection_candidates[c].append((ia, ib, total_score))
                        # 对每根肢体，在候选连接中按得分降序选取不冲突的连接对（贪心法）
                        connection_candidates[c].sort(key=lambda x: x[2], reverse=True)
                        usedA, usedB = set(), set()
                        conns = []
                        for ia, ib, score in connection_candidates[c]:
                            if ia not in usedA and ib not in usedB:
                                conns.append((ia, ib))
                                usedA.add(ia)
                                usedB.add(ib)
                        connection_candidates[c] = conns

                # 3. 组装多人骨架：根据得到的连接将关键点组合成若干人
                persons = []
                # 先根据connection_candidates把互相关联的关键点合并成person实例
                for c, (kp_a, kp_b) in enumerate(COCO_PERSON_SKELETON):
                    for (ia, ib) in connection_candidates[c]:
                        placed = False
                        for person in persons:
                            # 如果当前person已经有此连接的一端，则将另一端加入
                            if (kp_a in person and person[kp_a] == ia) or (kp_b in person and person[kp_b] == ib):
                                person[kp_a] = ia
                                person[kp_b] = ib
                                placed = True
                                break
                        if not placed:
                            # 新建一个person包含这条连接
                            persons.append({kp_a: ia, kp_b: ib})
                # 将仍未连接到任何person的孤立关键点，各自作为一个person
                for k, peaks in enumerate(all_peaks):
                    for idx_p in range(len(peaks)):
                        # 如果该峰值未被任何已有person使用，则新建person，仅含此关键点
                        if not any(person.get(k) == idx_p for person in persons):
                            persons.append({k: idx_p})

                # 4. 转换结果为COCO评估格式
                orig_img_info = coco_gt.loadImgs([img_id])[0]
                orig_h, orig_w = orig_img_info['height'], orig_img_info['width']
                for person in persons:
                    # person是一个字典: 关键点索引 -> 对应all_peaks列表中的索引
                    kps = []
                    valid_scores = []
                    for k in range(len(all_peaks)):
                        if k in person:
                            x_pred, y_pred, score_val = all_peaks[k][person[k]]
                            # 将坐标从特征图尺度映射回原图尺度
                            x_pred *= (orig_w / W)
                            y_pred *= (orig_h / H)
                            v = 2  # 可见标志v=2表示该关键点有预测
                        else:
                            x_pred, y_pred, score_val = 0.0, 0.0, 0.0
                            v = 0  # v=0表示关键点没有预测到
                        kps.extend([float(x_pred), float(y_pred), float(v)])
                        valid_scores.append(score_val)
                    # 计算person整体得分：采用所有存在关键点的平均分
                    person_score = float(np.mean([s for s in valid_scores if s > 0] or [0]))
                    results.append({
                        "image_id": img_id,
                        "category_id": 1,  # 人类类别ID固定为1（COCO）
                        "keypoints": kps,  # 长度51的列表 [x1,y1,v1,...,x17,y17,v17]
                        "score": person_score
                    })

                # 5. 可视化：叠加绘制GT和预测关键点骨架
                if img_id in vis_set:
                    img_info = orig_img_info
                    img_path = os.path.join(val_loader.dataset.root, val_loader.dataset.img_folder, img_info['file_name'])
                    orig_img = cv2.imread(img_path)
                    if orig_img is None:
                        # 如果无法读取图像，用空图像代替
                        orig_img = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
                    # Resize原图用于可视化（与训练输入尺寸一致）
                    inp_h, inp_w = val_loader.dataset.img_size  # 训练输入网络的尺寸 (h, w)
                    vis_img = cv2.resize(orig_img, (inp_w, inp_h))
                    # 绘制GT关键点（绿色）
                    gt_ann_ids = coco_gt.getAnnIds(imgIds=[img_id], catIds=[1], iscrowd=None)
                    gt_anns = coco_gt.loadAnns(gt_ann_ids)
                    vis_img = visualize_coco_keypoints(vis_img, gt_anns, COCO_PERSON_SKELETON,
                                                       resize=val_loader.dataset.img_size,
                                                       kp_color=(0, 255, 0), limb_color=(0, 255, 0))
                    # 绘制Pred关键点（红色）
                    pred_anns = []
                    for person in persons:
                        kp_list = []
                        num_visible = 0
                        for k in range(len(all_peaks)):
                            if k in person:
                                x_pred, y_pred, score_val = all_peaks[k][person[k]]
                                # 转换到原图尺度后再resize到显示尺寸
                                x_disp = x_pred * (orig_w / W) * (inp_w / orig_w)
                                y_disp = y_pred * (orig_h / H) * (inp_h / orig_h)
                                v = 2 if score_val > 0 else 0
                                num_visible += (1 if v == 2 else 0)
                            else:
                                x_disp, y_disp, v = 0.0, 0.0, 0
                            kp_list.extend([x_disp, y_disp, v])
                        pred_anns.append({'keypoints': kp_list, 'num_keypoints': num_visible})
                    vis_img = visualize_coco_keypoints(vis_img, pred_anns, COCO_PERSON_SKELETON,
                                                       resize=val_loader.dataset.img_size,
                                                       kp_color=(0, 0, 255), limb_color=(0, 0, 255))
                    # 转换BGR->RGB以构造WandB Image
                    vis_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
                    vis_list.append(
                        wandb.Image(vis_rgb, caption=f"Image {img_id} – GT (green) vs Pred (red)")
                    )
            idx_offset += B

    # 6. 计算COCO指标（如results非空）
    if len(results) == 0:
        # 若无检测结果，返回0指标
        return 0.0, 0.0, vis_list
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='keypoints')
    coco_eval.params.imgIds = img_ids  # 仅评价处理过的图片
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    mean_ap = coco_eval.stats[0]   # 平均AP (OKS 0.50:0.95)
    ap50 = coco_eval.stats[1]      # AP@0.5 (OKS=0.50)
    return mean_ap, ap50, vis_list
