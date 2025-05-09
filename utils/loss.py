import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.coco import COCO_PERSON_SKELETON  # 引入COCO骨骼连接定义列表

class PoseLoss(nn.Module):
    """
    PoseLossV2: 同时计算Heatmap和PAF的损失，并加入OHKM和结构约束。
    适用于单阶段或含精细化输出的模型。
    """
    def __init__(self, ohkm_k=8, struct_weight=0.0):
        """
        参数:
        ohkm_k (int): 在线困难关键点挖掘(OHKM)中每张图片选取的最难关键点数量。
                      若为0或None则不使用OHKM机制。
        struct_weight (float): 结构约束损失权重。如果>0，则会计算结构损失并乘以该权重加入总损失。
        """
        super().__init__()
        self.ohkm_k = ohkm_k
        self.struct_weight = struct_weight

    def forward(self, outputs, targets):
        """
        计算损失:
        outputs: 模型输出，可以是(torch.Tensor)或(tuple/list)。
         - 若为tuple长度4，视为(refined_heatmap, refined_paf, init_heatmap, init_paf)。
         - 若为tuple长度2，视为(heatmap, paf)。
         - 若为tensor，则仅代表heatmap输出，无PAF。
        targets: 包含真值的tuple，通常为(heatmap_gt, paf_gt)。
         如果需要计算结构损失且有真值关键点坐标，可扩展为(heatmap_gt, paf_gt, joint_coords_gt)。
        返回:
         total_loss: 总损失张量 (标量)
        """


        # 解包真值
        heatmap_gt = targets['heatmap']
        paf_gt = targets['paf']
        heatmap_weight = targets['heatmap_weight']
        paf_weight = targets['paf_weight']

        total_loss = 0.0
        # 模型输出可能为 tuple/list 或 tensor
        if isinstance(outputs, (tuple, list)):
            # 根据输出长度判断是否有精细化阶段
            if len(outputs) == 4:
                # refine=True 的情况，outputs = (refined_heatmap, refined_paf, init_heatmap, init_paf)
                refined_heatmap, refined_paf, init_heatmap, init_paf = outputs
                # 计算初始阶段的heatmap和paf损失
                diff_hm_i = (init_heatmap - heatmap_gt) ** 2
                loss_heat_init = (diff_hm_i * heatmap_weight).mean()
                diff_paf_i = (init_paf - paf_gt) ** 2
                loss_paf_init = (diff_paf_i * paf_weight).mean() if paf_gt is not None else 0.0
                # 计算精细化阶段的heatmap和paf损失
                # 若使用OHKM，则对精细化heatmap损失应用OHKM选择
                if self.ohkm_k is not None and self.ohkm_k > 0:
                    # 计算每个关键点通道的平均误差
                    # heatmap误差张量形状: [B, K, H, W]
                    diff_sq = (refined_heatmap - heatmap_gt) ** 2  # 平方误差
                    # 按空间求平均，得到每个关键点的MSE [B, K]
                    per_kp_loss = diff_sq.mean(dim=(2, 3))
                    # 针对每张图片选取 top-K 最大的关键点误差
                    K = min(self.ohkm_k, per_kp_loss.shape[1])
                    ohkm_loss_list = []
                    for i in range(per_kp_loss.size(0)):  # 遍历batch内每张图片
                        kp_losses = per_kp_loss[i]
                        topk_vals, _ = torch.topk(kp_losses, k=K, largest=True)
                        ohkm_loss_list.append(topk_vals.mean())
                    # 将每张图片的OHKM损失取平均作为总的heatmap损失
                    loss_heat_refine = torch.stack(ohkm_loss_list).mean()
                else:
                    # 不使用OHKM，直接对精细化heatmap计算全局平均MSE
                    diff_hm_r = (refined_heatmap - heatmap_gt) ** 2
                    loss_heat_refine = (diff_hm_r * heatmap_weight).mean()
                # PAF损失：对所有像素取平均
                diff_paf_r = (refined_paf - paf_gt) ** 2
                loss_paf_refine = (diff_paf_r * paf_weight).mean() if paf_gt is not None else 0.0
                # 合并heatmap损失和PAF损失（初始+精细化）
                weight_bias = 1.0
                if self.ohkm_k > 0:
                    weight_bias = 0.5
                heatmap_loss = loss_heat_init*weight_bias + loss_heat_refine
                paf_loss = loss_paf_init*weight_bias + loss_paf_refine
                # 选择用于结构损失的预测热图（使用精细化后的输出）
                main_heatmap_pred = refined_heatmap
            else:
                raise ValueError(f"未预期的模型输出长度: {len(outputs)}")
        else:
            # 输出为单个Tensor的情况（仅heatmap输出，没有PAF）
            main_heatmap_pred = outputs
            heatmap_loss = F.mse_loss(main_heatmap_pred, heatmap_gt, reduction='mean')
            paf_loss = 0.0

        # Heatmap损失和PAF损失之和作为基础总损失
        total_loss = heatmap_loss + paf_loss

        # 结构约束损失（如适用）
        if self.struct_weight is not None and self.struct_weight > 0.0:
            struct_loss = 0.0
            if joint_coords_gt is not None:
                # 计算预测每个关键点的坐标（使用heatmap的argmax近似）
                # main_heatmap_pred shape: [B, K, H, W]
                B, K, H, W = main_heatmap_pred.shape
                heat_flat = main_heatmap_pred.view(B, K, -1)  # 展平HW维度以便argmax
                max_indices = torch.argmax(heat_flat, dim=2)  # [B, K] 每个关键点通道的最大值索引
                pred_coords = []
                for b in range(B):
                    coords_b = []
                    for k in range(K):
                        idx = int(max_indices[b, k].item())
                        y = idx // W
                        x = idx % W
                        coords_b.append((x, y))
                    pred_coords.append(coords_b)
                struct_loss_list = []
                for b in range(B):
                    coords_pred = pred_coords[b]
                    # coords_gt假定为list/tuple长度K，每个元素为(x,y)或None（若该关键点无标注）
                    coords_gt = joint_coords_gt[b]
                    if coords_gt is not None and torch.is_tensor(coords_gt):
                        coords_gt = coords_gt.tolist()
                    loss_val = 0.0
                    count = 0
                    for (a, b_idx) in COCO_PERSON_SKELETON:  # 遍历定义的骨骼连接
                        if a < K and b_idx < K and coords_gt is not None and coords_gt[a] is not None and coords_gt[b_idx] is not None:
                            # 预测与GT的距离差
                            xa, ya = coords_pred[a]
                            xb, yb = coords_pred[b_idx]
                            xag, yag = coords_gt[a]
                            xbg, ybg = coords_gt[b_idx]
                            # 计算欧氏距离
                            pred_len = ((xa - xb) ** 2 + (ya - yb) ** 2) ** 0.5
                            gt_len = ((xag - xbg) ** 2 + (yag - ybg) ** 2) ** 0.5
                            # 累积骨骼长度差的平方
                            loss_val += (pred_len - gt_len) ** 2
                            count += 1
                    # 当前图片的平均结构损失
                    avg_struct_loss = loss_val / (count + 1e-6) if count > 0 else 0.0
                    struct_loss_list.append(avg_struct_loss)
                if struct_loss_list:
                    struct_loss = torch.tensor(struct_loss_list, device=main_heatmap_pred.device).mean()
            # 将结构损失加入总损失（乘以权重）
            total_loss = total_loss + self.struct_weight * struct_loss

        return total_loss
