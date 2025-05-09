import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.coco import COCO_PERSON_SKELETON  # 引入COCO骨骼连接定义

class PoseLossV2(nn.Module):
    """
    PoseLossV2: 同时计算Heatmap和PAF损失，并加入OHKM和骨架结构约束。
    适用于单阶段或含精细化输出的模型。
    """
    def __init__(self, ohkm_k=8, struct_weight=0.0):
        """
        参数:
          ohkm_k (int): 最难关键点数量K，用于在线困难关键点挖掘 (OHKM)。0或None表示不使用OHKM。
          struct_weight (float): 结构约束损失权重。>0时计算结构损失乘以该权重加入总损失。
        """
        super().__init__()
        self.ohkm_k = ohkm_k
        self.struct_weight = struct_weight  # 新增参数：用于控制是否启用结构损失

    def forward(self, outputs, targets):
        """
        计算总损失。
        outputs: 模型输出，可以是 Tensor 或 tuple/list。
           - 若为 tuple 长度4: (refined_heatmap, refined_paf, init_heatmap, init_paf)
           - 若为 tuple 长度2: (heatmap, paf)
           - 若为 Tensor: 仅 heatmap 输出（无PAF）
        targets: 真值 tuple，通常为 (heatmap_gt, paf_gt)，若包含关节坐标则为 (heatmap_gt, paf_gt, joint_coords_gt)。
        返回: total_loss 张量（标量）
        """
        # 解包真值 (兼容不传PAF或关节坐标的情况)
        if isinstance(targets, (tuple, list)):
            if len(targets) == 2:
                heatmap_gt, paf_gt = targets
                joint_coords_gt = None
            elif len(targets) == 3:
                heatmap_gt, paf_gt, joint_coords_gt = targets
            else:
                raise ValueError("targets 应为包含2或3个元素的tuple: (heatmap_gt, paf_gt, [joint_coords_gt])")
        else:
            # 若仅提供单个张量作为真值，则视为 heatmap GT，PAF和结构损失不适用（保持向后兼容）
            heatmap_gt = targets
            paf_gt = None
            joint_coords_gt = None

        # 按输出结构计算损失
        if isinstance(outputs, (tuple, list)):
            # 情况1: 模型有精细化阶段 (outputs长度为4)
            if len(outputs) == 4:
                refined_heatmap, refined_paf, init_heatmap, init_paf = outputs
                # 初始阶段损失：对初始 heatmap 和 PAF 使用普通 MSE
                loss_heat_init = F.mse_loss(init_heatmap, heatmap_gt, reduction='mean')
                loss_paf_init = F.mse_loss(init_paf, paf_gt, reduction='mean') if paf_gt is not None else 0.0

                # 精细化阶段损失：
                # **仅对精细化 heatmap 应用 OHKM**（若 ohkm_k>0），否则用全局 MSE
                if self.ohkm_k is not None and self.ohkm_k > 0:
                    # 计算每个关键点通道的平均平方误差 [B, K]
                    diff_sq = (refined_heatmap - heatmap_gt) ** 2         # 差平方 [B, K, H, W]
                    per_kp_loss = diff_sq.mean(dim=(2, 3))               # 每个关键点的平均误差 [B, K]
                    K = min(self.ohkm_k, per_kp_loss.shape[1])           # 选取的关键点数K
                    ohkm_loss_list = []
                    for i in range(per_kp_loss.size(0)):                # 遍历每张图片
                        kp_losses = per_kp_loss[i]
                        topk_vals, _ = torch.topk(kp_losses, k=K, largest=True)  # 选取最难的K个关键点损失
                        ohkm_loss_list.append(topk_vals.mean())
                    loss_heat_refine = torch.stack(ohkm_loss_list).mean()       # OHKM后的heatmap损失（批次平均）
                else:
                    loss_heat_refine = F.mse_loss(refined_heatmap, heatmap_gt, reduction='mean')  # 普通MSE

                # 精细化 PAF 损失：对PAF全像素平均 MSE（不使用OHKM）
                loss_paf_refine = F.mse_loss(refined_paf, paf_gt, reduction='mean') if paf_gt is not None else 0.0

                # 合并初始和精细化阶段的损失
                heatmap_loss = loss_heat_init + loss_heat_refine
                paf_loss = loss_paf_init + loss_paf_refine
                main_heatmap_pred = refined_heatmap  # 用精细化heatmap用于结构损失计算

            # 情况2: 模型无精细化输出 (outputs长度为2)
            elif len(outputs) == 2:
                heatmap_pred, paf_pred = outputs
                # Heatmap损失：如果启用了OHKM则应用，否则使用全局MSE
                if self.ohkm_k is not None and self.ohkm_k > 0:
                    diff_sq = (heatmap_pred - heatmap_gt) ** 2          # [B, K, H, W]
                    per_kp_loss = diff_sq.mean(dim=(2, 3))              # [B, K]
                    K = min(self.ohkm_k, per_kp_loss.shape[1])
                    ohkm_loss_list = []
                    for i in range(per_kp_loss.size(0)):
                        kp_losses = per_kp_loss[i]
                        topk_vals, _ = torch.topk(kp_losses, k=K, largest=True)
                        ohkm_loss_list.append(topk_vals.mean())
                    heatmap_loss = torch.stack(ohkm_loss_list).mean()   # OHKM后的heatmap损失
                else:
                    heatmap_loss = F.mse_loss(heatmap_pred, heatmap_gt, reduction='mean')
                # PAF损失：若模型有PAF输出且提供了GT则计算MSE，否则为0
                paf_loss = F.mse_loss(paf_pred, paf_gt, reduction='mean') if (paf_pred is not None and paf_gt is not None) else 0.0
                main_heatmap_pred = heatmap_pred  # 用模型输出heatmap用于结构损失

            else:
                raise ValueError(f"未预期的模型输出形式，outputs长度为{len(outputs)}")
        else:
            # 情况3: 模型输出为单个Tensor（仅heatmap）
            main_heatmap_pred = outputs
            heatmap_loss = F.mse_loss(main_heatmap_pred, heatmap_gt, reduction='mean')
            paf_loss = 0.0  # 无PAF输出则PAF损失为0

        # 基本总损失 = heatmap损失 + PAF损失
        total_loss = heatmap_loss + paf_loss

        # 如果启用了结构约束损失(struct_weight>0)，计算骨架结构损失并加权加入总损失
        if self.struct_weight is not None and self.struct_weight > 0.0:
            struct_loss = 0.0
            if joint_coords_gt is not None:
                struct_loss_batch = []
                B, K, H, W = main_heatmap_pred.shape
                # 预测关键点坐标：选取每个heatmap通道的峰值 (argmax近似关键点坐标)
                heat_flat = main_heatmap_pred.view(B, K, -1)             # 展平HW维度
                max_idx = torch.argmax(heat_flat, dim=2)                # [B, K] 每个关键点最大响应索引
                pred_coords = []
                for b in range(B):
                    coords_b = []
                    for k in range(K):
                        idx = int(max_idx[b, k].item())
                        y = idx // W
                        x = idx % W
                        coords_b.append((x, y))
                    pred_coords.append(coords_b)
                # 逐图片计算骨架结构损失（遍历骨骼连接对）
                for b in range(B):
                    coords_pred = pred_coords[b]
                    # joint_coords_gt[b] 假定为长度K的可迭代 (列表/元组)，每个元素为GT坐标(x,y)或None
                    coords_gt_b = joint_coords_gt[b]
                    loss_val = 0.0
                    count = 0
                    for (a, b_idx) in COCO_PERSON_SKELETON:  # 遍历COCO定义的骨骼连接
                        if a < K and b_idx < K and coords_gt_b[a] is not None and coords_gt_b[b_idx] is not None:
                            # 计算预测连杆长度与GT长度的差的平方
                            xa, ya = coords_pred[a]; xb, yb = coords_pred[b_idx]
                            xag, yag = coords_gt_b[a]; xbg, ybg = coords_gt_b[b_idx]
                            pred_len = ((xa - xb)**2 + (ya - yb)**2) ** 0.5
                            gt_len = ((xag - xbg)**2 + (yag - ybg)**2) ** 0.5
                            loss_val += (pred_len - gt_len) ** 2
                            count += 1
                    # 平均每条骨骼的长度差误差
                    avg_struct_err = loss_val / (count + 1e-6) if count > 0 else 0.0
                    struct_loss_batch.append(avg_struct_err)
                struct_loss = torch.tensor(struct_loss_batch, device=main_heatmap_pred.device).mean()
            total_loss = total_loss + self.struct_weight * struct_loss  # 将结构损失乘权重加入总损失

        return total_loss