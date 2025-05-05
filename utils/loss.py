import torch
import torch.nn as nn

class PoseLoss(nn.Module):
    """姿态估计损失 (Heatmap和PAF回归 + 在线困难关键点挖掘)"""
    def __init__(self, ohkm_k=8):
        super().__init__()
        self.ohkm_k = ohkm_k
        self.mse = nn.MSELoss(reduction='mean')
    def forward(self, pred, target):
        heat_target, paf_target = target
        # 多输出处理
        if isinstance(pred, (tuple, list)):
            if len(pred) == 2:
                # 单阶段输出
                heat_pred, paf_pred = pred
                # Heatmap损失 + OHKM
                N, K, H, W = heat_pred.shape
                heat_loss_all = self.mse(heat_pred, heat_target)
                per_kp_mse = ((heat_pred - heat_target) ** 2).view(N, K, -1).mean(dim=2)
                topk_val, _ = torch.topk(per_kp_mse, k=self.ohkm_k, dim=1, largest=True)
                ohkm_loss = topk_val.mean()
                heat_loss = heat_loss_all + ohkm_loss
                # PAF损失
                paf_loss = self.mse(paf_pred, paf_target)
                return heat_loss + paf_loss
            elif len(pred) == 4:
                # 精细化输出
                refined_heat, refined_paf, init_heat, init_paf = pred
                # 初始预测损失
                init_loss = self.mse(init_heat, heat_target) + self.mse(init_paf, paf_target)
                # 精细化预测损失 + OHKM
                N, K, H, W = refined_heat.shape
                heat_loss_all = self.mse(refined_heat, heat_target)
                per_kp_mse = ((refined_heat - heat_target) ** 2).view(N, K, -1).mean(dim=2)
                topk_val, _ = torch.topk(per_kp_mse, k=self.ohkm_k, dim=1, largest=True)
                ohkm_loss = topk_val.mean()
                refined_heat_loss = heat_loss_all + ohkm_loss
                refined_paf_loss = self.mse(refined_paf, paf_target)
                return init_loss + refined_heat_loss + refined_paf_loss
            else:
                raise ValueError(f"PoseLoss: unrecognized pred output length {len(pred)}; expected 2 or 4")

        else:
            # 单一输出 (仅热图)
            heat_pred = pred
            N, K, H, W = heat_pred.shape
            heat_loss_all = self.mse(heat_pred, heat_target)
            per_kp_mse = ((heat_pred - heat_target) ** 2).view(N, K, -1).mean(dim=2)
            topk_val, _ = torch.topk(per_kp_mse, k=self.ohkm_k, dim=1, largest=True)
            ohkm_loss = topk_val.mean()
            return heat_loss_all + ohkm_loss