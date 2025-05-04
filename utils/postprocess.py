import torch
import numpy as np
import torch.nn.functional as F

class PosePostProcessor:
    def __init__(self, num_keypoints=17, heat_thresh=0.1, max_people=30):
        self.num_keypoints = num_keypoints
        self.heat_thresh = heat_thresh
        self.max_people = max_people

    def _nms(self, heatmap, kernel=3):
        # 非极大值抑制
        pad = (kernel - 1) // 2
        hmax = F.max_pool2d(heatmap, kernel, stride=1, padding=pad)
        keep = (hmax == heatmap).float()
        return heatmap * keep

    def __call__(self, heatmaps):
        """
        Args:
            heatmaps: [B, K, H, W] torch.Tensor
        Returns:
            List[List[np.ndarray]]: [batch][person][K,2]
        """
        if isinstance(heatmaps, torch.Tensor):
            heatmaps = heatmaps.detach().cpu()
        B, K, H, W = heatmaps.shape
        all_results = []
        for b in range(B):
            person_kps = []
            for k in range(K):
                hm = heatmaps[b, k:k+1]
                nms_hm = self._nms(hm)
                yx = torch.nonzero(nms_hm[0] > self.heat_thresh, as_tuple=False)
                scores = nms_hm[0][yx[:,0], yx[:,1]]
                # 只取分数最高的max_people个
                if yx.shape[0] > self.max_people:
                    topk = torch.topk(scores, self.max_people)
                    yx = yx[topk.indices]
                # 记录所有候选点
                for pt in yx:
                    x, y = pt[1].item(), pt[0].item()
                    person_kps.append((k, x, y))
            # 简化：将同一batch下所有关键点按顺序分组为一个人
            # 实际项目中应结合PAF或嵌入向量分组
            if person_kps:
                # [K,2]，每个关键点一个[x, y]，无则-1
                kps_array = np.full((self.num_keypoints, 2), -1, dtype=np.float32)
                for k, x, y in person_kps:
                    kps_array[k] = [x, y]
                all_results.append([kps_array])
            else:
                all_results.append([])
        return all_results
