import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

def compute_miou(model, loader, device, threshold=0.5):
    """
    Calculate mean IoU for binary segmentation.
    Args:
        model: model that returns seg_logits as first output
        loader: DataLoader yielding (imgs, masks, ...)
        device: torch.device
        threshold: float, binarization threshold
    Returns:
        float: mean IoU
    """
    model.eval()
    ious = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="MIoU", leave=False):
            imgs, masks = batch[0].to(device), batch[1].to(device)
            out = model(imgs)
            seg_logits = out[0] if isinstance(out, tuple) else out
            probs = torch.sigmoid(seg_logits)
            preds = (probs > threshold).float()

            preds_flat = preds.view(-1)
            masks_flat = masks.view(-1)
            inter = (preds_flat * masks_flat).sum().item()
            union = ((preds_flat + masks_flat) > 0).sum().item()
            iou = inter / union if union > 0 else 1.0
            ious.append(iou)
    return float(np.mean(ious))


def compute_pck(model, loader, device, thresh_ratio=0.05):
    """
    Calculate PCK (Percentage of Correct Keypoints).
    Args:
        model: model that returns (seg, coords) or (seg, heatmaps, pafs)
        loader: DataLoader yielding (imgs, masks, kps_list, vis_list, paths, kps_orig)
        device: torch.device
        thresh_ratio: float, threshold ratio of max(image width, height)
    Returns:
        float: PCK value
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="PCK", leave=False):
            imgs, _, kps_list, vis_list, paths, kps_orig_list = batch
            imgs = imgs.to(device)
            out = model(imgs)

            # Extract predicted coords
            if isinstance(out, tuple) and out[1].ndim == 3:
                # coords output [B,K,2]
                pred_coords = out[1]
            else:
                # heatmaps output [B,K,H,W]
                heat = out[1]
                B, K, H, W = heat.shape
                pred_coords = torch.zeros((B, K, 2), device=device)
                for b in range(B):
                    for k in range(K):
                        hm = heat[b, k]
                        idx = torch.argmax(hm)
                        y, x = divmod(idx.item(), W)
                        pred_coords[b, k, 0] = x * (256 / W)
                        pred_coords[b, k, 1] = y * (256 / H)

            # Compute PCK per sample
            for b in range(pred_coords.size(0)):
                orig_size = Image.open(paths[b]).size  # (W, H)
                max_edge = max(orig_size)
                thresh = thresh_ratio * max_edge
                gt = kps_orig_list[b].numpy()  # (P,2)
                pred = pred_coords[b].cpu().numpy()  # (K,2)
                vis = vis_list[b].numpy()  # (P,)
                for (gx, gy), (px, py), v in zip(gt.reshape(-1,2), pred.reshape(-1,2), vis.reshape(-1)):
                    if v > 0:
                        dist = np.linalg.norm([gx - px, gy - py])
                        total += 1
                        if dist <= thresh:
                            correct += 1
    return correct / total if total > 0 else 0.0
