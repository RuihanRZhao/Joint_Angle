import torch
import torch.nn as nn
import numpy as np
import wandb

from config import DEVICE

# Loss functions
bce_loss = nn.BCEWithLogitsLoss()  # for segmentation (binary mask)
mse_loss = nn.MSELoss()            # for pose heatmaps

def dice_loss(pred, target):
    """Compute Dice loss for binary segmentation."""
    # pred: sigmoid logits [N,1,H,W], target: [N,1,H,W] 0/1
    pred = torch.sigmoid(pred)
    smooth = 1.0
    # Flatten
    pred_flat = pred.view(-1)
    target_flat = target.view(-1).float()
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return 1 - dice  # dice loss

def compute_segmentation_metrics(pred, target):
    """Compute IoU and pixel accuracy for segmentation batch."""
    # pred and target are binary tensors (after thresholding pred)
    pred = pred.byte()
    target = target.byte()
    intersection = (pred & target).float().sum()
    union = (pred | target).float().sum()
    iou = (intersection / (union + 1e-6)).item()
    # Pixel accuracy (fraction of pixels correctly classified)
    correct = (pred == target).float().sum()
    total = pred.numel()
    accuracy = (correct / total).item()
    return iou, accuracy

def compute_pose_PCK(pred_coords, true_coords, visibility, threshold):
    """
    Compute Percentage of Correct Keypoints (PCK) for one image.
    pred_coords, true_coords: arrays of shape (K,2)
    visibility: boolean array of shape (K,) indicating which keypoints are considered (visible).
    threshold: distance threshold (in pixels or normalized).
    """
    if len(pred_coords) != len(true_coords):
        return 0.0, 0  # no keypoints
    K = len(pred_coords)
    correct = 0
    total = 0
    for k in range(K):
        if visibility[k] == 0:
            continue  # skip keypoints not labeled (not in image)
        total += 1
        dist = np.linalg.norm(pred_coords[k] - true_coords[k])
        if dist < threshold:
            correct += 1
    if total == 0:
        return 0.0, 0
    return correct / total, total

def decode_heatmaps_to_coords(heatmaps):
    """
    Simple decoding: get the argmax of each heatmap channel.
    heatmaps: numpy array (K, H, W).
    returns: coords array (K,2) of (x, y) for each keypoint.
    """
    K, H, W = heatmaps.shape
    coords = np.zeros((K, 2))
    for k in range(K):
        hm = heatmaps[k]
        idx = np.argmax(hm)
        y, x = np.divmod(idx, W)
        coords[k] = np.array([x, y])
    return coords

def train_segmentation(model, train_loader, val_loader, optimizer, scheduler, num_epochs):
    best_iou = 0.0
    # WandB watch model
    wandb.watch(model, log="gradients", log_freq=100)
    for epoch in range(1, num_epochs+1):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)  # shape [N,1,H,W] logits
            loss_bce = bce_loss(outputs, masks.float())
            loss_dice = dice_loss(outputs, masks)
            loss = loss_bce + loss_dice
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        total_iou = 0.0
        total_acc = 0.0
        count = 0
        # For logging examples
        example_images = []
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                outputs = model(images)
                # Threshold output to get binary mask
                preds = (torch.sigmoid(outputs) > 0.5).cpu()
                # Compute metrics per batch
                for i in range(preds.size(0)):
                    iou, acc = compute_segmentation_metrics(preds[i], masks[i].cpu())
                    total_iou += iou
                    total_acc += acc
                    count += 1
                    if len(example_images) < 5:
                        # Log first few examples
                        img_np = (images[i].cpu().numpy().transpose(1,2,0) * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406]))  # de-normalize for visualization
                        img_np = (img_np * 255).astype(np.uint8).copy()
                        mask_pred = preds[i].squeeze().numpy()
                        mask_true = masks[i].cpu().numpy().squeeze()
                        # Use visualization utility to overlay mask (if available)
                        # Here we'll just use blending manually:
                        overlay_pred = img_np.copy()
                        overlay_true = img_np.copy()
                        overlay_pred[mask_pred==1] = [0,255,0]  # green overlay for predicted mask
                        overlay_true[mask_true==1] = [0,0,255]  # blue overlay for true mask
                        # Stack or side-by-side image
                        vis = np.hstack([overlay_true, overlay_pred])
                        example_images.append(wandb.Image(vis, caption=f"True (left) vs Pred (right) Mask"))
            # Compute average metrics
            avg_iou = total_iou / (count if count>0 else 1)
            avg_acc = total_acc / (count if count>0 else 1)
        # Log metrics to WandB
        wandb.log({
            "epoch": epoch,
            "seg_loss": epoch_loss,
            "val_seg_iou": avg_iou,
            "val_seg_accuracy": avg_acc,
            "examples": example_images
        })
        # Step the scheduler
        if scheduler:
            scheduler.step()
        # Save best model
        if avg_iou > best_iou:
            best_iou = avg_iou
            torch.save(model.state_dict(), "best_segmentation_model.pth")
            # Upload to wandb
            wandb.save("best_segmentation_model.pth")
    print("Best Segmentation IoU: {:.4f}".format(best_iou))

def train_pose(model, train_loader, val_loader, optimizer, scheduler, num_epochs, num_keypoints):
    best_pck = 0.0
    wandb.watch(model, log="gradients", log_freq=100)
    for epoch in range(1, num_epochs+1):
        model.train()
        running_loss = 0.0
        for images, heatmaps in train_loader:
            images = images.to(DEVICE)
            heatmaps = heatmaps.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)  # [N, K, H_out, W_out]
            loss = mse_loss(outputs, heatmaps)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        total_pck = 0.0
        total_count = 0
        example_images = []
        with torch.no_grad():
            for images, heatmaps in val_loader:
                images = images.to(DEVICE)
                heatmaps = heatmaps.to(DEVICE)
                outputs = model(images)  # [N, K, H_out, W_out]
                # Compute PCK for each image in batch
                for i in range(outputs.size(0)):
                    pred_hm = outputs[i].cpu().numpy()
                    true_hm = heatmaps[i].cpu().numpy()
                    pred_coords = decode_heatmaps_to_coords(pred_hm)
                    true_coords = decode_heatmaps_to_coords(true_hm)
                    # We consider all keypoints visible since val data typically has full annotations (or we can have a visibility mask if available)
                    vis = np.ones(num_keypoints, dtype=bool)
                    # Define threshold: e.g., 0.05 * max(dimensions) for COCO or head-size for MPII (simplified here)
                    thresh = 0.05 * max(outputs.size(2), outputs.size(3))  # relative to heatmap size (which is 1/4 of image)
                    pck, count = compute_pose_PCK(pred_coords * 4, true_coords * 4, vis, threshold=thresh * 4)
                    # multiplied by 4 to scale back to original image coordinate space
                    total_pck += pck * count
                    total_count += count
                    if len(example_images) < 5:
                        # Visualization: overlay keypoints on image
                        img_np = (images[i].cpu().numpy().transpose(1,2,0) * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406]))
                        img_np = (img_np * 255).astype(np.uint8).copy()
                        for k, (x, y) in enumerate(pred_coords * 4):  # scale up to image coords
                            if vis[k]:
                                cv2.circle(img_np, (int(x), int(y)), 3, (0,255,0), -1)  # green predicted
                        for k, (x, y) in enumerate(true_coords * 4):
                            if vis[k]:
                                cv2.circle(img_np, (int(x), int(y)), 3, (0,0,255), -1)  # red ground truth
                        example_images.append(wandb.Image(img_np, caption="Red: GT, Green: Predicted"))
            avg_pck = total_pck / (total_count if total_count>0 else 1)
        wandb.log({
            "epoch": epoch,
            "pose_loss": epoch_loss,
            "val_PCK": avg_pck,
            "examples": example_images
        })
        if scheduler:
            scheduler.step()
        if avg_pck > best_pck:
            best_pck = avg_pck
            torch.save(model.state_dict(), "best_pose_model.pth")
            wandb.save("best_pose_model.pth")
    print("Best Pose PCK: {:.4f}".format(best_pck))
