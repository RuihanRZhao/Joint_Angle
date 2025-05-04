import os
import argparse
import torch
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms

from models.SegKP_Model import SegmentKeypointModel, PosePostProcessor
from utils.visualization import SKELETON


def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for segmentation and pose estimation")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input image or directory of images')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save outputs')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for inference: cuda or cpu')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Input image size (square)')
    return parser.parse_args()


def setup_model(checkpoint_path, device):
    # Initialize model and load weights
    model = SegmentKeypointModel()
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    postprocessor = PosePostProcessor()
    return model, postprocessor


def preprocess_image(image_path, img_size, device):
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = img.size
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    tensor = transform(img).unsqueeze(0).to(device)
    return img, tensor, (orig_w, orig_h)


def save_segmentation(mask, orig_size, out_path):
    # mask: numpy array [H, W] values 0/255
    mask_resized = cv2.resize(mask, orig_size, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(out_path, mask_resized)


def overlay_segmentation(img_np, mask, out_path):
    # Overlay red mask on original image
    overlay = img_np.copy()
    overlay[mask > 0] = [0, 0, 255]
    blend = cv2.addWeighted(img_np, 0.7, overlay, 0.3, 0)
    cv2.imwrite(out_path, blend)


def save_pose(img_np, persons, out_path):
    vis = np.zeros_like(img_np)
    # Draw skeleton and keypoints
    for person in persons:
        # Skeleton lines
        for (a, b) in SKELETON:
            if person[a][0] >= 0 and person[b][0] >= 0:
                pt1 = tuple(person[a].astype(int))
                pt2 = tuple(person[b].astype(int))
                cv2.line(vis, pt1, pt2, (0, 255, 0), 2)
        # Keypoint circles
        for (x, y) in person:
            if x >= 0 and y >= 0:
                cv2.circle(vis, (int(x), int(y)), 3, (255, 0, 0), -1)
    blend = cv2.addWeighted(img_np, 0.7, vis, 0.3, 0)
    cv2.imwrite(out_path, blend)


def process_image(path, model, postproc, args):
    # Prepare
    img, tensor, (orig_w, orig_h) = preprocess_image(path, args.img_size, args.device)
    img_np = np.array(img)

    # Inference
    with torch.no_grad():
        seg_logits, multi_kps = model(tensor)

    # Segmentation processing
    seg_prob = torch.sigmoid(seg_logits)[0, 0].cpu().numpy()
    seg_mask = (seg_prob > 0.5).astype(np.uint8) * 255
    base = os.path.splitext(os.path.basename(path))[0]

    # Save segmentation mask
    mask_out = os.path.join(args.output_dir, f"{base}_mask.png")
    save_segmentation(seg_mask, (orig_w, orig_h), mask_out)

    # Save overlay
    overlay_out = os.path.join(args.output_dir, f"{base}_overlay.png")
    overlay_segmentation(img_np, seg_mask, overlay_out)

    # Pose visualization
    pose_out = os.path.join(args.output_dir, f"{base}_pose.png")
    save_pose(img_np, multi_kps[0], pose_out)

    print(f"Processed {path}: mask -> {mask_out}, overlay -> {overlay_out}, pose -> {pose_out}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model, postproc = setup_model(args.checkpoint, device)

    # Handle single image or directory
    if os.path.isdir(args.input):
        for fname in sorted(os.listdir(args.input)):
            fpath = os.path.join(args.input, fname)
            process_image(fpath, model, postproc, args)
    else:
        process_image(args.input, model, postproc, args)

if __name__ == '__main__':
    main()
