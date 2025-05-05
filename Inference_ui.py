import cv2
import torch
import numpy as np
from torchvision import transforms
from models.SegKP_Model import SegmentKeypointModel, PosePostProcessor, SKELETON


def main():
    # 参数设置
    checkpoint_path = 'run/models/best_keypoint_model_epoch.pth'  # 请替换为实际模型路径
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_size = 256

    # 模型加载
    model = SegmentKeypointModel()
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    postprocessor = PosePostProcessor()

    # 图像预处理
    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # 视频捕捉
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # BGR to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]

        # 预处理
        img_pil = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = preprocess(transforms.ToPILImage()(img_pil)).unsqueeze(0).to(device)

        # 推理
        with torch.no_grad():
            seg_logits, multi_kps = model(tensor)

        # 分割结果
        seg_prob = torch.sigmoid(seg_logits)[0,0].cpu().numpy()
        seg_mask = (seg_prob > 0.5).astype(np.uint8) * 255
        seg_mask = cv2.resize(seg_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        # 分割叠加
        seg_color = np.zeros_like(frame)
        seg_color[seg_mask>0] = (0, 0, 255)
        seg_overlay = cv2.addWeighted(frame, 0.7, seg_color, 0.3, 0)

        # 姿态结果
        persons = multi_kps[0]  # only first batch
        pose_vis = np.zeros_like(frame)
        for person in persons:
            # 绘制骨架线
            for (a,b) in SKELETON:
                if person[a][0] >=0 and person[b][0]>=0:
                    pt1 = tuple(person[a].astype(int))
                    pt2 = tuple(person[b].astype(int))
                    cv2.line(pose_vis, pt1, pt2, (0,255,0), 2)
            # 绘制关键点
            for (x,y) in person:
                if x>=0 and y>=0:
                    cv2.circle(pose_vis, (int(x),int(y)), 3, (255,0,0), -1)
        pose_overlay = cv2.addWeighted(frame, 0.7, pose_vis, 0.3, 0)

        # 显示
        cv2.imshow('Segmentation', seg_overlay)
        cv2.imshow('Pose Estimation', pose_overlay)

        # 按 q 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
