import cv2
import torch
import numpy as np
import argparse

from networks.Joint_Pose import JointPoseNet
from utils import get_config
from torchvision import transforms

# ------ SimCC 解码函数 ------
def decode_simcc(logits, axis_bins):
    """
    将 SimCC logits 解码为实际坐标（argmax）
    """
    prob = torch.nn.functional.softmax(logits, dim=-1)
    index = torch.argmax(prob, dim=-1)
    return index * (1.0 / axis_bins)

# ------ 可视化函数 ------
def draw_keypoints(img, keypoints, color=(0, 255, 0)):
    for x, y in keypoints:
        if x > 0 and y > 0:
            cv2.circle(img, (int(x), int(y)), 4, color, -1)
    return img

# ------ 主推理函数 ------
def run_webcam_inference(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = JointPoseNet(num_keypoints=17, bins=cfg['bins'], image_size=cfg['input_size'])
    model.load_state_dict(torch.load(cfg['checkpoint_path'], map_location=device)['model'])
    model.eval().to(device)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((cfg['input_size'], cfg['input_size'])),
        transforms.ToTensor(),
    ])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    print("开始摄像头推理，按 Q 退出")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = transform(input_img).unsqueeze(0).to(device)

        with torch.no_grad():
            x_logits, y_logits = model(resized)
            x_coords = decode_simcc(x_logits, cfg['bins']).cpu().numpy()[0]
            y_coords = decode_simcc(y_logits, cfg['bins']).cpu().numpy()[0]
            keypoints = np.stack([x_coords * frame.shape[1], y_coords * frame.shape[0]], axis=-1)

        # 显示关键点
        vis_frame = draw_keypoints(frame.copy(), keypoints)
        cv2.imshow("Pose Estimation", vis_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ------ 启动入口 ------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default="run/checkpoints/best_model_200.pth",
                        help="路径：保存的模型文件")
    args = parser.parse_args()

    config = get_config()
    config['checkpoint_path'] = args.checkpoint

    run_webcam_inference(config)
