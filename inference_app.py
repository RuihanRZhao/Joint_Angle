# inference_app.py
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from models.segmentation_model import UNetSegmentation
from models.pose_model import PoseEstimationModel
from models.teacher_model import TeacherModel
from models.student_model import StudentModel


class HumanPoseSegmentation:
    def __init__(self, model_type='student', device='cuda'):
        """
        初始化模型
        Args:
            model_type: 'teacher' 或 'student'
            device: 推理设备
        """
        self.device = torch.device(device)
        self.model_type = model_type
        self.input_size = (512, 512)  # 模型输入尺寸

        # 初始化模型结构
        if model_type == 'teacher':
            self.seg_model = UNetSegmentation().to(self.device)
            self.pose_model = PoseEstimationModel(in_channels=4).to(self.device)
            self.model = TeacherModel(self.seg_model, self.pose_model).to(self.device)
        else:
            self.model = StudentModel(num_keypoints=17).to(self.device)

        # 加载预训练权重
        self.load_weights(f'checkpoints/best_{model_type}.pth')

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.input_size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def load_weights(self, checkpoint_path):
        """加载训练好的模型权重"""
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        if self.model_type == 'teacher':
            self.seg_model.load_state_dict(state_dict['seg'])
            self.pose_model.load_state_dict(state_dict['pose'])
        else:
            self.model.load_state_dict(state_dict)

    def preprocess(self, image):
        """图像预处理"""
        orig_size = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        return tensor, orig_size

    def postprocess(self, seg_logits, pose_logits, orig_size):
        """后处理输出结果"""
        # 人体分割处理
        seg_mask = torch.sigmoid(seg_logits).squeeze().cpu().numpy()
        seg_mask = (seg_mask > 0.5).astype(np.uint8)
        seg_mask = cv2.resize(seg_mask, orig_size[::-1], interpolation=cv2.INTER_NEAREST)

        # 姿态估计处理
        pose_heatmaps = torch.sigmoid(pose_logits).cpu().detach().numpy()
        keypoints = []
        for i in range(pose_heatmaps.shape[1]):
            heatmap = cv2.resize(pose_heatmaps[0, i], orig_size[::-1])
            y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            keypoints.append((x, y))

        return seg_mask, np.array(keypoints)

    def visualize(self, image, seg_mask, keypoints):
        """可视化结果"""
        # 叠加分割掩码
        overlay = image.copy()
        overlay[seg_mask == 1] = [0, 255, 0]
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)

        # 绘制关键点
        for (x, y) in keypoints:
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

        return image

    def predict(self, image_path, visualize=True):
        """完整推理流程"""
        # 读取并预处理图像
        image = cv2.imread(image_path)
        tensor, orig_size = self.preprocess(image)

        # 模型推理
        with torch.no_grad():
            if self.model_type == 'teacher':
                seg_logits, pose_logits = self.model(tensor)
            else:
                seg_logits, pose_logits = self.model(tensor)

        # 后处理
        seg_mask, keypoints = self.postprocess(seg_logits, pose_logits, orig_size)

        # 可视化
        if visualize:
            result = self.visualize(image, seg_mask, keypoints)
            cv2.imshow('Result', result)
            cv2.waitKey(0)

        return seg_mask, keypoints


if __name__ == '__main__':
    # 使用示例
    detector = HumanPoseSegmentation(model_type='student')
    seg_mask, keypoints = detector.predict('test_image.jpg')
