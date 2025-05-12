
# Joint Angle Estimation via SimCC

This repository contains the implementation of a lightweight deep learning model for **single-person joint angle estimation** based on the SimCC framework.

---

## 📦 Project Structure

- `networks/Joint_Pose.py`: Main model definition (SimCC-based)
- `single_person_dataset.py`: Convert COCO to single-person format
- `train.py`: Training pipeline with evaluation and logging
- `utils/`: Utility functions (data loader, training, evaluation, loss)
- `paper/`: LaTeX draft for academic report
- `config.yaml`: Training configuration
- `infer_webcam.py`: Real-time webcam inference (see below)

---

## 📋 Installation

```bash
# Clone repo
git clone https://github.com/RuihanRZhao/Joint_Angle.git
cd Joint_Angle

# Install dependencies
pip install -r requirements.txt
```

---

## 📁 Dataset Preparation

This repo supports COCO dataset (keypoints):

1. Download COCO 2017 keypoints:
   - `train2017/`, `val2017/`
   - `annotations/person_keypoints_train2017.json`, etc.

2. Place them under:
```
run/data/
├── train2017/
├── val2017/
└── annotations/
    ├── person_keypoints_train2017.json
    └── person_keypoints_val2017.json
```

3. Convert to single-person dataset:

```bash
python single_person_dataset.py
```

Output will be in `run/single_person/` (cropped images and new annotations).

---

## 🏋️ Training

```bash
python train.py
```

Training supports:
- AMP training via `torch.cuda.amp`
- EMA (exponential moving average)
- Learning rate warmup and cosine decay
- `wandb` logging

Adjust config in `config.yaml` as needed.

---

## 🎥 Real-Time Webcam Inference

1. Make sure you have a trained model, e.g., `run/checkpoints/best_model_200.pth`

2. Run:

```bash
python infer_webcam.py --checkpoint run/checkpoints/best_model_200.pth
```

- Press `Q` to quit.
- Webcam feed will show predicted keypoints overlay.

---

## 📊 Model Overview

The model is composed of:

- Efficient CNN backbone with SE and CSP blocks
- SimCC output head with 1D conv and transposed conv
- X/Y logits are separately estimated then decoded

---

## ✏️ Citation

If you find this work useful, please consider citing the project or referencing the LaTeX paper draft in `paper/`.

---

## 🔧 Notes

- Ensure `get_config()` works properly — it reads from `config.yaml`
- Only single-person keypoint estimation is supported
- Training is based on COCO keypoints, but can be adapted to other datasets

---

## 📬 Contact

Created by Ruihan Zhao. For questions, please open an issue or reach out.

