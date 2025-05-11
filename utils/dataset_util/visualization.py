import torch
import numpy as np
import wandb
import cv2

# COCO 17 点标准骨架（0-based）
_COCO_SKELETON = [
    (0,1), (0,2), (1,3), (2,4),
    (0,5), (0,6), (5,7), (7,9),
    (6,8), (8,10), (5,6), (11,12),
    (5,11), (6,12), (11,13), (13,15),
    (12,14), (14,16)
]

def draw_pose_on_image(image,
                       keypoints,
                       color=(255,0,0),
                       radius: int = 3,
                       thickness: int = 2,
                       use_wandb: bool = False):
    """
    在单张图像上绘制所有可见的 COCO 关键点及其连线，仅用 cv2。
    Args:
      image: numpy.ndarray (H,W,3) 或 torch.Tensor (3,H,W)/(1,3,H,W)
      keypoints: numpy.ndarray (K,3)/(1,K,3) 或 torch.Tensor 同形状
      color: RGB 三元组
      radius: 点半径
      thickness: 线宽
      use_wandb: 是否生成 wandb.Image
    Returns:
      out: 与输入 image 相同类型/形状的绘制后图像
      wb_img: wandb.Image 或 None
    """
    # 1) 准备底图 (BGR)，并记录元信息
    is_numpy = isinstance(image, np.ndarray)
    if is_numpy:
        img_rgb = image
        had_batch = False
        orig_dtype = image.dtype
    else:
        img_tensor = image
        # squeeze batch
        if img_tensor.ndim == 4 and img_tensor.shape[0] == 1:
            had_batch = True
            img_tensor = img_tensor.squeeze(0)
        else:
            had_batch = False
        # Tensor C×H×W -> numpy H×W×C
        arr = img_tensor.cpu().numpy()
        if arr.dtype in (np.float32, np.float64):
            arr = (arr*255).astype(np.uint8)
        img_rgb = arr.transpose(1,2,0)
        orig_dtype = img_tensor.dtype
    # 转 BGR
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # 2) 准备关键点 (K,3) array
    if isinstance(keypoints, np.ndarray):
        kps = keypoints
    else:
        kps = keypoints.cpu().numpy()
    # squeeze batch
    if kps.ndim == 3 and kps.shape[0] == 1:
        kps = kps.squeeze(0)
    # kps shape must be (K,>=3)
    # 取 x,y,v
    coords = kps[:, :2]
    vs     = kps[:, 2]

    H, W = img_bgr.shape[:2]
    bgr = (int(color[2]), int(color[1]), int(color[0]))

    # 3) 画连线：只连两端都可见的点
    for i,j in _COCO_SKELETON:
        if vs[i] > 0 and vs[j] > 0:
            x1,y1 = coords[i]
            x2,y2 = coords[j]
            if 0<= x1 < W and 0<= y1 < H and 0<= x2 < W and 0<= y2 < H:
                cv2.line(
                    img_bgr,
                    (int(x1),int(y1)),
                    (int(x2),int(y2)),
                    bgr, thickness, lineType=cv2.LINE_AA
                )

    # 4) 画点：只画可见点
    for (x,y), v in zip(coords, vs):
        if v > 0 and 0<= x < W and 0<= y < H:
            cv2.circle(
                img_bgr,
                (int(x),int(y)),
                radius, bgr, thickness=-1, lineType=cv2.LINE_AA
            )

    # 5) 恢复到原始格式
    # BGR -> RGB
    out_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if is_numpy:
        out = out_rgb.astype(orig_dtype)
    else:
        # numpy HWC -> Tensor CHW
        tch = torch.from_numpy(out_rgb.transpose(2,0,1))
        if orig_dtype.is_floating_point:
            tch = tch.float().div(255)
        else:
            tch = tch.to(orig_dtype)
        if had_batch:
            tch = tch.unsqueeze(0)
        out = tch

    # 6) wandb.Image 可选
    wb_img = None
    if use_wandb:
        if isinstance(out, np.ndarray):
            arr = out
        else:
            arr = out.detach().cpu().numpy().transpose(1,2,0)
            if arr.dtype in (np.float32, np.float64):
                arr = (arr*255).astype(np.uint8)
        wb_img = wandb.Image(arr)

    return out, wb_img


if __name__ == "__main__":
    import os
    import argparse
    import cv2
    import numpy as np
    from pycocotools.coco import COCO

    # 引入本模块的绘图函数
    from visualization import draw_pose_on_image

    parser = argparse.ArgumentParser(
        description="验证 draw_pose_on_image 在 COCO 数据集上的效果"
    )
    parser.add_argument(
        "--data_root", type=str, default='../../run/data', )
    parser.add_argument(
        "--ann_file", type=str, default="annotations/person_keypoints_val2017.json",)
    parser.add_argument(
        "--img_dir", type=str, default="val2017")
    parser.add_argument(
        "--out_dir", type=str, default="../../run/temp/",)
    parser.add_argument(
        "--num", type=int, default=5,)
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)

    # 加载 COCO 骨架关键点注释
    ann_path = os.path.join(args.data_root, args.ann_file)
    coco = COCO(ann_path)
    img_dir = os.path.join(args.data_root, args.img_dir)

    # 随机挑选若干张图
    img_ids = coco.getImgIds(catIds=[1])
    np.random.shuffle(img_ids)
    img_ids = img_ids[: args.num]

    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info["file_name"])
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"无法读取图像：{img_path}")
            continue
        # 转为 RGB ndarray
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 只取第一个人（单人姿态）
        ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=[1])
        anns = coco.loadAnns(ann_ids)
        if len(anns) == 0:
            print(f"图像 {img_path} 中没有检测到人员注释")
            continue
        kps = np.array(anns[0]["keypoints"], dtype=np.float32).reshape(-1, 3)

        # 绘制关键点和骨架
        vis, _ = draw_pose_on_image(img_rgb, kps, color=(255, 0, 0))

        # 输出结果（vis 是 H×W×3 ndarray，因为输入是 numpy）
        out_path = os.path.join(args.out_dir, img_info["file_name"])
        # 转回 BGR 保存
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, vis_bgr)
        print(f"Saved visualization for image {img_id} → {out_path}")