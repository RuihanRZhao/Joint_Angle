import json
import math
import os
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
from tqdm import tqdm

from utils import get_config, ensure_coco_data


def load_coco_annotations(json_path):
    """Load COCO annotations from a JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def save_coco_annotations(json_path, images, annotations, categories):
    """Save COCO annotations (images, annotations, categories) to a JSON file."""
    output_data = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    with open(json_path, 'w') as f:
        json.dump(output_data, f)

def compute_bbox_from_polygon(segmentation):
    """
    Compute the tight bounding box [x_min, y_min, x_max, y_max] for a given segmentation polygon.
    The segmentation can be a list of lists (multiple polygons) or a single list (single polygon).
    """
    xs = []
    ys = []
    if isinstance(segmentation, list):
        if len(segmentation) > 0 and isinstance(segmentation[0], list):
            # Multiple polygons
            for poly in segmentation:
                xs.extend(poly[0::2])
                ys.extend(poly[1::2])
        else:
            # Single polygon
            xs.extend(segmentation[0::2])
            ys.extend(segmentation[1::2])
    else:
        # If segmentation is in RLE format (dict), this function does not handle it.
        return None
    if not xs or not ys:
        return None
    x_min = min(xs); x_max = max(xs)
    y_min = min(ys); y_max = max(ys)
    return [x_min, y_min, x_max, y_max]

def get_expanded_crop_coords(bbox, image_width, image_height, margin_ratio=0.1):
    """
    Given a bounding box [x_min, y_min, x_max, y_max] and image dimensions,
    compute expanded crop coordinates (x1, y1, x2, y2) by adding a margin around the box.
    margin_ratio is the fraction of the bounding box size to add (total) in each dimension.
    """
    x_min, y_min, x_max, y_max = bbox
    box_width = x_max - x_min
    box_height = y_max - y_min
    # Margin to add on each side (half of margin_ratio per side)
    margin_x = box_width * margin_ratio / 2.0
    margin_y = box_height * margin_ratio / 2.0
    # Expanded region, clamped to image boundaries
    x1 = math.floor(max(0, x_min - margin_x))
    y1 = math.floor(max(0, y_min - margin_y))
    x2 = math.ceil(min(image_width, x_max + margin_x))
    y2 = math.ceil(min(image_height, y_max + margin_y))
    return x1, y1, x2, y2

def process_image(image_info, annotations, input_dir, output_dir, margin_ratio):
    """
    Process a single image: crop one or more person instances from the image (each annotation in annotations list).
    Saves each crop as a new image. Returns a list of new image entries and annotation entries.
    """
    image_id = image_info['id']
    file_name = image_info['file_name']
    width = image_info['width']
    height = image_info['height']
    img_path = os.path.join(input_dir, file_name)
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Failed to open image {img_path}: {e}")
        return [], []
    new_images = []
    new_annotations = []
    for ann in annotations:
        ann_id = ann['id']
        # Compute person bounding box from segmentation polygon
        bbox_coords = compute_bbox_from_polygon(ann['segmentation'])
        if bbox_coords is None:
            continue
        x_min, y_min, x_max, y_max = bbox_coords
        # Expand bounding box with margin
        x1, y1, x2, y2 = get_expanded_crop_coords([x_min, y_min, x_max, y_max], width, height, margin_ratio)
        new_w = x2 - x1
        new_h = y2 - y1
        # Crop and save image
        crop = image.crop((x1, y1, x2, y2))
        new_file_name = f"{image_id}_{ann_id}.jpg"
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, new_file_name)
        crop.save(out_path, "JPEG")
        # Temporary image id using original image_id_annotation_id combined (as string for uniqueness)
        temp_image_id = f"{image_id}_{ann_id}"
        new_images.append({
            "id": temp_image_id,
            "file_name": new_file_name,
            "width": new_w,
            "height": new_h
        })
        # Adjust keypoints to the new crop coordinates
        keypoints = ann['keypoints']
        new_keypoints = []
        for i in range(0, len(keypoints), 3):
            kx, ky, kv = keypoints[i], keypoints[i+1], keypoints[i+2]
            if kv != 0:
                new_kx = kx - x1
                new_ky = ky - y1
            else:
                new_kx = 0
                new_ky = 0
            new_keypoints.extend([new_kx, new_ky, kv])
        # Compute bbox relative to new image coordinates
        new_bbox = [x_min - x1, y_min - y1, x_max - x_min, y_max - y_min]
        new_ann = {
            "id": ann_id,
            "image_id": temp_image_id,
            "category_id": ann['category_id'],
            "bbox": new_bbox,
            "num_keypoints": ann['num_keypoints'],
            "keypoints": new_keypoints,
            "iscrowd": 0
        }
        if 'area' in ann:
            new_ann['area'] = ann['area']
        new_annotations.append(new_ann)
    image.close()
    return new_images, new_annotations

def process_dataset(split_name, input_ann_path, input_img_dir, output_img_dir, output_ann_path, margin_ratio=0.1, num_workers=8):
    """
    Process a COCO dataset split (e.g., train or val), generating a single-person dataset subset.
    """
    data = load_coco_annotations(input_ann_path)
    images_info = {img['id']: img for img in data['images']}
    annotations = data['annotations']
    categories = data.get('categories', [])
    # Filter person annotations that have at least one keypoint and are not crowd
    person_annotations = [ann for ann in annotations if ann.get('category_id') == 1 and ann.get('num_keypoints', 0) > 0 and ann.get('iscrowd', 0) == 0]
    # Group annotations by image_id for processing
    ann_by_image = {}
    for ann in person_annotations:
        img_id = ann['image_id']
        if img_id in images_info:  # ensure image exists
            ann_by_image.setdefault(img_id, []).append(ann)
    # Prepare output containers
    all_new_images = []
    all_new_annotations = []
    # Create output directories
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_ann_path), exist_ok=True)
    # Process each image (with multi-threading for speed)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for img_id, ann_list in tqdm(
                ann_by_image.items(),
                desc=f"Submitting jobs for {split_name}",
                total=len(ann_by_image)
        ):
            img_info = images_info[img_id]
            futures.append(executor.submit(process_image, img_info, ann_list, input_img_dir, output_img_dir, margin_ratio))
        for future in tqdm(
                futures,
                desc=f"Processing images for {split_name}",
                total=len(futures)
        ):
            new_imgs, new_anns = future.result()
            all_new_images.extend(new_imgs)
            all_new_annotations.extend(new_anns)
    # Assign new sequential IDs for images and annotations
    temp_to_new_image_id = {}
    for idx, img_entry in enumerate(all_new_images, start=1):
        temp_id = img_entry['id']
        img_entry['id'] = idx
        temp_to_new_image_id[temp_id] = idx
    for idx, ann_entry in enumerate(all_new_annotations, start=1):
        temp_image_id = ann_entry['image_id']
        ann_entry['image_id'] = temp_to_new_image_id.get(temp_image_id, temp_image_id)
        ann_entry['id'] = idx
    # Filter categories to only include person category (id=1)
    if categories:
        categories = [cat for cat in categories if cat.get('id') == 1 or cat.get('name').lower() == 'person']
    else:
        categories = [{"id": 1, "name": "person"}]
    # Save new JSON annotations
    save_coco_annotations(output_ann_path, all_new_images, all_new_annotations, categories)


if __name__ == "__main__":
    args = get_config()
    input_dir = os.path.join(args['run_root'], 'data')
    output_dir = args['data_root']
    margin_ratio = args['person_margin_ratio']
    workers = args['num_workers_train']

    ensure_coco_data(input_dir, retries=2, backoff_factor=1.0)
    # Input paths
    train_ann = os.path.join(input_dir, "annotations", "person_keypoints_train2017.json")
    val_ann = os.path.join(input_dir, "annotations", "person_keypoints_val2017.json")
    train_imgs = os.path.join(input_dir, "train2017")
    val_imgs = os.path.join(input_dir, "val2017")
    # Output paths
    out_train_dir = os.path.join(output_dir, "train")
    out_val_dir = os.path.join(output_dir, "val")
    out_ann_dir = os.path.join(output_dir, "annotations")
    out_train_ann = os.path.join(out_ann_dir, "single_person_keypoints_train.json")
    out_val_ann = os.path.join(out_ann_dir, "single_person_keypoints_val.json")
    # Process training split
    print("Processing train split...")
    process_dataset("train", train_ann, train_imgs, out_train_dir, out_train_ann, margin_ratio, workers)
    # Process validation split
    print("Processing val split...")
    process_dataset("val", val_ann, val_imgs, out_val_dir, out_val_ann, margin_ratio, workers)
    print("Conversion completed. Output dataset is saved to:", output_dir)
