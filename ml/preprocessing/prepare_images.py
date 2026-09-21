import os
import sys
import shutil
import cv2
import argparse
from tqdm import tqdm

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from detection.detector import YOLOv8Detector
from ml.preprocessing.split_dataset import split_dataset_by_video
from ml.training.evaluate import evaluate_model


def process_random_images(images_input_dir="ml/dataset/raw_images",
                          labels_input_dir="ml/dataset/raw_labels",
                          frames_staging_dir="ml/dataset/extracted_frames",
                          output_yolo_dir="ml/dataset/yolo",
                          auto_annotate=True):
    """
    Takes random images from Kaggle (either labeled or unlabeled),
    auto-generates YOLO format bounding box annotations if labels are missing,
    and splits them into 70% Train / 20% Val / 10% Test in ml/dataset/yolo.
    """
    os.makedirs(images_input_dir, exist_ok=True)
    os.makedirs(labels_input_dir, exist_ok=True)
    os.makedirs(frames_staging_dir, exist_ok=True)

    # 1. Collect all images
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = []

    # Check images_input_dir
    for root, _, files in os.walk(images_input_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in valid_exts:
                image_files.append((os.path.join(root, f), f))

    # Also check frames_staging_dir if images were placed directly there
    for f in os.listdir(frames_staging_dir):
        if os.path.splitext(f)[1].lower() in valid_exts:
            full_p = os.path.join(frames_staging_dir, f)
            if (full_p, f) not in image_files:
                image_files.append((full_p, f))

    print(f"[Image Prep] Found {len(image_files)} image(s) to process.")
    if len(image_files) == 0:
        print(f"[Image Prep Info] No images found. Place your Kaggle images in: '{images_input_dir}'")
        return

    # 2. Copy images to staging directory
    for src_path, fname in image_files:
        dst_path = os.path.join(frames_staging_dir, fname)
        if os.path.abspath(src_path) != os.path.abspath(dst_path):
            shutil.copy2(src_path, dst_path)

    # 3. Handle Annotations
    detector = None
    annotated_count = 0

    print("[Image Prep] Checking / Generating YOLO bounding box annotations...")
    for _, fname in image_files:
        base_name = os.path.splitext(fname)[0]
        label_path = os.path.join(labels_input_dir, f"{base_name}.txt")

        # If label already exists, keep it
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            continue

        # If auto-annotate is enabled, generate person detections
        if auto_annotate:
            if detector is None:
                detector = YOLOv8Detector()

            img_path = os.path.join(frames_staging_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w = img.shape[:2]
            boxes, scores, classes, _ = detector.detect_and_refine(img, use_soft_nms=True)

            with open(label_path, "w") as lf:
                for box, score, cls_id in zip(boxes, scores, classes):
                    x1, y1, x2, y2 = box
                    bw = max(0.0, min(1.0, (x2 - x1) / w))
                    bh = max(0.0, min(1.0, (y2 - y1) / h))
                    cx = max(0.0, min(1.0, (x1 + x2) / (2.0 * w)))
                    cy = max(0.0, min(1.0, (y1 + y2) / (2.0 * h)))
                    lf.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            
            annotated_count += 1

    print(f"[Image Prep] Finished annotation. {annotated_count} images were auto-annotated.")

    # 4. Split dataset into Train (70%), Val (20%), Test (10%)
    print("[Image Prep] Splitting images into Train (70%) / Val (20%) / Test (10%)...")
    split_counts = split_dataset_by_video(
        frames_dir=frames_staging_dir,
        labels_dir=labels_input_dir,
        output_yolo_dir=output_yolo_dir,
        train_ratio=0.70,
        val_ratio=0.20,
        test_ratio=0.10
    )

    print("[Image Prep] Success! Ready for training: python ml/training/train_yolo.py")
    return split_counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare random image datasets for YOLO training.")
    parser.add_argument("--images-dir", type=str, default="ml/dataset/raw_images", help="Folder with raw image files")
    parser.add_argument("--labels-dir", type=str, default="ml/dataset/raw_labels", help="Folder with YOLO label files")
    args = parser.parse_args()

    process_random_images(images_input_dir=args.images_dir, labels_input_dir=args.labels_dir)
