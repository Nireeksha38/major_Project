import os
import sys

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import cv2
import numpy as np
from detection.detector import YOLOv8Detector
from ml.preprocessing.video_to_frames import extract_frames_from_video
from ml.preprocessing.split_dataset import split_dataset_by_video
from ml.training.evaluate import evaluate_model


def prepare_dataset_from_videos(videos_root="videos",
                                raw_dir="ml/dataset/raw",
                                raw_images_dir="ml/dataset/raw_images",
                                frames_dir="ml/dataset/extracted_frames",
                                labels_dir="ml/dataset/raw_labels",
                                yolo_dir="ml/dataset/yolo",
                                frame_skip=30,
                                clean_staging=True):
    """
    Extracts frames from all available videos in videos/ and ml/dataset/raw/,
    copies any direct images from ml/dataset/raw/ or ml/dataset/raw_images/,
    generates YOLO format labels using YOLOv8 + Soft-NMS,
    and performs a Train/Val/Test split with zero temporal leakage.
    """
    import shutil
    if clean_staging:
        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir, ignore_errors=True)
        if os.path.exists(labels_dir):
            shutil.rmtree(labels_dir, ignore_errors=True)
        if os.path.exists(yolo_dir):
            shutil.rmtree(yolo_dir, ignore_errors=True)

    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(raw_images_dir, exist_ok=True)

    video_exts = (".mp4", ".avi", ".mov", ".mkv")
    image_exts = (".jpg", ".jpeg", ".png", ".bmp")

    video_paths = []
    direct_images = []

    # 1. Check videos/ subdirectories
    for root, _, files in os.walk(videos_root):
        for f in files:
            if f.lower().endswith(video_exts):
                video_paths.append(os.path.join(root, f))

    # 2. Check ml/dataset/raw/ for videos OR images
    if os.path.exists(raw_dir):
        for root, _, files in os.walk(raw_dir):
            for f in files:
                ext = f.lower()
                if ext.endswith(video_exts):
                    video_paths.append(os.path.join(root, f))
                elif ext.endswith(image_exts):
                    direct_images.append(os.path.join(root, f))

    # 3. Check ml/dataset/raw_images/ for images
    if os.path.exists(raw_images_dir):
        for root, _, files in os.walk(raw_images_dir):
            for f in files:
                if f.lower().endswith(image_exts):
                    direct_images.append(os.path.join(root, f))

    print(f"[Dataset Prep] Found {len(video_paths)} video source(s) and {len(direct_images)} image file(s).")
    for v in video_paths:
        print(f"  - Video: {v}")

    # Copy direct images to frames staging dir
    for img_p in direct_images:
        fname = os.path.basename(img_p)
        dst = os.path.join(frames_dir, fname)
        if os.path.abspath(img_p) != os.path.abspath(dst):
            shutil.copy2(img_p, dst)

    # Extract frames from each video
    total_frames = len(direct_images)
    for vpath in video_paths:
        total_frames += extract_frames_from_video(vpath, frames_dir, frame_skip=frame_skip)

    print(f"[Dataset Prep] Total dataset items staged: {total_frames}")

    if total_frames == 0:
        print("[Dataset Prep Warning] No videos or images found to prepare.")
        return

    # Generate annotations using YOLOv8 + Soft-NMS
    print("[Dataset Prep] Generating YOLO format bounding box annotations...")
    detector = YOLOv8Detector()
    frame_files = [f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    for idx, fname in enumerate(frame_files):
        img_path = os.path.join(frames_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        boxes, scores, classes, detections = detector.detect_and_refine(img, use_soft_nms=True)

        base_name = os.path.splitext(fname)[0]
        label_file = os.path.join(labels_dir, f"{base_name}.txt")

        with open(label_file, "w") as lf:
            for box, score, cls_id in zip(boxes, scores, classes):
                # Convert xyxy to normalized YOLO format: class_id cx cy w h
                x1, y1, x2, y2 = box
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                cx = (x1 + x2) / (2.0 * w)
                cy = (y1 + y2) / (2.0 * h)

                # Clamp to [0, 1]
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                bw = max(0.0, min(1.0, bw))
                bh = max(0.0, min(1.0, bh))

                lf.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    print(f"[Dataset Prep] Annotated {len(frame_files)} frames.")

    # Split dataset by video ID into Train (70%), Val (20%), Test (10%)
    print("[Dataset Prep] Splitting dataset into 70% Train / 20% Val / 10% Test (Video-level grouping)...")
    split_counts = split_dataset_by_video(
        frames_dir=frames_dir,
        labels_dir=labels_dir,
        output_yolo_dir=yolo_dir,
        train_ratio=0.70,
        val_ratio=0.20,
        test_ratio=0.10
    )

    from ml.preprocessing.augmentation import augment_training_set
    print("[Dataset Prep] Applying realistic data augmentations to training partition...")
    train_images = os.path.join(yolo_dir, "images", "train")
    train_labels = os.path.join(yolo_dir, "labels", "train")
    aug_count = augment_training_set(images_dir=train_images, labels_dir=train_labels)
    print(f"[Dataset Prep] Added {aug_count} augmented samples to training set.")

    # Evaluate baseline model on test split
    print("[Dataset Prep] Running baseline model evaluation on held-out test split...")
    dataset_yaml = os.path.join(yolo_dir, "dataset.yaml")
    evaluate_model(model_path="yolov8n.pt", dataset_yaml=dataset_yaml)

    print("[Dataset Prep] Dataset preparation & augmentation complete!")
    return split_counts


if __name__ == "__main__":
    prepare_dataset_from_videos()
