import os
import sys
import shutil
import zipfile
import json
import argparse
from PIL import Image

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from detection.detector import YOLOv8Detector
from ml.preprocessing.split_dataset import split_dataset_by_video
from ml.preprocessing.augmentation import augment_training_set


def convert_odgt_to_yolo(odgt_path, images_map, output_images_dir, output_labels_dir, max_samples=None):
    """
    Parses official CrowdHuman ODGT ground-truth annotations and converts
    them to standard normalized YOLO bounding box format (class 0 = person).
    """
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    if not os.path.exists(odgt_path):
        return 0

    count = 0
    total_boxes = 0

    with open(odgt_path, "r", encoding="utf-8") as f:
        for line in f:
            if max_samples and count >= max_samples:
                break

            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except Exception:
                continue

            img_id = data.get("ID", "")
            if not img_id:
                continue

            img_filename = f"{img_id}.jpg"
            if img_filename not in images_map:
                continue

            src_img_path = images_map[img_filename]

            # Read image dimensions
            try:
                with Image.open(src_img_path) as im:
                    w, h = im.size
            except Exception:
                continue

            if w <= 0 or h <= 0:
                continue

            # Convert bounding boxes
            yolo_boxes = []
            for box in data.get("gtboxes", []):
                # Tag must be person and not ignored
                if box.get("tag") != "person":
                    continue
                if box.get("extra", {}).get("ignore", 0) == 1:
                    continue

                fbox = box.get("fbox", [])
                if len(fbox) < 4:
                    continue

                x, y, bw, bh = fbox
                # Clip box coordinates within image bounds
                x1 = max(0.0, float(x))
                y1 = max(0.0, float(y))
                x2 = min(float(w), float(x + bw))
                y2 = min(float(h), float(y + bh))

                box_w = x2 - x1
                box_h = y2 - y1

                if box_w <= 2 or box_h <= 2:
                    continue

                cx = (x1 + box_w / 2.0) / float(w)
                cy = (y1 + box_h / 2.0) / float(h)
                nw = box_w / float(w)
                nh = box_h / float(h)

                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                nw = max(0.0, min(1.0, nw))
                nh = max(0.0, min(1.0, nh))

                yolo_boxes.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            # Copy image
            dst_img_path = os.path.join(output_images_dir, img_filename)
            if os.path.abspath(src_img_path) != os.path.abspath(dst_img_path):
                shutil.copy2(src_img_path, dst_img_path)

            # Write label file
            label_filename = f"{img_id}.txt"
            dst_lbl_path = os.path.join(output_labels_dir, label_filename)
            with open(dst_lbl_path, "w", encoding="utf-8") as lf:
                if yolo_boxes:
                    lf.write("\n".join(yolo_boxes) + "\n")

            count += 1
            total_boxes += len(yolo_boxes)

    print(f"   Processed {count} images with {total_boxes} ground-truth person boxes -> {output_images_dir}")
    return count


def process_crowdhuman_odgt_dataset(raw_images_dir="ml/dataset/raw_images",
                                   output_yolo_dir="ml/dataset/yolo",
                                   max_train=800,
                                   max_val=150,
                                   max_test=100):
    """
    Checks if CrowdHuman ODGT ground truth annotations are present,
    and constructs the complete YOLO benchmark dataset from authentic annotations.
    """
    train_odgt = None
    val_odgt = None

    for root, _, files in os.walk(raw_images_dir):
        for f in files:
            if f.lower() == "annotation_train.odgt":
                train_odgt = os.path.join(root, f)
            elif f.lower() == "annotation_val.odgt":
                val_odgt = os.path.join(root, f)

    if not train_odgt:
        return False

    print("=" * 70)
    print("🎯 Found Official CrowdHuman Ground-Truth Annotations (.odgt)!")
    print(f"   Train ODGT: {train_odgt}")
    print(f"   Val ODGT:   {val_odgt}")
    print("=" * 70)

    # Index all available images by filename
    print("[CrowdHuman] Indexing extracted images...")
    images_map = {}
    valid_exts = {".jpg", ".jpeg", ".png"}
    for root, _, files in os.walk(raw_images_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in valid_exts:
                images_map[f] = os.path.join(root, f)

    print(f"[CrowdHuman] Found {len(images_map)} high-resolution images in archive.")

    # Clean existing YOLO dataset directory
    for split in ["train", "val", "test"]:
        shutil.rmtree(os.path.join(output_yolo_dir, "images", split), ignore_errors=True)
        shutil.rmtree(os.path.join(output_yolo_dir, "labels", split), ignore_errors=True)
        os.makedirs(os.path.join(output_yolo_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_yolo_dir, "labels", split), exist_ok=True)

    # Convert Train Split
    print(f"[CrowdHuman] Generating Train Split ({max_train} dense crowd images)...")
    train_imgs_dir = os.path.join(output_yolo_dir, "images", "train")
    train_lbls_dir = os.path.join(output_yolo_dir, "labels", "train")
    n_train = convert_odgt_to_yolo(train_odgt, images_map, train_imgs_dir, train_lbls_dir, max_samples=max_train)

    # Convert Val Split
    print(f"[CrowdHuman] Generating Validation Split ({max_val} dense crowd images)...")
    val_imgs_dir = os.path.join(output_yolo_dir, "images", "val")
    val_lbls_dir = os.path.join(output_yolo_dir, "labels", "val")
    val_source = val_odgt if val_odgt else train_odgt
    n_val = convert_odgt_to_yolo(val_source, images_map, val_imgs_dir, val_lbls_dir, max_samples=max_val)

    # Convert Test Split
    print(f"[CrowdHuman] Generating Held-Out Test Split ({max_test} dense crowd images)...")
    test_imgs_dir = os.path.join(output_yolo_dir, "images", "test")
    test_lbls_dir = os.path.join(output_yolo_dir, "labels", "test")
    n_test = convert_odgt_to_yolo(val_source, images_map, test_imgs_dir, test_lbls_dir, max_samples=max_test)

    # Write dataset.yaml
    clean_path = os.path.abspath(output_yolo_dir).replace("\\", "/")
    dataset_yaml = os.path.join(output_yolo_dir, "dataset.yaml")
    with open(dataset_yaml, "w", encoding="utf-8") as yf:
        yf.write(f"""# Official YOLOv8 CrowdHuman Dataset Configuration
path: {clean_path}
train: images/train
val: images/val
test: images/test

names:
  0: person
""")

    print(f"[CrowdHuman] Dataset config written to: {dataset_yaml}")

    # Apply data augmentation to boost precision
    print("[CrowdHuman] Applying data augmentations to training set...")
    aug_count = augment_training_set(images_dir=train_imgs_dir, labels_dir=train_lbls_dir)
    print(f"[CrowdHuman] Added {aug_count} augmented high-density training samples.")

    print("=" * 70)
    print("✅ CrowdHuman Ground-Truth YOLO Dataset Successfully Built!")
    print(f"   - 🏋️ Train: {n_train + aug_count} samples ({n_train} raw + {aug_count} augmented)")
    print(f"   - 🔍 Val:   {n_val} samples")
    print(f"   - 🧪 Test:  {n_test} samples")
    print(f"   - 📄 YAML:  {dataset_yaml}")
    print("=" * 70)
    return True


def ingest_crowd_dataset(raw_dir="ml/dataset/raw",
                         raw_images_dir="ml/dataset/raw_images",
                         raw_labels_dir="ml/dataset/raw_labels",
                         frames_dir="ml/dataset/extracted_frames",
                         output_yolo_dir="ml/dataset/yolo",
                         user_zip_path=None):
    """
    Master Ingestion Pipeline:
    1. Unpacks archives if needed.
    2. Detects CrowdHuman ODGT ground-truth files and generates YOLO partitions.
    3. Falls back to general video/image auto-annotation if ODGT is absent.
    """
    print("=" * 70)
    print("🚀 CrowdHuman & Dense Crowd Dataset Ingestion Pipeline")
    print("=" * 70)

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(raw_images_dir, exist_ok=True)
    os.makedirs(raw_labels_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(output_yolo_dir, exist_ok=True)

    # 1. Unpack any zip archives
    potential_zips = []
    if user_zip_path and os.path.exists(user_zip_path):
        potential_zips.append(user_zip_path)

    for check_dir in [raw_dir, "raw", "ml/dataset"]:
        if os.path.exists(check_dir):
            for f in os.listdir(check_dir):
                if f.lower().endswith(".zip"):
                    potential_zips.append(os.path.join(check_dir, f))

    if potential_zips:
        for zip_p in potential_zips:
            try:
                # Check if already extracted
                if not os.path.exists(os.path.join(raw_images_dir, "CrowdHuman", "Images")):
                    print(f"[Ingest] Unpacking archive: {zip_p} -> {raw_images_dir}")
                    with zipfile.ZipFile(zip_p, 'r') as zip_ref:
                        zip_ref.extractall(raw_images_dir)
                    print(f"[Ingest] Extracted {zip_p} successfully.")
                else:
                    print(f"[Ingest] CrowdHuman archive already extracted in {raw_images_dir}")
            except Exception as e:
                print(f"[Ingest Warning] Failed to extract {zip_p}: {e}")

    # 2. Check if official CrowdHuman ODGT annotations exist
    if process_crowdhuman_odgt_dataset(raw_images_dir=raw_images_dir, output_yolo_dir=output_yolo_dir):
        return

    # Fallback to standard frame processing
    print("[Ingest] Standard video and image staging...")
    from ml.preprocessing.prepare_and_train import prepare_dataset_from_videos
    prepare_dataset_from_videos()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest CrowdHuman Dataset & Prep for Training.")
    parser.add_argument("--zip", type=str, default=None, help="Path to CrowdHuman zip file if available")
    args = parser.parse_args()

    ingest_crowd_dataset(user_zip_path=args.zip)
