import os
import cv2
import numpy as np


def horizontal_flip(image, labels=None):
    """
    Horizontally flips image and adjusts YOLO bounding box coordinates.
    labels: list of [cls_id, cx, cy, w, h]
    """
    flipped_img = cv2.flip(image, 1)
    flipped_labels = []

    if labels is not None:
        for lbl in labels:
            cls_id, cx, cy, w, h = lbl
            # New cx = 1.0 - cx
            flipped_labels.append([cls_id, 1.0 - cx, cy, w, h])

    return flipped_img, flipped_labels


def adjust_brightness_contrast(image, alpha=1.1, beta=10):
    """Adjusts contrast (alpha) and brightness (beta)."""
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted


def augment_training_set(images_dir="ml/dataset/yolo/images/train", labels_dir="ml/dataset/yolo/labels/train"):
    """Applies realistic crowd dataset augmentations to train partition."""
    if not os.path.exists(images_dir):
        print(f"[Augment] Directory '{images_dir}' not found.")
        return 0

    aug_count = 0
    valid_exts = {".jpg", ".jpeg", ".png"}

    for fname in os.listdir(images_dir):
        if "_aug_" in fname:
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext in valid_exts:
            img_path = os.path.join(images_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue

            base_name = os.path.splitext(fname)[0]
            lbl_path = os.path.join(labels_dir, f"{base_name}.txt")

            labels = []
            if os.path.exists(lbl_path):
                with open(lbl_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            labels.append([int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])

            # Augmentation 1: Horizontal Flip
            flip_img, flip_lbls = horizontal_flip(img, labels)
            flip_fname = f"{base_name}_aug_flip{ext}"
            cv2.imwrite(os.path.join(images_dir, flip_fname), flip_img)

            if flip_lbls and os.path.exists(labels_dir):
                with open(os.path.join(labels_dir, f"{base_name}_aug_flip.txt"), "w") as f:
                    for l in flip_lbls:
                        f.write(f"{l[0]} {l[1]:.6f} {l[2]:.6f} {l[3]:.6f} {l[4]:.6f}\n")

            # Augmentation 2: Brightness variation
            bright_img = adjust_brightness_contrast(img, alpha=0.9, beta=15)
            bright_fname = f"{base_name}_aug_bright{ext}"
            cv2.imwrite(os.path.join(images_dir, bright_fname), bright_img)

            if labels and os.path.exists(labels_dir):
                with open(os.path.join(labels_dir, f"{base_name}_aug_bright.txt"), "w") as f:
                    for l in labels:
                        f.write(f"{l[0]} {l[1]:.6f} {l[2]:.6f} {l[3]:.6f} {l[4]:.6f}\n")

            aug_count += 2

    print(f"[Augment] Successfully generated {aug_count} augmented samples.")
    return aug_count


if __name__ == "__main__":
    augment_training_set()
