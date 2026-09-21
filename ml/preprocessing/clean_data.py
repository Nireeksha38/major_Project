import os
import json
import cv2
import numpy as np


def validate_yolo_label(label_path):
    """Checks if a YOLO label file is valid."""
    if not os.path.exists(label_path):
        return False, "Missing label file"

    with open(label_path, "r") as f:
        lines = f.readlines()

    if len(lines) == 0:
        return True, "Empty annotations (background image)"

    for idx, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) < 5:
            return False, f"Line {idx+1} has invalid format (<5 tokens)"
        try:
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                return False, f"Line {idx+1} coordinates out of [0, 1] bounds"
        except ValueError:
            return False, f"Line {idx+1} has non-numeric coordinates"

    return True, "Valid"


def clean_dataset(images_dir, labels_dir=None, report_path="reports/data_quality_report.json"):
    """
    Scans dataset, removes or flags corrupted images and invalid labels.
    Generates a data quality report.
    """
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    report = {
        "images_scanned": 0,
        "valid_images": 0,
        "corrupted_images": [],
        "invalid_labels": [],
        "missing_labels": [],
        "resolutions": {}
    }

    if not os.path.exists(images_dir):
        print(f"[CleanData] Directory '{images_dir}' not found.")
        return report

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    for root, _, files in os.walk(images_dir):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in valid_exts:
                report["images_scanned"] += 1
                img_path = os.path.join(root, fname)

                # Test image readability
                try:
                    img = cv2.imread(img_path)
                    if img is None or img.size == 0:
                        report["corrupted_images"].append(img_path)
                        continue

                    h, w = img.shape[:2]
                    res_key = f"{w}x{h}"
                    report["resolutions"][res_key] = report["resolutions"].get(res_key, 0) + 1
                    report["valid_images"] += 1

                except Exception as e:
                    report["corrupted_images"].append(f"{img_path} (Exception: {str(e)})")
                    continue

                # Check associated label if label directory is provided
                if labels_dir:
                    base_name = os.path.splitext(fname)[0]
                    lbl_path = os.path.join(labels_dir, f"{base_name}.txt")
                    valid_lbl, reason = validate_yolo_label(lbl_path)
                    if not valid_lbl:
                        report["invalid_labels"].append({"file": lbl_path, "reason": reason})

    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    print(f"[CleanData] Data quality report saved to {report_path}")
    print(f"[CleanData] Scanned: {report['images_scanned']}, Valid: {report['valid_images']}, Corrupted: {len(report['corrupted_images'])}")
    return report


if __name__ == "__main__":
    clean_dataset("ml/dataset/extracted_frames", "ml/dataset/yolo/labels/train")
