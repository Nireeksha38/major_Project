import os
import sys
import shutil

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import argparse
from ultralytics import YOLO


def train_crowd_model(dataset_yaml="ml/dataset/yolo/dataset.yaml",
                      base_model="yolov8n.pt",
                      epochs=10,
                      img_size=640,
                      batch_size=8,
                      output_dir="ml/models"):
    """
    Fine-tunes YOLOv8 on crowd person dataset.
    Auto-detects GPU (CUDA) or CPU.
    Saves best.pt and last.pt weights.
    """
    if not os.path.exists(dataset_yaml):
        print(f"[Train Error] Dataset config '{dataset_yaml}' not found.")
        print("Please extract frames and split dataset first.")
        return None

    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Initializing YOLOv8 model ({base_model}) on device: {device}...")

    model = YOLO(base_model)

    print(f"[Train] Starting fine-tuning for {epochs} epochs (batch={batch_size}, imgsz={img_size})...")
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        project=output_dir,
        name="crowd_yolov8_run",
        exist_ok=True,
        plots=True,
        verbose=True
    )

    # Locate and copy best.pt / last.pt
    models_weights_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "models", "weights")
    os.makedirs(models_weights_dir, exist_ok=True)

    run_weights = os.path.join(output_dir, "crowd_yolov8_run", "weights")
    if os.path.exists(run_weights):
        best_path = os.path.join(run_weights, "best.pt")
        last_path = os.path.join(run_weights, "last.pt")
        
        target_best = os.path.join(output_dir, "best.pt")
        target_last = os.path.join(output_dir, "last.pt")
        target_models_best = os.path.join(models_weights_dir, "best.pt")

        if os.path.exists(best_path):
            shutil.copy2(best_path, target_best)
            shutil.copy2(best_path, target_models_best)
            print(f"[Train] Best model saved to {target_best} and {target_models_best}")
        if os.path.exists(last_path):
            shutil.copy2(last_path, target_last)
            print(f"[Train] Last model saved to {target_last}")

    print("[Train] YOLOv8 fine-tuning completed.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Crowd Dataset.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=4, help="Batch size.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    args = parser.parse_args()

    train_crowd_model(epochs=args.epochs, batch_size=args.batch, img_size=args.imgsz)
