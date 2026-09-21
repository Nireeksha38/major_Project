import os
import shutil
import random
from collections import defaultdict


def split_dataset_by_video(frames_dir="ml/dataset/extracted_frames",
                           labels_dir="ml/dataset/raw_labels",
                           output_yolo_dir="ml/dataset/yolo",
                           train_ratio=0.70,
                           val_ratio=0.20,
                           test_ratio=0.10,
                           seed=42):
    """
    Splits extracted frames into Train / Val / Test (70% / 20% / 10%).
    
    IMPORTANT: DATA LEAKAGE PREVENTION:
    Groups frames by their source video ID so that all frames from Video A go to Train,
    all frames from Video B go to Val, and all frames from Video C go to Test.
    Never scatters adjacent frames of the same video across train and test sets.
    """
    random.seed(seed)

    # 1. Group frame files by video prefix
    if not os.path.exists(frames_dir):
        print(f"[Split] Frames directory '{frames_dir}' does not exist.")
        return

    video_groups = defaultdict(list)
    valid_exts = {".jpg", ".jpeg", ".png"}

    for fname in os.listdir(frames_dir):
        ext = os.path.splitext(fname)[1].lower()
        if ext in valid_exts:
            # Extract video prefix if extracted from video, or treat image as individual sample
            if "_frame_" in fname:
                video_key = fname.rsplit("_frame_", 1)[0]
            else:
                video_key = os.path.splitext(fname)[0]
            video_groups[video_key].append(fname)

    video_keys = list(video_groups.keys())
    if len(video_keys) == 0:
        print("[Split] No frames found to split.")
        return

    # 2. Compute video group sizes and sort
    sorted_groups = sorted(video_groups.items(), key=lambda x: len(x[1]), reverse=True)
    total_frames = sum(len(frames) for _, frames in sorted_groups)

    target_train_frames = int(total_frames * train_ratio)
    target_val_frames = int(total_frames * val_ratio)

    train_videos = set()
    val_videos = set()
    test_videos = set()

    train_count = 0
    val_count = 0
    test_count = 0

    # Distribute videos to splits targeting frame quotas
    for vkey, frame_list in sorted_groups:
        count = len(frame_list)
        if train_count + count <= target_train_frames or (train_count == 0 and count <= total_frames):
            train_videos.add(vkey)
            train_count += count
        elif val_count + count <= target_val_frames or (val_count == 0 and len(val_videos) == 0):
            val_videos.add(vkey)
            val_count += count
        else:
            test_videos.add(vkey)
            test_count += count

    # Guarantee non-empty test and val sets if >= 3 videos
    if len(test_videos) == 0 and len(train_videos) > 1:
        # Move the smallest train video to test
        smallest_train = min(train_videos, key=lambda k: len(video_groups[k]))
        train_videos.remove(smallest_train)
        test_videos.add(smallest_train)

    if len(val_videos) == 0 and len(train_videos) > 1:
        smallest_train = min(train_videos, key=lambda k: len(video_groups[k]))
        train_videos.remove(smallest_train)
        val_videos.add(smallest_train)

    print(f"[Split] Total Videos: {len(video_keys)} | Total Frames: {total_frames}")
    print(f"  Train: {len(train_videos)} videos ({sum(len(video_groups[k]) for k in train_videos)} frames)")
    print(f"  Val:   {len(val_videos)} videos ({sum(len(video_groups[k]) for k in val_videos)} frames)")
    print(f"  Test:  {len(test_videos)} videos ({sum(len(video_groups[k]) for k in test_videos)} frames)")

    # 3. Create YOLO structure
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_yolo_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_yolo_dir, "labels", split), exist_ok=True)

    counts = {"train": 0, "val": 0, "test": 0}

    # 4. Copy frames and labels
    for vkey, frame_list in video_groups.items():
        if vkey in train_videos:
            target_split = "train"
        elif vkey in val_videos:
            target_split = "val"
        else:
            target_split = "test"

        for frame_name in frame_list:
            src_img = os.path.join(frames_dir, frame_name)
            dst_img = os.path.join(output_yolo_dir, "images", target_split, frame_name)
            shutil.copy2(src_img, dst_img)

            # Copy corresponding label if present
            base_name = os.path.splitext(frame_name)[0]
            if labels_dir and os.path.exists(labels_dir):
                src_lbl = os.path.join(labels_dir, f"{base_name}.txt")
                if os.path.exists(src_lbl):
                    dst_lbl = os.path.join(output_yolo_dir, "labels", target_split, f"{base_name}.txt")
                    shutil.copy2(src_lbl, dst_lbl)

            counts[target_split] += 1

    print(f"[Split] Successfully distributed frames:")
    print(f"  Train: {counts['train']} frames")
    print(f"  Val:   {counts['val']} frames")
    print(f"  Test:  {counts['test']} frames")

    # 5. Create dataset.yaml for YOLO
    clean_path = os.path.abspath(output_yolo_dir).replace("\\", "/")
    dataset_yaml_path = os.path.join(output_yolo_dir, "dataset.yaml")
    with open(dataset_yaml_path, "w") as f:
        f.write(f"""# YOLOv8 Crowd Dataset Configuration
path: {clean_path}
train: images/train
val: images/val
test: images/test

# Classes
names:
  0: person
""")
    print(f"[Split] Created YOLO dataset config at {dataset_yaml_path}")
    return counts


if __name__ == "__main__":
    split_dataset_by_video()
