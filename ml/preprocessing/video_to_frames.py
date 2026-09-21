import os
import cv2
import argparse


def extract_frames_from_video(video_path, output_dir, frame_skip=5, prefix="frame"):
    """
    Extracts frames from a video with a configurable frame skip.
    video_path: path to input video
    output_dir: directory where extracted frames will be saved
    frame_skip: sample every N-th frame (default=5)
    """
    if not os.path.exists(video_path):
        print(f"[Error] Video not found: {video_path}")
        return 0

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Error] Could not open video: {video_path}")
        return 0

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved_count = 0
    frame_idx = 0

    print(f"[FrameExtractor] Processing '{video_name}' ({total_frames} frames, skip={frame_skip})...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            frame_filename = f"{video_name}_{prefix}_{saved_count:05d}.jpg"
            save_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(save_path, frame)
            saved_count += 1

        frame_idx += 1

    cap.release()
    print(f"[FrameExtractor] Extracted {saved_count} frames to {output_dir}")
    return saved_count


def process_dataset_directory(raw_dir="ml/dataset/raw", output_dir="ml/dataset/extracted_frames", frame_skip=5):
    """Processes all videos found in raw dataset directory."""
    if not os.path.exists(raw_dir):
        print(f"[Info] Raw directory '{raw_dir}' does not exist yet.")
        return 0

    total_extracted = 0
    for root, _, files in os.walk(raw_dir):
        for f in files:
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                vpath = os.path.join(root, f)
                total_extracted += extract_frames_from_video(vpath, output_dir, frame_skip)

    return total_extracted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from crowd videos.")
    parser.add_argument("--video", type=str, help="Path to single video file.")
    parser.add_argument("--raw_dir", type=str, default="ml/dataset/raw", help="Directory of raw videos.")
    parser.add_argument("--output_dir", type=str, default="ml/dataset/extracted_frames", help="Output directory.")
    parser.add_argument("--skip", type=int, default=5, help="Frame sample skip interval.")
    args = parser.parse_args()

    if args.video:
        extract_frames_from_video(args.video, args.output_dir, args.skip)
    else:
        process_dataset_directory(args.raw_dir, args.output_dir, args.skip)
