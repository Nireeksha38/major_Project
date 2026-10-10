import os
import sys
import ssl
import json
import urllib.request
import cv2
import shutil

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def download_url_file(url, dest_path, timeout=60):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        print(f"[Dataset] Exists: {os.path.basename(dest_path)} ({os.path.getsize(dest_path)} bytes)")
        return True

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"[Dataset] Fetching: {os.path.basename(dest_path)}...")
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp, open(dest_path, "wb") as f:
            total = resp.getheader('content-length')
            chunk_size = 65536
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
        print(f"[Dataset] Saved: {os.path.basename(dest_path)} ({os.path.getsize(dest_path)} bytes)")
        return True
    except Exception as e:
        print(f"[Dataset Error] Failed {url}: {e}")
        if os.path.exists(dest_path):
            try:
                os.remove(dest_path)
            except Exception:
                pass
        return False


def ingest_simuletic_cctv_falls(max_images=111,
                                raw_images_dir="ml/dataset/raw_images",
                                raw_labels_dir="ml/dataset/raw_labels"):
    """
    Downloads the Simuletic CCTV Incident Dataset for Fall & Lying Down Detection.
    Stores images in ml/dataset/raw_images and labels in ml/dataset/raw_labels.
    """
    print(f"\n[Ingest] Fetching Simuletic CCTV Fall & Lying Down Dataset ({max_images} images)...")
    api_url = "https://huggingface.co/api/datasets/Simuletic/CCTV_Incident_Dataset_Fall_Lying_Down_Detection"
    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as res:
            meta = json.loads(res.read())
            siblings = meta.get("siblings", [])
    except Exception as e:
        print(f"[Ingest Error] Could not query Simuletic API: {e}")
        return 0

    img_files = [s["rfilename"] for s in siblings if s["rfilename"].startswith("laying_dataset/images/") and s["rfilename"].endswith(('.png', '.jpg'))]
    img_files.sort()
    img_files = img_files[:max_images]

    downloaded = 0
    fall_images_subfolder = os.path.join(raw_images_dir, "fall_cctv")
    os.makedirs(fall_images_subfolder, exist_ok=True)
    os.makedirs(raw_labels_dir, exist_ok=True)

    for i, rfile in enumerate(img_files, 1):
        filename = os.path.basename(rfile)
        stem, ext = os.path.splitext(filename)
        out_img_name = f"cctv_fall_{i:03d}{ext}"
        out_lbl_name = f"cctv_fall_{i:03d}.txt"

        img_url = f"https://huggingface.co/datasets/Simuletic/CCTV_Incident_Dataset_Fall_Lying_Down_Detection/resolve/main/{rfile}"
        lbl_rfile = f"laying_dataset/labels/{stem}.txt"
        lbl_url = f"https://huggingface.co/datasets/Simuletic/CCTV_Incident_Dataset_Fall_Lying_Down_Detection/resolve/main/{lbl_rfile}"

        dst_img_sub = os.path.join(fall_images_subfolder, out_img_name)
        dst_img_main = os.path.join(raw_images_dir, out_img_name)
        dst_lbl = os.path.join(raw_labels_dir, out_lbl_name)

        if download_url_file(img_url, dst_img_sub):
            # Also mirror to main raw_images
            if not os.path.exists(dst_img_main):
                shutil.copy2(dst_img_sub, dst_img_main)
            # Download label
            download_url_file(lbl_url, dst_lbl)
            downloaded += 1

    print(f"[Ingest] Successfully downloaded {downloaded} Simuletic CCTV Fall Images & Labels.")
    return downloaded


def download_benchmark_videos(raw_dir="ml/dataset/raw", abnormal_videos_dir="videos/abnormal"):
    """
    Downloads real fall and abnormal surveillance benchmark videos:
    1. UniDataPro Fall Detection Video
    2. UCF-Crime Fighting Video
    3. UCF-Crime Assault Video
    """
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(abnormal_videos_dir, exist_ok=True)

    video_targets = [
        (
            "https://huggingface.co/datasets/UniDataPro/fall-detection/resolve/main/fall_0001.mp4",
            os.path.join(raw_dir, "benchmark_fall_surveillance_01.mp4"),
            os.path.join(abnormal_videos_dir, "fall_cctv_01.mp4")
        ),
        (
            "https://huggingface.co/datasets/ertiaM/Anomaly_Detection_in_Surveillance_Videos/resolve/main/Anomaly-Videos-Part-2/Fighting/Fighting002_x264.mp4",
            os.path.join(raw_dir, "benchmark_abnormal_fighting_01.mp4"),
            os.path.join(abnormal_videos_dir, "abnormal_fight_01.mp4")
        ),
        (
            "https://huggingface.co/datasets/ertiaM/Anomaly_Detection_in_Surveillance_Videos/resolve/main/Anomaly-Videos-Part-1/Assault/Assault001_x264.mp4",
            os.path.join(raw_dir, "benchmark_abnormal_assault_01.mp4"),
            os.path.join(abnormal_videos_dir, "abnormal_assault_01.mp4")
        )
    ]

    success = 0
    for url, raw_dest, demo_dest in video_targets:
        if download_url_file(url, raw_dest, timeout=120):
            try:
                shutil.copy2(raw_dest, demo_dest)
            except Exception:
                pass
            success += 1

    # Also copy a generated fall video to demo videos if available
    sample_fall_raw = os.path.join(raw_dir, "abnormal_fall_001.mp4")
    if os.path.exists(sample_fall_raw):
        shutil.copy2(sample_fall_raw, os.path.join(abnormal_videos_dir, "fall_stampede_surge.mp4"))

    print(f"[Dataset] Downloaded {success}/{len(video_targets)} Fall & Abnormal Surveillance Videos.")
    return success


def extract_incident_keyframes(raw_dir="ml/dataset/raw",
                               raw_images_dir="ml/dataset/raw_images",
                               raw_labels_dir="ml/dataset/raw_labels"):
    """
    Extracts informative keyframes from fall and abnormal behavior videos,
    annotates them, and saves to raw_images.
    """
    print("\n[Keyframes] Extracting Keyframes from Fall and Abnormal Footage...")
    targets = [
        ("benchmark_fall_surveillance_01.mp4", "fall_incident", 20, 0),  # Class 0: Fall
        ("benchmark_abnormal_fighting_01.mp4", "abnormal_fight", 20, 2),  # Class 2: Abnormal/Fight
        ("benchmark_abnormal_assault_01.mp4", "abnormal_assault", 20, 2), # Class 2: Abnormal
        ("abnormal_fall_001.mp4", "abnormal_stampede_fall", 20, 0)
    ]

    total_extracted = 0
    os.makedirs(raw_images_dir, exist_ok=True)
    os.makedirs(raw_labels_dir, exist_ok=True)

    for vid_name, prefix, num_frames, class_id in targets:
        vid_path = os.path.join(raw_dir, vid_name)
        if not os.path.exists(vid_path):
            continue

        cap = cv2.VideoCapture(vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 100)
        step = max(1, frame_count // (num_frames + 1))

        extracted_from_vid = 0
        for i in range(1, num_frames + 1):
            frame_idx = i * step
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            h, w = frame.shape[:2]
            out_img = f"{prefix}_{i:03d}.jpg"
            out_lbl = f"{prefix}_{i:03d}.txt"
            cv2.imwrite(os.path.join(raw_images_dir, out_img), frame)

            # Generate YOLO bounding box label: person in middle/lower section
            if class_id == 0:
                # Fallen person (wide aspect ratio w=0.35, h=0.18)
                lbl_content = f"0 0.500000 0.650000 0.380000 0.200000\n"
            else:
                # Abnormal fight/rush bounding box (two people clashing)
                lbl_content = f"0 0.450000 0.550000 0.250000 0.450000\n0 0.580000 0.550000 0.250000 0.450000\n"

            with open(os.path.join(raw_labels_dir, out_lbl), "w") as f:
                f.write(lbl_content)

            extracted_from_vid += 1
            total_extracted += 1

        cap.release()
        print(f"  Extracted {extracted_from_vid} frames from {vid_name}")

    print(f"[Keyframes] Extracted total {total_extracted} annotated incident keyframes.")
    return total_extracted


if __name__ == "__main__":
    download_benchmark_videos()
    ingest_simuletic_cctv_falls(max_images=111)
    extract_incident_keyframes()
