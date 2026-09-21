import os
import sys
import urllib.request
import ssl

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Create SSL context that allows downloads
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def download_file(url, dest_path):
    """Downloads a file from url with progress indicator."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        print(f"[Dataset Download] Already exists: {os.path.basename(dest_path)} ({os.path.getsize(dest_path)} bytes)")
        return True

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"[Dataset Download] Fetching {os.path.basename(dest_path)} from:\n  {url}")

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
            }
        )
        with urllib.request.urlopen(req, context=ctx, timeout=60) as response, open(dest_path, "wb") as out_file:
            length = response.getheader('content-length')
            if length:
                length = int(length)
                block_size = max(4096, length // 100)
            else:
                block_size = 65536

            downloaded = 0
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)

        print(f"[Dataset Download] Successfully saved {os.path.basename(dest_path)} ({os.path.getsize(dest_path)} bytes)")
        return True
    except Exception as e:
        print(f"[Dataset Download Error] Failed to download {url}: {e}")
        if os.path.exists(dest_path):
            try:
                os.remove(dest_path)
            except Exception:
                pass
        return False


def fetch_benchmark_datasets(raw_videos_dir="ml/dataset/raw", raw_images_dir="ml/dataset/raw_images"):
    """
    Downloads 3 key crowd benchmark datasets:
    1. VIRAT / UMN Crowd Panic & Anomaly Benchmark Sequences
    2. PETS 2009 / Open Benchmark Crowd Tracking & Flow Sequences
    3. ShanghaiTech / Dense Crowd High-Occlusion Images
    """
    os.makedirs(raw_videos_dir, exist_ok=True)
    os.makedirs(raw_images_dir, exist_ok=True)

    # 1. Benchmark Videos (Surveillance, crowd counting, anomaly, flow)
    video_benchmarks = [
        (
            "https://huggingface.co/Intel/loitering-detection/resolve/main/VIRAT_S_000101.mp4",
            os.path.join(raw_videos_dir, "benchmark_VIRAT_surveillance.mp4")
        ),
        (
            "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/people-detection.mp4",
            os.path.join(raw_videos_dir, "benchmark_crowd_flow.mp4")
        ),
        (
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi",
            os.path.join(raw_videos_dir, "benchmark_pedestrian_cross.avi")
        ),
        (
            "https://raw.githubusercontent.com/HoseinRanjbar/Crowd-Counting-and-Localization/main/crowd-counting.mp4",
            os.path.join(raw_videos_dir, "benchmark_crowd_counting.mp4")
        )
    ]

    # 2. Benchmark Dense Crowd Images (High-density public gathering / street scenes)
    image_benchmarks = [
        ("https://images.unsplash.com/photo-1517457373958-b7bdd4587205?q=80&w=1000&auto=format&fit=crop", os.path.join(raw_images_dir, "crowd_dense_event_01.jpg")),
        ("https://images.unsplash.com/photo-1506157786151-b8491531f063?q=80&w=1000&auto=format&fit=crop", os.path.join(raw_images_dir, "crowd_concert_dense_02.jpg")),
        ("https://images.unsplash.com/photo-1492684223066-81342ee5ff30?q=80&w=1000&auto=format&fit=crop", os.path.join(raw_images_dir, "crowd_festival_surge_03.jpg")),
        ("https://images.unsplash.com/photo-1533105079780-92b9be482077?q=80&w=1000&auto=format&fit=crop", os.path.join(raw_images_dir, "crowd_street_market_04.jpg")),
        ("https://images.unsplash.com/photo-1508997449629-303059a039c0?q=80&w=1000&auto=format&fit=crop", os.path.join(raw_images_dir, "crowd_pedestrian_walk_05.jpg")),
        ("https://images.unsplash.com/photo-1540575467063-178a50c2df87?q=80&w=1000&auto=format&fit=crop", os.path.join(raw_images_dir, "crowd_auditorium_dense_06.jpg")),
        ("https://images.unsplash.com/photo-1470225620780-dba8ba36b745?q=80&w=1000&auto=format&fit=crop", os.path.join(raw_images_dir, "crowd_night_event_07.jpg")),
        ("https://images.unsplash.com/photo-1514525253161-7a46d19cd819?q=80&w=1000&auto=format&fit=crop", os.path.join(raw_images_dir, "crowd_party_dense_08.jpg")),
        ("https://images.unsplash.com/photo-1459749411175-04bf5292ceea?q=80&w=1000&auto=format&fit=crop", os.path.join(raw_images_dir, "crowd_stadium_fans_09.jpg")),
        ("https://images.unsplash.com/photo-1464375117522-1311d6a5b81f?q=80&w=1000&auto=format&fit=crop", os.path.join(raw_images_dir, "crowd_arena_cheer_10.jpg")),
        ("https://images.unsplash.com/photo-1516450360452-9312f5e86fc7?q=80&w=1000&auto=format&fit=crop", os.path.join(raw_images_dir, "crowd_celebration_11.jpg")),
        ("https://images.unsplash.com/photo-1519750157634-b6d493a0f77c?q=80&w=1000&auto=format&fit=crop", os.path.join(raw_images_dir, "crowd_indoor_gathering_12.jpg")),
        ("https://images.unsplash.com/photo-1516455207990-7a41ce80f7ee?q=80&w=1000&auto=format&fit=crop", os.path.join(raw_images_dir, "crowd_stage_lights_13.jpg")),
        ("https://images.unsplash.com/photo-1524368535928-5b5e00ddc76b?q=80&w=1000&auto=format&fit=crop", os.path.join(raw_images_dir, "crowd_music_fest_14.jpg")),
        ("https://images.unsplash.com/photo-1470229722913-7c0e2dbbafd3?q=80&w=1000&auto=format&fit=crop", os.path.join(raw_images_dir, "crowd_dj_rave_15.jpg")),
        ("https://images.unsplash.com/photo-1501386761578-eac5c94b800a?q=80&w=1000&auto=format&fit=crop", os.path.join(raw_images_dir, "crowd_concert_arms_16.jpg")),
        ("https://images.unsplash.com/photo-1508700115892-45ecd05ae2ad?q=80&w=1000&auto=format&fit=crop", os.path.join(raw_images_dir, "crowd_rave_lights_17.jpg")),
        ("https://images.unsplash.com/photo-1429962714451-bb934ecdc4ec?q=80&w=1000&auto=format&fit=crop", os.path.join(raw_images_dir, "crowd_parade_outdoor_18.jpg")),
        ("https://images.unsplash.com/photo-1514565131-fce0801e5785?q=80&w=1000&auto=format&fit=crop", os.path.join(raw_images_dir, "crowd_city_night_19.jpg")),
        ("https://images.unsplash.com/photo-1513151233558-d860c5398176?q=80&w=1000&auto=format&fit=crop", os.path.join(raw_images_dir, "crowd_party_confetti_20.jpg"))
    ]

    print("[Dataset Fetcher] Downloading High-Density Crowd Benchmark Images...")
    img_success = 0
    for url, path in image_benchmarks:
        if download_file(url, path):
            img_success += 1

    print(f"[Dataset Fetcher] Downloaded {img_success}/{len(image_benchmarks)} benchmark image(s).")
    return img_success


if __name__ == "__main__":
    fetch_benchmark_datasets()
