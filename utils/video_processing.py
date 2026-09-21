import cv2
import os


# ==============================
# Extract Frames From Video
# ==============================

def extract_frames(video_path):

    frames_folder = "static/uploads/frames"

    os.makedirs(
        frames_folder,
        exist_ok=True
    )


    cap = cv2.VideoCapture(video_path)


    if not cap.isOpened():

        return False



    frame_count = 0


    while True:

        success, frame = cap.read()


        if not success:
            break



        frame_name = f"frame_{frame_count}.jpg"


        frame_path = os.path.join(
            frames_folder,
            frame_name
        )


        cv2.imwrite(
            frame_path,
            frame
        )


        frame_count += 1



    cap.release()


    return frame_count 