import cv2
import numpy as np
from config import Config


class OpticalFlowAnalyzer:
    """
    Computes dense optical flow between consecutive frames using Farneback's algorithm.
    Extracts magnitude, angle, motion intensity, and flow vectors.
    """
    def __init__(self, downscale_factor=None):
        self.downscale_factor = downscale_factor or Config.OPTICAL_FLOW_DOWNSCALE
        self.prev_gray = None
        self.prev_time = None

    def reset(self):
        self.prev_gray = None

    def compute_flow(self, frame):
        """
        Computes optical flow on the provided BGR frame.
        Returns:
            flow: optical flow array (H_down, W_down, 2)
            magnitude: magnitude array
            angle: direction angle array in degrees [0, 360)
            motion_intensity: scalar normalized motion score [0.0, 1.0]
            flow_summary: dict of key metrics
        """
        if frame is None:
            return None, None, None, 0.0, {}

        # Downscale and convert to grayscale for fast CPU computation
        h, w = frame.shape[:2]
        new_w = max(160, int(w * self.downscale_factor))
        new_h = max(120, int(h * self.downscale_factor))
        small_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        curr_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.GaussianBlur(curr_gray, (3, 3), 0)

        if self.prev_gray is None:
            self.prev_gray = curr_gray
            return (
                np.zeros((new_h, new_w, 2), dtype=np.float32),
                np.zeros((new_h, new_w), dtype=np.float32),
                np.zeros((new_h, new_w), dtype=np.float32),
                0.0,
                {"mean_magnitude": 0.0, "max_magnitude": 0.0, "motion_ratio": 0.0}
            )

        # Fast Farneback Optical Flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray,
            curr_gray,
            None,
            pyr_scale=0.5,
            levels=2,
            winsize=13,
            iterations=2,
            poly_n=5,
            poly_sigma=1.1,
            flags=0
        )

        self.prev_gray = curr_gray

        # Polar coordinates: magnitude and angle
        magnitude, angle_rad = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        angle_deg = np.rad2deg(angle_rad)

        # Filter noise threshold
        motion_mask = magnitude > 0.8
        moving_magnitudes = magnitude[motion_mask]

        if len(moving_magnitudes) > 0:
            mean_mag = float(np.mean(moving_magnitudes))
            max_mag = float(np.max(moving_magnitudes))
            motion_ratio = float(np.sum(motion_mask) / (new_h * new_w))
        else:
            mean_mag = 0.0
            max_mag = 0.0
            motion_ratio = 0.0

        # Normalized motion intensity [0.0, 1.0] (capped at 10.0 magnitude)
        normalized_intensity = min(1.0, (mean_mag / 8.0) * 0.6 + motion_ratio * 0.4)

        flow_summary = {
            "mean_magnitude": mean_mag,
            "max_magnitude": max_mag,
            "motion_ratio": motion_ratio,
            "normalized_intensity": float(normalized_intensity)
        }

        return flow, magnitude, angle_deg, float(normalized_intensity), flow_summary

    def draw_flow_arrows(self, frame, flow, step=30, color=(0, 255, 255)):
        """Draws sparse optical flow vectors on frame for visual inspection."""
        if flow is None:
            return frame

        h, w = frame.shape[:2]
        fh, fw = flow.shape[:2]
        scale_x = w / fw
        scale_y = h / fh

        vis_frame = frame.copy()
        for y in range(0, fh, step):
            for x in range(0, fw, step):
                fx, fy = flow[y, x]
                if np.sqrt(fx*fx + fy*fy) > 1.0:
                    start_pt = (int(x * scale_x), int(y * scale_y))
                    end_pt = (int((x + fx * 2) * scale_x), int((y + fy * 2) * scale_y))
                    cv2.arrowedLine(vis_frame, start_pt, end_pt, color, 1, tipLength=0.3)

        return vis_frame
