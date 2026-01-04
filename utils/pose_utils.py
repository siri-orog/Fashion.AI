import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

pose_detector = mp_pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False)

def detect_pose_landmarks(image_rgb):
    """
    Returns pose landmarks (list) in normalized coords (x,y) or None.
    image_rgb: RGB numpy image
    """
    results = pose_detector.process(image_rgb)
    if results.pose_landmarks:
        return results.pose_landmarks.landmark
    return None

def get_torso_bbox_from_landmarks(landmarks, img_w, img_h, scale=1.2):
    """
    Given MediaPipe landmarks, compute torso bounding box (x,y,w,h).
    Uses shoulders (11,12) and hips (23,24).
    Returns bbox in pixel coords: (x, y, w, h) or None if landmarks missing.
    """
    try:
        # MediaPipe landmark indices
        l_sh = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_sh = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

        # convert normalized to pixel coords
        sh_xs = np.array([l_sh.x, r_sh.x]) * img_w
        sh_ys = np.array([l_sh.y, r_sh.y]) * img_h
        hip_xs = np.array([l_hip.x, r_hip.x]) * img_w
        hip_ys = np.array([l_hip.y, r_hip.y]) * img_h

        x_min = min(sh_xs.min(), hip_xs.min())
        x_max = max(sh_xs.max(), hip_xs.max())
        y_min = min(sh_ys.min(), hip_ys.min())
        y_max = max(sh_ys.max(), hip_ys.max())

        # expand bbox a bit
        w = (x_max - x_min) * scale
        h = (y_max - y_min) * scale
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2

        x = int(cx - w / 2)
        y = int(cy - h / 2)
        w = int(w)
        h = int(h)

        return (x, y, w, h)
    except Exception:
        return None
