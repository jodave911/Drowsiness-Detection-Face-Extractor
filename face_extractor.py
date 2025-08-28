# -*- coding: utf-8 -*-
"""
Drowsiness Detection Face Extractor

This script processes video files to create a labeled facial image dataset for
training drowsiness detection models. It uses MediaPipe Face Mesh to detect
facial landmarks and classifies drowsiness based on head pose, eye aspect ratio (EAR),
and mouth aspect ratio (MAR).

Note: This script's readability and structure were enhanced with assistance
      from Google's Gemini. The core functionality remains the author's original work.

      
Author: Joshua David
Date: 28 August 2025
"""

import cv2
import numpy as np
import os
import logging
import glob
from collections import deque
from typing import List, Tuple, Dict, Deque

# Third-party libraries
import mediapipe as mp

# ----------------------------
# 1. SCRIPT CONFIGURATION
# ----------------------------

# --- Input/Output Paths ---
VIDEO_FOLDER = "downloads"
VIDEO_EXTENSIONS = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.webm']
OUTPUT_BASE_DIR = "extracted_faces_drowsy"

# --- Processing Parameters ---
FRAME_SKIP = 8  # Process every Nth frame to speed up analysis.
MAX_FACES_PER_VIDEO = 1500  # Stop processing a video after extracting this many faces.

# --- Drowsiness Detection Thresholds ---
YAW_THRESHOLD = 30.0  # Head turn (left/right) in degrees.
PITCH_DOWN_THRESHOLD = 15.0  # Head nod (downwards) in degrees.
MOUTH_ASPECT_RATIO_THRESHOLD = 0.75  # Threshold for detecting a yawn.
EYE_ASPECT_RATIO_THRESHOLD = 0.25  # Threshold for detecting closed eyes.
SMOOTHING_BUFFER_SIZE = 5 # Number of frames to average for smoothing drowsiness detection.

# ----------------------------
# 2. GLOBAL INITIALIZATION
# ----------------------------

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Output Directory Setup ---
OUTPUT_FOLDERS = {
    "awake": os.path.join(OUTPUT_BASE_DIR, "awake"),
    "drowsy": os.path.join(OUTPUT_BASE_DIR, "drowsy")
}
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
for path in OUTPUT_FOLDERS.values():
    os.makedirs(path, exist_ok=True)

# --- MediaPipe Face Mesh ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Landmark Indices for Head Pose and MAR ---
class LandmarkIndices:
    NOSE_TIP = 1
    CHIN = 152
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263
    LEFT_MOUTH_CORNER = 61
    RIGHT_MOUTH_CORNER = 291
    MOUTH_TOP = 13
    MOUTH_BOTTOM = 14

# --- Landmark Indices for Eye Aspect Ratio (EAR) ---
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# --- 3D Model Points for Head Pose Estimation ---
# These points correspond to the landmark indices above.
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye outer corner
    (225.0, 170.0, -135.0),      # Right eye outer corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
])

# ----------------------------
# 3. HELPER FUNCTIONS
# ----------------------------

def get_next_available_counter(folder_path: str, label: str) -> int:
    """
    Finds the next available integer for a filename to avoid overwrites.
    Scans for files like 'label_000001.jpg' and returns the next number.
    """
    pattern = os.path.join(folder_path, f"{label}_*.jpg")
    existing_files = glob.glob(pattern)
    if not existing_files:
        return 1
    
    max_counter = 0
    for file_path in existing_files:
        filename = os.path.basename(file_path)
        try:
            # Assumes format "label_counter_..."
            counter_str = filename.split('_')[1]
            counter = int(counter_str)
            if counter > max_counter:
                max_counter = counter
        except (ValueError, IndexError):
            continue
            
    return max_counter + 1

def calculate_eye_aspect_ratio(eye_landmarks: np.ndarray) -> float:
    """Calculates the Eye Aspect Ratio (EAR) from 6 eye landmarks."""
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear

def rotation_matrix_to_euler_angles(rmat: np.ndarray) -> Tuple[float, float, float]:
    """Converts a rotation matrix to Euler angles (pitch, yaw, roll) in degrees."""
    sy = np.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(rmat[2, 1], rmat[2, 2])
        y = np.arctan2(-rmat[2, 0], sy)
        z = np.arctan2(rmat[1, 0], rmat[0, 0])
    else:
        x = np.arctan2(-rmat[1, 2], rmat[1, 1])
        y = np.arctan2(-rmat[2, 0], sy)
        z = 0

    # Convert radians to degrees
    pitch, yaw, roll = np.degrees([x, y, z])
    return pitch, yaw, roll

# ----------------------------
# 4. CORE PROCESSING LOGIC
# ----------------------------

def process_video(video_path: str, global_counters: Dict[str, int]):
    """
    Processes a single video file to extract and classify faces.
    
    Args:
        video_path (str): The path to the video file.
        global_counters (dict): A dictionary tracking the count of saved images.
    """
    logger.info(f"🎥 Processing video: {os.path.basename(video_path)}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return

    # --- 1. Initialization for this video ---
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    # Camera internals for solvePnP
    focal_length = w
    camera_center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, camera_center[0]],
        [0, focal_length, camera_center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1)) # Assuming no lens distortion

    frame_idx = 0
    faces_saved_this_video = 0
    drowsiness_buffer: Deque[bool] = deque(maxlen=SMOOTHING_BUFFER_SIZE)
    initial_counts = global_counters.copy()

    # --- 2. Frame-by-frame processing loop ---
    while faces_saved_this_video < MAX_FACES_PER_VIDEO:
        ret, frame = cap.read()
        if not ret:
            logger.info("End of video reached.")
            break

        if frame_idx % FRAME_SKIP != 0:
            frame_idx += 1
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        frame_idx += 1

        if not results.multi_face_landmarks:
            continue

        # --- 3. Landmark and Drowsiness Analysis ---
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            is_drowsy = False
            reasons = []

            # Head Pose Estimation
            try:
                pose_indices = [
                    LandmarkIndices.NOSE_TIP, LandmarkIndices.CHIN,
                    LandmarkIndices.LEFT_EYE_OUTER, LandmarkIndices.RIGHT_EYE_OUTER,
                    LandmarkIndices.LEFT_MOUTH_CORNER, LandmarkIndices.RIGHT_MOUTH_CORNER
                ]
                image_points = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in pose_indices], dtype="double")
                
                success, rot_vec, _ = cv2.solvePnP(
                    MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP
                )
                
                if success:
                    rmat, _ = cv2.Rodrigues(rot_vec)
                    pitch, yaw, _ = rotation_matrix_to_euler_angles(rmat)
                    if abs(yaw) > YAW_THRESHOLD:
                        is_drowsy = True
                        reasons.append(f"yaw={abs(yaw):.1f}")
                    if pitch > PITCH_DOWN_THRESHOLD:
                        is_drowsy = True
                        reasons.append(f"pitch_down={pitch:.1f}")
            except Exception as e:
                logger.warning(f"Head pose calculation failed: {e}")

            # Mouth Aspect Ratio (Yawn Detection)
            try:
                mouth_pts = {
                    'top': (landmarks[LandmarkIndices.MOUTH_TOP].x * w, landmarks[LandmarkIndices.MOUTH_TOP].y * h),
                    'bottom': (landmarks[LandmarkIndices.MOUTH_BOTTOM].x * w, landmarks[LandmarkIndices.MOUTH_BOTTOM].y * h),
                    'left': (landmarks[LandmarkIndices.LEFT_MOUTH_CORNER].x * w, landmarks[LandmarkIndices.LEFT_MOUTH_CORNER].y * h),
                    'right': (landmarks[LandmarkIndices.RIGHT_MOUTH_CORNER].x * w, landmarks[LandmarkIndices.RIGHT_MOUTH_CORNER].y * h)
                }
                mouth_height = np.linalg.norm(np.array(mouth_pts['top']) - np.array(mouth_pts['bottom']))
                mouth_width = np.linalg.norm(np.array(mouth_pts['left']) - np.array(mouth_pts['right']))
                mar = mouth_height / mouth_width if mouth_width > 0 else 0
                if mar > MOUTH_ASPECT_RATIO_THRESHOLD:
                    is_drowsy = True
                    reasons.append(f"yawn_MAR={mar:.2f}")
            except Exception as e:
                logger.warning(f"MAR calculation failed: {e}")

            # Eye Aspect Ratio (Eye Closure Detection)
            try:
                left_eye_pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in LEFT_EYE_INDICES])
                right_eye_pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in RIGHT_EYE_INDICES])
                ear = (calculate_eye_aspect_ratio(left_eye_pts) + calculate_eye_aspect_ratio(right_eye_pts)) / 2.0
                if ear < EYE_ASPECT_RATIO_THRESHOLD:
                    is_drowsy = True
                    reasons.append(f"closed_eyes_EAR={ear:.2f}")
            except Exception as e:
                logger.warning(f"EAR calculation failed: {e}")
            
            # --- 4. Crop and Save Face ---
            try:
                drowsiness_buffer.append(is_drowsy)
                # Classify based on majority of recent frames
                is_smoothed_drowsy = sum(drowsiness_buffer) > len(drowsiness_buffer) / 2
                
                label = "drowsy" if is_smoothed_drowsy else "awake"
                reason_str = "_".join(reasons) if is_smoothed_drowsy and reasons else "awake"

                # Get bounding box with padding
                x_coords = [int(lm.x * w) for lm in landmarks]
                y_coords = [int(lm.y * h) for lm in landmarks]
                x_min, x_max = max(0, min(x_coords)), min(w, max(x_coords))
                y_min, y_max = max(0, min(y_coords)), min(h, max(y_coords))
                pad_w = int(0.2 * (x_max - x_min))
                pad_h = int(0.2 * (y_max - y_min))
                
                face_img = frame[
                    max(0, y_min - pad_h):min(h, y_max + pad_h),
                    max(0, x_min - pad_w):min(w, x_max + pad_w)
                ]

                if face_img.size > 0:
                    current_count = global_counters[label]
                    filename = f"{label}_{current_count:06d}_{reason_str}.jpg"
                    filepath = os.path.join(OUTPUT_FOLDERS[label], filename)
                    
                    cv2.imwrite(filepath, face_img)
                    
                    global_counters[label] += 1
                    faces_saved_this_video += 1
            except Exception as e:
                logger.error(f"Face crop/save failed: {e}")

    # --- 5. Cleanup and Summary for this video ---
    cap.release()
    new_awake = global_counters['awake'] - initial_counts['awake']
    new_drowsy = global_counters['drowsy'] - initial_counts['drowsy']
    logger.info(f"✅ Video complete. Saved: {new_awake} awake, {new_drowsy} drowsy faces.")

# ----------------------------
# 5. MAIN EXECUTION
# ----------------------------

def main():
    """
    Main function to run the batch processing of videos.
    """
    # Find all video files in the target directory
    video_files = []
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(glob.glob(os.path.join(VIDEO_FOLDER, ext)))
        video_files.extend(glob.glob(os.path.join(VIDEO_FOLDER, ext.upper())))

    if not video_files:
        logger.error(f"No video files found in '{VIDEO_FOLDER}'. Please check the path and extensions.")
        return

    logger.info(f"Found {len(video_files)} video(s). Starting batch processing...")

    # Initialize global counters by checking existing files
    global_counters = {
        "awake": get_next_available_counter(OUTPUT_FOLDERS["awake"], "awake"),
        "drowsy": get_next_available_counter(OUTPUT_FOLDERS["drowsy"], "drowsy")
    }
    logger.info(f"📁 Starting counters at - Awake: {global_counters['awake']}, Drowsy: {global_counters['drowsy']}")

    # Process each video
    for video_path in video_files:
        try:
            process_video(video_path, global_counters)
        except Exception as e:
            logger.error(f"An unexpected error occurred while processing {video_path}: {e}", exc_info=True)

    # Final summary
    logger.info("🎉 Batch processing completed!")
    logger.info("Final counts:")
    logger.info(f"  - Awake: {global_counters['awake'] - 1}")
    logger.info(f"  - Drowsy: {global_counters['drowsy'] - 1}")


if __name__ == "__main__":
    main()
