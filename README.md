# Drowsiness Detection Face Extractor

This Python script automates the process of creating a facial image dataset for training drowsiness detection models. It processes video files, analyzes frames for drowsiness cues using MediaPipe Face Mesh, and saves cropped face images classified as either "awake" or "drowsy".

## Features

-   **Batch Processing:** Processes all videos in a specified folder.
-   **Multi-Cue Detection:** Classifies drowsiness based on a combination of:
    -   **Head Pose:** Detects significant head nodding (pitch) and turning (yaw).
    -   **Eye Aspect Ratio (EAR):** Detects closed or squinting eyes.
    -   **Mouth Aspect Ratio (MAR):** Detects yawning.
-   **Robust Classification:** Uses a smoothing buffer to reduce frame-by-frame classification noise.
-   **Automatic Resuming:** Intelligently finds the last saved image number to avoid overwriting files when re-running the script.
-   **Configurable:** Easily adjust thresholds, file paths, and processing parameters at the top of the script.

## How It Works

The script iterates through video frames and performs the following steps for each:

1.  **Face Landmark Detection:** It uses the MediaPipe Face Mesh model to detect 478 facial landmarks in high detail.
2.  **Head Pose Estimation:** Using a 3D model of a face and key landmarks (nose, chin, eyes, mouth corners), it calculates the head's pitch, yaw, and roll angles via OpenCV's `solvePnP` function.
3.  **Yawn Detection:** It calculates the Mouth Aspect Ratio (MAR) by measuring the distance between vertical and horizontal mouth landmarks. A high MAR value suggests a yawn.
4.  **Eye Closure Detection:** It calculates the Eye Aspect Ratio (EAR) for both eyes. A low EAR value indicates that the eyes are closed.
5.  **Classification & Saving:**
    -   If any of the drowsiness metrics (high yaw, high pitch, high MAR, low EAR) exceed their predefined thresholds, the frame is flagged as "drowsy".
    -   The script crops the bounding box of the detected face, adds padding, and saves the image to the `drowsy` or `awake` folder.
    -   Filenames are descriptive, including the class, a unique ID, and the reason for the classification (e.g., `drowsy_000123_yaw=35.1_closed_eyes_EAR=0.19.jpg`).

## Prerequisites

-   Python 3.7+
-   OpenCV
-   NumPy
-   MediaPipe

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Install the required packages:**
    It's highly recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Place your videos:** Create a folder (e.g., `downloads`) and place all your source video files (`.mp4`, `.mov`, `.avi`, etc.) inside it.

2.  **Configure the script:** Open `face_extractor.py` and modify the parameters in the `Configuration` section as needed:
    ```python
    # Input: Folder containing videos
    video_folder = "downloads"

    # Output base directory
    output_base = "extracted_faces_drowsy"

    # Drowsiness thresholds
    yaw_threshold = 30
    pitch_down_threshold = 15
    mouth_aspect_ratio_threshold = 0.75
    eye_aspect_ratio_threshold = 0.25
    ```

3.  **Run the script:**
    ```bash
    python face_extractor.py
    ```

4.  **Check the output:** The script will create the output directory (e.g., `extracted_faces_drowsy`) with two subfolders: `awake` and `drowsy`, populated with the cropped face images. The console will log the progress.

## License

This project is licensed under the [MIT License](LICENSE) / [GPLv3 License](LICENSE). See the `LICENSE` file for details.
