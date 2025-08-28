# Drowsiness Detection Face Extractor

![License](https://img.shields.io/github/license/jodave911/Drowsiness-Detection-Face-Extractor)
![Python Version](https://img.shields.io/badge/python-3.7+-blue.svg)

This project provides a powerful tool for building custom datasets for machine learning-based drowsiness detection. The script processes video files, leveraging Google's MediaPipe Face Mesh for high-fidelity facial landmark detection. It analyzes each frame to calculate key drowsiness indicators, including Eye Aspect Ratio (EAR) for eye closure, Mouth Aspect Ratio (MAR) for yawning, and 3D head pose for nodding. Based on configurable thresholds, it automatically classifies faces as "awake" or "drowsy," crops them from the frame, and saves them into organized folders, streamlining the data collection pipeline.

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

1.  **Face Landmark Detection:** It uses the MediaPipe Face Mesh model to detect 478 facial landmarks.
2.  **Head Pose Estimation:** It calculates the head's pitch, yaw, and roll angles using OpenCV's `solvePnP` function.
3.  **Yawn & Eye Closure Detection:** It calculates the Mouth Aspect Ratio (MAR) and Eye Aspect Ratio (EAR) to detect signs of fatigue.
4.  **Classification & Saving:** Based on a set of predefined thresholds, the frame is flagged as "drowsy" or "awake." The script then crops the detected face and saves it to the corresponding labeled folder.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/jodave911/Drowsiness-Detection-Face-Extractor.git](https://github.com/jodave911/Drowsiness-Detection-Face-Extractor.git)
    cd Drowsiness-Detection-Face-Extractor
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Place your videos:** Create a folder (e.g., `downloads` by default) and place your source video files (`.mp4`, `.mov`, etc.) inside it.

2.  **Configure the script:** Open `face_extractor.py` and modify the parameters in the `SCRIPT CONFIGURATION` section as needed.

3.  **Run the script:**
    ```bash
    python face_extractor.py
    ```

4.  **Check the output:** The script will create the output directory (e.g., `extracted_faces_drowsy`) with two subfolders, `awake` and `drowsy`, populated with the cropped face images. Progress will be logged to the console.

## Project Purpose & Status

The primary goal of this project is to provide a fast and simple tool for creating facial image datasets for drowsiness detection. It is designed to process publicly available videos and automatically extract and label faces based on heuristic cues, significantly speeding up the data collection phase for machine learning projects.

### Current Accuracy

The current implementation uses a **heuristic-based approach**, relying on fixed thresholds for Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and head pose angles.

-   **Effectiveness:** This method is effective for generating a large, baseline dataset quickly. It successfully identifies obvious drowsiness events like yawning, significant head nods, and prolonged eye closure.
-   **Limitations:** The accuracy is considered **functional but not robust**. Since it relies on static thresholds, its performance can vary based on video quality, lighting, camera angle, and individual differences in facial structure. It is not designed to be a high-precision, real-time detection system in its current state, but rather an efficient data extraction tool.

## Future Improvements (TODO)

The following is a list of planned improvements to enhance the robustness and accuracy of the detection and classification logic.

-   [ ] **Improve Face Detection with a YOLO Model:**
    -   While MediaPipe is effective, integrating a dedicated face detector like **YOLOv8** or **RetinaFace** could improve the initial detection rate, especially for non-frontal faces or in challenging lighting.

-   [ ] **Replace Heuristics with a Machine Learning Classifier:**
    -   The next major step is to replace the threshold-based logic with a trained ML model (e.g., SVM, Gradient Boosting, or a small neural network). This model would use the calculated EAR, MAR, and pose angles as input features to provide a more nuanced classification.

-   [ ] **Train an End-to-End Deep Learning Model:**
    -   For maximum performance, the final goal is to train an end-to-end CNN model that takes the cropped face image as direct input and outputs a drowsiness probability.

-   [ ] **Add Configuration File:**
    -   Move all hard-coded parameters into an external configuration file (e.g., `config.yaml`) for easier tuning without modifying the source code.

## Acknowledgements

The core logic of this script was developed by the author. To improve maintainability and readability, Google's Gemini was utilized for the following enhancements:

-   Refactoring the code structure into logical sections.
-   Adding comprehensive docstrings and type hints.
-   Applying Python (PEP 8) conventions and best practices.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

Copyright (c) 2025, jodave911
