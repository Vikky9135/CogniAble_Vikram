# Human Detection and Tracking with YOLOv8 and DeepSORT

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Explanation of Key Sections](#explanation-of-key-sections)
  - [1. Importing Necessary Modules](#1-importing-necessary-modules)
  - [2. Video Download from YouTube](#2-video-download-from-youtube)
  - [3. Running Video Download Function](#3-running-video-download-function)
  - [4. Initializing YOLOv8 Model](#4-initializing-yolov8-model)
  
## Overview
This project demonstrates human detection and tracking using the YOLOv8 object detection algorithm combined with DeepSORT for multi-object tracking. The system can detect humans in videos and track their movements across frames.

## Features
- YOLOv8 for real-time object detection
- DeepSORT for tracking objects (humans) between frames
- Video input from local storage or YouTube
- Outputs the processed video with bounding boxes and tracking IDs

## Prerequisites
Before running the project, ensure you have the following dependencies:

- Python 3.x
- OpenCV
- Ultralytics YOLOv8
- DeepSORT
- pytube (for YouTube video downloads)

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/human-detection-tracking.git
    cd human-detection-tracking
    ```

2. Install the required Python libraries:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
To run the project, follow these steps:

1. Download a video from YouTube using the provided function or use a local video file.
2. Initialize the YOLOv8 model and run the tracking pipeline.

### Example Command:
```bash
python main.py --video input_video.mp4
