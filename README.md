# Real-Time-Crowd-Analysis-for-Resource-Management-Using-YOLOv8

A real-time AI-powered system for detecting and analyzing crowds using drone camera feeds and images. This project uses a YOLOv8 object detection model to count people from video streams and static images, visualize crowd density using a grid approach, and log data periodically for further analysis and resource allocation.

## 🚀 Features

- Real-time people detection using YOLOv8
- Grid-based crowd density analysis
- Live camera support (drone or webcam)
- Image upload for offline analysis
- Automatic logging of people count every 30 seconds
- GUI built with Tkinter for user interaction
- CSV logging for future resource planning and analytics

## 🛠️ Technologies Used

- [Python 3.x](https://www.python.org/)
- [OpenCV](https://opencv.org/)
- [Tkinter](https://wiki.python.org/moin/TkInter)
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [Pillow (PIL)](https://python-pillow.org/)
- NumPy, CSV, threading

## 📂 Project Structure
    📁 DroneBasedCrowdAnalysis/
    ├── crowd_analysis.py # Main GUI + Real-time logic
    ├── runs/
    │ └── detect/
    │ └── train/
    │ └── weights/
    │ └── best.pt # Trained YOLOv8 weights
    ├── people_count_log.csv # People count log (auto-generated)
    └── README.md # This file


## 📷 How it Works

- The webcam (or drone camera) captures real-time video.
- The frame is divided into a grid (default: 4x4), and each section is analyzed separately to better capture crowd density.
- YOLOv8 detects and counts people in each region.
- Annotated video with bounding boxes and count is displayed in a separate OpenCV window.
- Every 30 seconds, the current people count is logged with a timestamp in `people_count_log.csv`.
- Users can also upload an image to analyze crowd density from static files.

## 🖥️ GUI Features

- **Upload Image**: Detect and count people in a single image
- **Live Feed Window**: Continuously shows crowd detection and people count
- **Auto-Logging**: Saves crowd count every 30 seconds into CSV format

## 📦 Installation

## 1. Clone this repository:

    git clone https://github.com/your-username/DroneBasedCrowdAnalysis.git
    cd DroneBasedCrowdAnalysis

## 2. Install the required packages:

    pip install ultralytics opencv-python pillow numpy

## 3. Download or train your YOLOv8 model and place the best.pt file in:

    runs/detect/train/weights/best.pt

## 🧪 Running the Project

    python crowd_analysis.py

## 🛡️ Use Case
This system can assist in:

  Crowd monitoring during festivals or protest
  
  Emergency evacuation planning
  
  Optimizing deployment of resources in real-time
  
  Urban area congestion management
