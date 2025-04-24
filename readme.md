# Vision-Based CCTV Object Detection & Logging System

This project is a vision-based CCTV system that performs real-time object detection, generates captions for detected objects, and logs the results to a CSV file. It uses YOLOv8 for object detection and BLIP (Bootstrapped Language-Image Pretraining) for generating captions.

## Features

- **Real-Time Object Detection**: Detects objects in live video feeds using YOLOv8.
- **Caption Generation**: Generates descriptive captions for detected objects using the BLIP model.
- **Data Logging**: Logs detection data (timestamp, object category, and caption) to a CSV file.
- **User-Friendly Interface**: Provides a simple GUI for starting/stopping the video feed and viewing results.

## Technologies Used

- **YOLOv8**: For object detection.
- **BLIP**: For generating captions for detected objects.
- **OpenCV**: For video processing and frame handling.
- **Tkinter**: For the graphical user interface.
- **PyTorch**: For running the BLIP model.
- **Transformers**: For loading and using the BLIP model.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/RAPTOR-sr/Vision-Based-CCTV-Object-Detection---Logging-System
   cd vision-based-cctv
   ```

## Project Structure

```
Vision-Based CCTV Object Detection & Logging System/
├── caption_generator.py   # Handles caption generation using BLIP
├── data_logger.py         # Logs detection data to a CSV file
├── main.py                # Main application with GUI
├── object_detector.py     # Handles object detection using YOLOv8
├── output/                # Directory for storing logs
│   └── detections.csv     # CSV file with logged detection data
├── yolov8n.pt             # YOLOv8 model file (not included in the repo)
└── requirements.txt       # List of required Python packages
```

## Getting Started

1. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure you have the YOLOv8 model file (`yolov8n.pt`) in the project directory.
3. Run the application:
   ```bash
   python main.py
   ```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes and push to your fork:
   ```bash
   git commit -m "Description of changes"
   git push origin feature-name
   ```
4. Open a pull request to the main repository.

## License
Copyright (c) 2025 Shivansh

This project is currently unlicensed. We intend to add an open-source license in the future.

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics) for YOLOv8.
- [Salesforce Research](https://github.com/salesforce) for the BLIP model.
- The open-source community for their contributions to PyTorch, Transformers, and other libraries used in this project.