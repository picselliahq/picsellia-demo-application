# Video Stream Application

A full-screen video stream application built with PyQt5 that provides real-time object detection using YOLOv8 and integration with Picsellia for advanced AI predictions.

## Features

- Full-screen video streaming from camera or screen capture
- Real-time object detection using YOLOv8
- Confidence threshold adjustment for predictions
- Integration with Picsellia for advanced AI predictions
- Capture and save images
- Multiple model support
- Transparent UI with customizable elements
- Prediction monitoring and display
- Logger panel for debugging and information

## Requirements

- Python 3.8 or higher
- PyQt5
- OpenCV
- Ultralytics (YOLOv8)
- Picsellia SDK
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/video-stream-app.git
cd video-stream-app
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Configure Picsellia credentials:
Create a `.env` file in the project root with your Picsellia credentials:
```
PICSELLIA_TOKEN=your_token_here
PICSELLIA_ORGANIZATION=your_organization_here
```

## Usage

1. Run the application:
```bash
python camera_app.py
```

2. Application Controls:
   - Use the dropdown menu to switch between camera and screen capture
   - Adjust the confidence threshold using the slider
   - Click the "Capture" button to save the current frame
   - Click the "Predict" button to send the current frame to Picsellia for prediction
   - Toggle the logger panel visibility using the dedicated button
   - Select different YOLO models from the dropdown menu

## Project Structure

```
.
├── camera_app.py          # Main application file
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── assets/              # Static assets (images, icons)
├── captures/            # Directory for saved images
├── models/              # Directory for YOLO models
└── .env                 # Environment variables (create this file)
```

## License

MIT License
