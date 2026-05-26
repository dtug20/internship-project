# ATIN - RTSP Stream Detection

Real-time object detection and OCR from RTSP video streams using YOLOv8.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirement.txt
   ```

2. **Install system dependencies:**
   - **FFmpeg** (for connect_ffmpeg.py):
     ```bash
     # macOS
     brew install ffmpeg
     ```
   - **GStreamer** (for detect_gstreamer.py):
     ```bash
     # macOS
     brew install gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad
     ```

## Run

### Option 1: FFmpeg-based RTSP decoder
```bash
python3 connect_ffmpeg.py
```

### Option 2: GStreamer-based RTSP decoder
```bash
python3 detect_gstreamer.py
```

**Note:** Update the RTSP URL in the scripts before running (default: `rtsp://192.168.1.198:28537/den33`)

## Requirements

- Python 3.8+
- FFmpeg or GStreamer
- PyQt5
- OpenCV
- YOLOv8
- EasyOCR
