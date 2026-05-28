"""
GStreamer helper utilities.
- gst_discover: probe RTSP stream dimensions via ffprobe
- start_gstreamer: launch gst-launch-1.0 subprocess piping raw BGR to stdout
"""

import json
import subprocess


def gst_discover(url: str, timeout: int = 15) -> tuple[int, int, float]:
    """Probe stream dimensions using ffprobe."""
    try:
        cmd = [
            'ffprobe', '-rtsp_transport', 'tcp',
            '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,r_frame_rate',
            '-of', 'json', url
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        info = json.loads(r.stdout)
        s = info['streams'][0]
        w, h = int(s['width']), int(s['height'])
        num, den = map(int, s['r_frame_rate'].split('/'))
        fps = num / max(den, 1)
        return w, h, fps
    except Exception as e:
        print(f"[probe] ffprobe failed: {e}. Using defaults 1920x1080@25fps.")
        return 1920, 1080, 25.0


def start_gstreamer(url: str, width: int, height: int,
                    max_fps: int = 15) -> subprocess.Popen:
    """
    Start gst-launch-1.0 pipeline that pipes raw BGR24 frames to stdout.
    Each element/property is a separate arg for gst-launch-1.0.
    """
    cmd = [
        'gst-launch-1.0', '-q', '-e',
        'rtspsrc', f'location={url}', 'latency=100', 'protocols=tcp', '!',
        'rtph264depay', '!',
        'h264parse', '!',
        'vtdec', '!',
        'videoconvert', '!',
        'videorate', '!',
        f'video/x-raw,format=BGR,framerate={max_fps}/1', '!',
        'filesink', 'location=/dev/stdout', 'sync=false'
    ]
    print(f"[GStreamer] cmd: {' '.join(cmd)}")
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=width * height * 3 * 2
    )
