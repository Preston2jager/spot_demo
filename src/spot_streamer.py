import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, render_template_string
from bosdyn.api import image_pb2
from bosdyn.client.image import build_image_request

from .spot_tracker import SpotTracker

_DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Spot Live Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; background: #121212; color: white; margin: 0; padding: 20px; }
        .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 20px; }
        .tabs { display: flex; gap: 10px; }
        .tab-btn { background: #333; color: white; border: none; padding: 10px 20px; cursor: pointer; border-radius: 5px; font-size: 16px; }
        .tab-btn.active { background: #4CAF50; font-weight: bold; }
        .tab-btn:hover:not(.active) { background: #555; }
        .tab-content { display: none; width: 100%; text-align: center; }
        .tab-content.active { display: block; }
        img.video-stream { max-width: 100%; max-height: 80vh; border: 2px solid #333; border-radius: 5px; background: #000; }
    </style>
</head>
<body>
    <div class="header">
        <h2>Spot Live Dashboard</h2>
        <div class="tabs">
            <button class="tab-btn active" onclick="showTab('tab-hand', this)">Hand Camera</button>
            <button class="tab-btn" onclick="showTab('tab-grid', this)">2x2 Cameras</button>
        </div>
    </div>
    <div id="tab-hand" class="tab-content active">
        <img class="video-stream" src="/video_feed/hand" alt="Hand Camera Stream">
    </div>
    <div id="tab-grid" class="tab-content">
        <img class="video-stream" src="/video_feed/grid" alt="2x2 Grid Stream">
    </div>
    <script>
        function showTab(tabId, btn) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            btn.classList.add('active');
        }
    </script>
</body>
</html>
"""


class SpotStreamer:
    """Flask-based live dashboard streaming hand and fisheye camera feeds."""

    @SpotTracker("Initialise Streamer")
    def __init__(self, robot: object, host: str = "0.0.0.0", web_port: int = 5555) -> None:
        """Create the Flask app, thread locks, and register routes."""
        self.robot = robot
        self.host = host
        self.web_port = web_port
        self._streaming = False
        self.frames: dict = {"hand": None, "grid": None}
        self.lock = threading.Lock()
        self.app = Flask("spot_web_dashboard")
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Register the dashboard index and per-stream MJPEG endpoints."""
        @self.app.route("/")
        def index() -> str:
            return render_template_string(_DASHBOARD_HTML)

        @self.app.route("/video_feed/<stream_type>")
        def video_feed(stream_type: str) -> Response:
            if stream_type not in ("hand", "grid"):
                return Response("Invalid stream type", status=400)
            return Response(
                self.generate_frames(stream_type),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

    def generate_frames(self, stream_type: str):
        """Yield MJPEG-encoded frames for the given stream type while streaming is active."""
        while self._streaming:
            with self.lock:
                frame = self.frames[stream_type]
            if frame is not None:
                ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                    )
            time.sleep(0.05)

    @SpotTracker("Stream Loop Task", exit_on_fail=False)
    def _stream_loop(self) -> None:
        """Continuously capture images from all cameras and write to the shared frame buffer."""
        image_client = self.robot.ensure_client("image")
        source_names = [
            "hand_color_image",
            "left_fisheye_image",
            "right_fisheye_image",
            "back_fisheye_image",
        ]
        W_CELL, H_CELL = 640, 480
        W_CELL_2, H_CELL_2 = 1280, 960
        reqs = [
            build_image_request(
                src, pixel_format=image_pb2.Image.PIXEL_FORMAT_RGB_U8, quality_percent=70
            )
            for src in source_names
        ]
        empty_cell = np.zeros((H_CELL, W_CELL, 3), dtype=np.uint8)
        empty_cell_2 = np.zeros((H_CELL_2, W_CELL_2, 3), dtype=np.uint8)

        while self._streaming:
            try:
                responses = image_client.get_image(reqs)
                img_map: dict = {}
                for res in responses:
                    source_name = res.source.name
                    if res.status == image_pb2.ImageResponse.STATUS_OK:
                        arr = np.frombuffer(res.shot.image.data, dtype=np.uint8)
                        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if decoded is not None:
                            rotated = decoded
                            if source_name == "right_fisheye_image":
                                rotated = cv2.rotate(decoded, cv2.ROTATE_180)
                            final_img = cv2.resize(rotated, (W_CELL, H_CELL))
                            label = source_name.replace("_image", "").upper()
                            cv2.putText(
                                final_img, label, (15, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
                            )
                            img_map[source_name] = final_img
                current_hand = img_map.get("hand_color_image", empty_cell_2)
                row1 = np.hstack([
                    img_map.get("hand_color_image", empty_cell),
                    img_map.get("back_fisheye_image", empty_cell),
                ])
                row2 = np.hstack([
                    img_map.get("left_fisheye_image", empty_cell),
                    img_map.get("right_fisheye_image", empty_cell),
                ])
                current_grid = np.vstack([row1, row2])
                with self.lock:
                    self.frames["hand"] = current_hand
                    self.frames["grid"] = current_grid
                time.sleep(0.05)
            except Exception as e:
                print(f"Stream loop error: {e}")
                time.sleep(0.5)

    @SpotTracker("Flask Web Server", exit_on_fail=False)
    def _run_web_server(self) -> None:
        """Start the Flask development server on the configured host and port."""
        import logging
        import flask.cli
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)
        flask.cli.show_server_banner = lambda *args: None
        try:
            print(f"Web Dashboard at http://{self.host}:{self.web_port}")
            self.app.run(
                host=self.host, port=self.web_port,
                debug=False, use_reloader=False, threaded=True,
            )
        except OSError as e:
            print(f"Port {self.web_port} unavailable: {e}")
        except Exception as e:
            print(f"Web server error: {e}")

    @SpotTracker("Start Streaming", exit_on_fail=False)
    def start(self) -> None:
        """Launch the capture loop and Flask server as daemon threads."""
        self._streaming = True
        threading.Thread(target=self._stream_loop, daemon=True).start()
        threading.Thread(target=self._run_web_server, daemon=True).start()

    def stop(self) -> None:
        """Signal the stream loop to stop."""
        self._streaming = False
