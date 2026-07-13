import cv2
import time
import threading
from flask import Flask, render_template_string, Response

class SpotWebStreamer:
    def __init__(self, host="0.0.0.0"):
        self.host = host
        self.frames = {
            'cameras': None,
            'map': None
        }
        self.lock = threading.Lock()
        self.app_cameras = Flask("cameras_app")
        self.app_map = Flask("map_app")
        self._setup_routes()

    def _setup_routes(self):
        self.app_cameras.add_url_rule('/', 'index_cameras', self.index_cameras)
        self.app_cameras.add_url_rule('/video_feed', 'video_feed_cameras', self.video_feed_cameras)
        self.app_map.add_url_rule('/', 'index_map', self.index_map)
        self.app_map.add_url_rule('/video_feed', 'video_feed_map', self.video_feed_map)

    def update_camera_grid(self, frame):
        with self.lock:
            self.frames['cameras'] = frame

    def update_map(self, frame):
        with self.lock:
            self.frames['map'] = frame

    def index_cameras(self):
        return render_template_string("""
            <html>
            <head><title>Spot Cameras (2x2)</title></head>
            <body style="background: #111; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0;">
                <img src="/video_feed" style="width: 80%; border: 2px solid #555;">
            </body>
            </html>
        """)

    def gen_frames_cameras(self):
        while True:
            with self.lock:
                frame = self.frames['cameras']
            
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.05)

    def video_feed_cameras(self):
        return Response(self.gen_frames_cameras(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def index_map(self):
        return render_template_string("""
            <html>
            <head><title>Spot Map View</title></head>
            <body style="background: #111; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0;">
                <img src="/video_feed" style="width: 80%; border: 2px solid #555;">
            </body>
            </html>
        """)

    def gen_frames_map(self):
        while True:
            with self.lock:
                frame = self.frames['map']
            
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.05)

    def video_feed_map(self):
        return Response(self.gen_frames_map(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def start(self):
        threading.Thread(
            target=lambda: self.app_cameras.run(host=self.host, port=5555, debug=False, use_reloader=False), 
            daemon=True
        ).start()
        print(f"[WebUI] Cameras server started at http://{self.host}:5555")
        
        threading.Thread(
            target=lambda: self.app_map.run(host=self.host, port=5556, debug=False, use_reloader=False), 
            daemon=True
        ).start()
        print(f"[WebUI] Map server started at http://{self.host}:5556")