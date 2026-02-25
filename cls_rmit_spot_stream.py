import cv2
import numpy as np
import time
import threading
from flask import Flask, render_template_string, Response
from bosdyn.api import image_pb2
from bosdyn.client.image import build_image_request

from cls_rmit_spot_tracker import SpotTracker

class SpotStreamer:
    def __init__(self, robot, host="0.0.0.0", web_port=5555):
        """
        :param robot: 初始化的 Spot robot 对象
        :param host: 网页绑定的 IP (默认 0.0.0.0 允许所有网络访问)
        :param web_port: Flask 网页和推流的访问端口 (默认 5000)
        """
        self.robot = robot
        self.host = host
        self.web_port = web_port
        self._streaming = False
        
        # 共享内存字典，用于存放最新处理好的画面
        self.frames = {
            'hand': None,
            'grid': None
        }
        # 线程锁，防止读写冲突导致画面撕裂
        self.lock = threading.Lock()
        
        # 初始化 Flask 网页应用
        self.app = Flask("spot_web_dashboard")
        self._setup_routes()

    def _setup_routes(self):
        @self.app.route('/')
        def index():
            # 注意这里：我们用 <img> 标签直接接收 MJPEG 流，取代了 iframe
            html_content = """
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
                    /* 限制画面最大宽度，保持比例 */
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
            return render_template_string(html_content)

        @self.app.route('/video_feed/<stream_type>')
        def video_feed(stream_type):
            """路由：根据请求类型返回对应的视频流"""
            if stream_type not in ['hand', 'grid']:
                return "Invalid stream type", 400
            # 使用 multipart/x-mixed-replace 协议推送 MJPEG 流
            return Response(self.generate_frames(stream_type),
                            mimetype='multipart/x-mixed-replace; boundary=frame')

    def generate_frames(self, stream_type):
        """生成器：不断将 numpy 数组编码为 JPEG 字节流发送给网页"""
        while self._streaming:
            with self.lock:
                frame = self.frames[stream_type]
            if frame is not None:
                # 压缩成 JPEG，quality=70 可以在画质和带宽间取得平衡
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            # 控制发送频率，避免占满 CPU (大约 20 FPS)
            time.sleep(0.05)

    def _stream_loop(self):
        """后台抓图与拼接处理线程"""
        image_client = self.robot.ensure_client("image")
        source_names = ['hand_color_image', 'left_fisheye_image', 'right_fisheye_image', 'back_fisheye_image']
        W_CELL, H_CELL = 640, 480 
        W_CELL_2, H_CELL_2 = 1280, 960
        
        reqs = [
            build_image_request(src, pixel_format=image_pb2.Image.PIXEL_FORMAT_RGB_U8, quality_percent=70) 
            for src in source_names
        ]

        empty_cell = np.zeros((H_CELL, W_CELL, 3), dtype=np.uint8)
        empty_cell_2 = np.zeros((H_CELL_2, W_CELL_2, 3), dtype=np.uint8)

        while self._streaming:
            try:
                responses = image_client.get_image(reqs)
                img_map = {}
                for res in responses:
                    source_name = res.source.name
                    if res.status == image_pb2.ImageResponse.STATUS_OK:
                        arr = np.frombuffer(res.shot.image.data, dtype=np.uint8)
                        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if decoded is not None:
                            # 根据相机位置旋转画面
                            rotated = decoded 
                            if source_name == 'right_fisheye_image':
                                rotated = cv2.rotate(decoded, cv2.ROTATE_180)
                            
                            final_img = cv2.resize(rotated, (W_CELL, H_CELL))
                            label = source_name.replace('_image', '').upper()
                            cv2.putText(final_img, label, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                            img_map[source_name] = final_img

                # --- 1. 单独的 Hand 画面 ---
                current_hand = img_map.get('hand_color_image', empty_cell_2)
                
                # --- 2. 拼接 2x2 画面 ---
                row1 = np.hstack([img_map.get('hand_color_image', empty_cell), img_map.get('back_fisheye_image', empty_cell)])
                row2 = np.hstack([img_map.get('left_fisheye_image', empty_cell), img_map.get('right_fisheye_image', empty_cell)])
                current_grid = np.vstack([row1, row2])

                # 使用线程锁更新共享内存
                with self.lock:
                    self.frames['hand'] = current_hand
                    self.frames['grid'] = current_grid

                # 控制抓图频率
                time.sleep(0.05) 

            except Exception as e:
                print(f"Error: {e}")
                time.sleep(0.5)

    @SpotTracker("Start Streaming")
    def start(self):
        """同时启动视频抓取线程和 Flask WebUI"""
        self._streaming = True
        threading.Thread(target=self._stream_loop, daemon=True).start()
        threading.Thread(
            target=lambda: self.app.run(host=self.host, port=self.web_port, debug=False, use_reloader=False, threaded=True), 
            daemon=True
        ).start()

    def stop(self):
        self._streaming = False