"""RealSense colour and depth, served off Spot's CoreIO for the web console.

The camera is not ours to open. `realsense2_camera` already holds the USB device — that
is what the `realsense_ros2` container is — and a RealSense streams to exactly one
process, so a second `pyrealsense2` pipeline would not get a frame, it would get a
"device busy" and take the ROS graph down with it on the way. This subscribes to the
topics that node already publishes instead, which costs the camera nothing.

Aligned depth is the topic worth having: `aligned_depth_to_color` is depth resampled into
the colour camera's frame, so pixel (x, y) means the same thing in both images and a
distance can be read straight off the picture the operator is pointing at. Raw
`depth/image_rect_raw` is a different intrinsic and would need a reprojection to say
anything about what is on screen.

Served as plain HTTP on CoreIO-localhost, reached the same way the detector is: Spot
forwards only 20022 and 21443 off the CoreIO, so the console rides a `direct-tcpip`
channel on its SSH session rather than anything new being exposed.

    GET /health            what is arriving, how old it is, at what size
    GET /color.jpg         the latest colour frame
    GET /depth.jpg         the latest depth frame, colour-mapped for display
    GET /depth/at?u=&v=    the distance in metres where the operator pointed
    GET /rgbd?step=        colour + raw aligned depth + intrinsics in one reply, for the
                           console's 3D reconstruction (which runs on the laptop, not here)

Runs inside the `realsense-coreio` image: ROS 2 Humble, Python 3.10, cv_bridge, cv2 and
numpy are all already there. Nothing here needs a `pip install`, which matters because
the CoreIO has no route to the internet. Kept to the same Python 3.8 subset as
`yolo_server.py` — the two services share a lint config and may yet share an image.
"""
import json
import os
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:  # pragma: no cover - the import error is the message, and only ever seen on the robot
    from urllib.parse import parse_qs, urlparse

    import cv2
    import numpy as np
    import rclpy
    from cv_bridge import CvBridge
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "realsense_server needs ROS 2 and cv_bridge: run it inside the realsense image, "
        f"e.g. the spot-realsense Dockge stack ({exc})"
    )

COLOR_TOPIC = os.environ.get("RS_COLOR_TOPIC", "/realsense/camera/color/image_raw")
DEPTH_TOPIC = os.environ.get("RS_DEPTH_TOPIC", "/realsense/camera/aligned_depth_to_color/image_raw")
# What depth falls back to when nothing is aligned. `realsense_ros2` is launched with
# `align_depth.enable:=false` as often as not — it is someone else's container — and a
# bridge that only knew the aligned topic then served colour and a depth tab stuck on
# "waiting" forever. Unaligned depth is still depth; `/health` says which one is in use, so
# a click reading off it is known to be approximate rather than silently so.
RAW_DEPTH_TOPIC = os.environ.get("RS_RAW_DEPTH_TOPIC", "/realsense/camera/depth/image_rect_raw")
# The colour camera's intrinsics. Aligned depth is in the colour camera's frame, so these
# are the intrinsics of both — what turns a depth pixel into a point.
INFO_TOPIC = os.environ.get("RS_INFO_TOPIC", "/realsense/camera/color/camera_info")
HOST = os.environ.get("RS_HOST", "127.0.0.1")
PORT = int(os.environ.get("RS_PORT", "8782"))
JPEG_QUALITY = int(os.environ.get("RS_JPEG_QUALITY", "80"))
# The display range for the colour map, in metres. Depth outside it is clamped rather
# than dropped: a wall at 12 m should read as "far", not as "no reading", which is what
# the black of an invalid pixel means.
NEAR_M = float(os.environ.get("RS_NEAR_M", "0.3"))
FAR_M = float(os.environ.get("RS_FAR_M", "6.0"))
# A frame older than this is reported as stale by /health. Two seconds is far longer than
# any real gap at 30 fps, so saying "stale" means the node has actually stopped.
STALE_SEC = float(os.environ.get("RS_STALE_SEC", "2.0"))
# The camera is mounted inverted on this robot, so both streams arrive upside down. 180 is
# therefore the default rather than 0 — the honest description of the hardware, and one
# fewer thing to remember at deploy time. 0/90/180/270; anything else is ignored with a
# warning rather than guessed at.
#
# Applied once, on arrival, so *everything* downstream agrees: the JPEG, the depth probe
# and the scene stats all describe the picture the operator is actually looking at.
# Rotating only the pictures would leave a click measuring the diagonally opposite corner.
ROTATE = int(os.environ.get("RS_ROTATE", "180"))
_ROTATIONS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}

_bridge = CvBridge()
_lock = threading.Lock()
_color = {"frame": None, "stamp": 0.0}
_depth = {"frame": None, "stamp": 0.0, "aligned": False}
_info = {"intrinsics": None}


def _now():
    return time.time()


def _oriented(frame):
    """Put an arriving frame the right way up, whatever the mounting is."""
    turn = _ROTATIONS.get(ROTATE)
    if turn is None:
        return frame
    return cv2.rotate(frame, turn)


def _oriented_intrinsics(k, width, height):
    """fx, fy, cx, cy and the frame size after the same rotation `_oriented` applies."""
    fx, fy, cx, cy = float(k[0]), float(k[4]), float(k[2]), float(k[5])
    if ROTATE == 180:
        return {"fx": fx, "fy": fy, "cx": width - 1 - cx, "cy": height - 1 - cy,
                "width": width, "height": height}
    if ROTATE == 90:  # clockwise: (u, v) -> (height - 1 - v, u)
        return {"fx": fy, "fy": fx, "cx": height - 1 - cy, "cy": cx, "width": height, "height": width}
    if ROTATE == 270:  # counter-clockwise: (u, v) -> (v, width - 1 - u)
        return {"fx": fy, "fy": fx, "cx": cy, "cy": width - 1 - cx, "width": height, "height": width}
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": width, "height": height}


class _Camera:
    """The ROS 2 side: two subscriptions, each keeping only the newest frame.

    Only the latest matters — the console renders what is happening now, and a queue would
    just mean serving something older under load. Hence depth of 1 and best-effort QoS,
    which is what a camera publisher offers anyway.
    """

    def __init__(self, node):
        self.node = node
        node.create_subscription(Image, COLOR_TOPIC, self._on_color, qos_profile_sensor_data)
        node.create_subscription(
            Image, DEPTH_TOPIC, lambda msg: self._on_depth(msg, True), qos_profile_sensor_data
        )
        if RAW_DEPTH_TOPIC and RAW_DEPTH_TOPIC != DEPTH_TOPIC:
            node.create_subscription(
                Image, RAW_DEPTH_TOPIC, lambda msg: self._on_depth(msg, False),
                qos_profile_sensor_data,
            )
        node.create_subscription(CameraInfo, INFO_TOPIC, self._on_info, qos_profile_sensor_data)

    def _on_info(self, msg):
        if not msg.width or not msg.height or not msg.k[0]:
            return
        with _lock:
            _info["intrinsics"] = _oriented_intrinsics(msg.k, int(msg.width), int(msg.height))

    def _on_color(self, msg):
        try:
            frame = _oriented(_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8"))
        except Exception as exc:
            self.node.get_logger().warn(f"colour frame dropped: {exc}")
            return
        with _lock:
            _color["frame"] = frame
            _color["stamp"] = _now()

    def _on_depth(self, msg, aligned):
        # Raw depth only stands in while aligned depth is absent: when both are being
        # published, the unaligned frames would otherwise overwrite the useful ones.
        if not aligned:
            with _lock:
                if _depth["aligned"] and _now() - _depth["stamp"] < STALE_SEC:
                    return
        try:
            # passthrough, not a conversion: this is 16UC1 in millimetres, and asking
            # cv_bridge for anything else would quietly rescale the distances.
            frame = _oriented(_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough"))
        except Exception as exc:
            self.node.get_logger().warn(f"depth frame dropped: {exc}")
            return
        with _lock:
            _depth["frame"] = frame
            _depth["stamp"] = _now()
            _depth["aligned"] = aligned


def _snapshot(store):
    with _lock:
        frame, stamp = store["frame"], store["stamp"]
    return frame, stamp


def _to_metres(depth_frame):
    """Millimetre integers to metres, with 0 (no reading) as NaN rather than 0 m."""
    metres = depth_frame.astype("float32") * 0.001
    metres[depth_frame == 0] = float("nan")
    return metres


def _colorize(depth_frame):
    """Colour-map depth for display, leaving pixels with no reading black.

    TURBO rather than JET: it is perceptually ordered, so "closer" reads as one direction
    along the ramp instead of as two different-looking ends meeting in the middle.
    """
    metres = depth_frame.astype("float32") * 0.001
    valid = depth_frame > 0
    span = max(FAR_M - NEAR_M, 1e-6)
    normalised = np.clip((metres - NEAR_M) / span, 0.0, 1.0)
    picture = cv2.applyColorMap((normalised * 255).astype("uint8"), cv2.COLORMAP_TURBO)
    picture[~valid] = (0, 0, 0)
    return picture


def _encode(picture):
    ok, buffer = cv2.imencode(".jpg", picture, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buffer.tobytes()


def _depth_at(depth_frame, x, y, window):
    """The distance at one pixel, as the median of the valid readings around it.

    A single depth pixel drops out often — a dark edge, a specular highlight — and a
    reading that flickers between 1.4 m and nothing is worse than useless to whoever is
    pointing at it. The median over a small window is stable and still local.
    """
    height, width = depth_frame.shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        return None, 0, 0
    half = max(0, window // 2)
    patch = depth_frame[
        max(0, y - half):min(height, y + half + 1),
        max(0, x - half):min(width, x + half + 1),
    ]
    valid = patch[patch > 0]
    if valid.size == 0:
        return None, 0, int(patch.size)
    return float(np.median(valid)) * 0.001, int(valid.size), int(patch.size)


def _stats(depth_frame):
    """A one-line summary of the scene: what is nearest, what the middle of the view is at."""
    metres = _to_metres(depth_frame)
    valid = np.isfinite(metres)
    coverage = float(valid.mean()) if valid.size else 0.0
    if not valid.any():
        return {"coverage": coverage, "near_m": None, "far_m": None, "centre_m": None}
    height, width = depth_frame.shape[:2]
    centre, _, _ = _depth_at(depth_frame, width // 2, height // 2, 9)
    return {
        "coverage": round(coverage, 3),
        "near_m": round(float(np.nanmin(metres)), 3),
        "far_m": round(float(np.nanmax(metres)), 3),
        "centre_m": round(centre, 3) if centre is not None else None,
    }


class Handler(BaseHTTPRequestHandler):
    """Plain GETs. No POST body to frame, so this is plain HTTP rather than the detector's
    length-prefixed protocol — the console asks for a picture and gets a picture."""

    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass  # one line per frame at 10 fps is not a log, it is a denial of service

    def do_GET(self):
        route = urlparse(self.path)
        try:
            if route.path == "/health":
                self._json(200, self._health())
            elif route.path == "/color.jpg":
                self._image(_snapshot(_color)[0], lambda f: f)
            elif route.path == "/depth.jpg":
                self._image(_snapshot(_depth)[0], _colorize)
            elif route.path == "/depth/at":
                self._json(200, self._probe(parse_qs(route.query)))
            elif route.path == "/rgbd":
                self._rgbd(parse_qs(route.query))
            else:
                self._json(404, {"error": f"no such path: {route.path}"})
        except Exception as exc:
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

    # ── the answers ───────────────────────────────────────────────────────────

    def _health(self):
        color_frame, color_stamp = _snapshot(_color)
        depth_frame, depth_stamp = _snapshot(_depth)
        now = _now()
        return {
            "status": "ok" if color_frame is not None or depth_frame is not None else "waiting",
            "color": self._stream_state(color_frame, color_stamp, now),
            "depth": self._stream_state(depth_frame, depth_stamp, now),
            "topics": {"color": COLOR_TOPIC, "depth": DEPTH_TOPIC, "raw_depth": RAW_DEPTH_TOPIC},
            "depth_aligned": bool(_depth["aligned"]) if depth_frame is not None else None,
            "range_m": [NEAR_M, FAR_M],
            "rotate": ROTATE if ROTATE in _ROTATIONS else 0,
            "stats": _stats(depth_frame) if depth_frame is not None else None,
        }

    @staticmethod
    def _stream_state(frame, stamp, now):
        if frame is None:
            return {"present": False, "age": None, "size": None, "stale": True}
        age = round(now - stamp, 2)
        return {
            "present": True,
            "age": age,
            "size": [int(frame.shape[1]), int(frame.shape[0])],
            "stale": age > STALE_SEC,
        }

    def _probe(self, query):
        depth_frame, stamp = _snapshot(_depth)
        aligned = bool(_depth["aligned"])
        if depth_frame is None:
            return {"error": "no depth frame yet", "distance_m": None}
        height, width = depth_frame.shape[:2]
        try:
            # u/v are fractions of the frame, which is what a browser can actually supply:
            # the picture on the page is a scaled copy, and the client has no business
            # knowing the sensor is 1280x720 today and something else after a relaunch.
            # x/y stay for anything asking in real pixels.
            if "u" in query and "v" in query:
                x = round(float(query["u"][0]) * (width - 1))
                y = round(float(query["v"][0]) * (height - 1))
            else:
                x = int(float(query.get("x", ["0"])[0]))
                y = int(float(query.get("y", ["0"])[0]))
        except (TypeError, ValueError, IndexError):
            return {"error": "x/y (pixels) or u/v (0-1 fractions) must be numbers",
                    "distance_m": None}
        window = int(float(query.get("window", ["9"])[0]))
        distance, used, total = _depth_at(depth_frame, x, y, window)
        return {
            "x": x,
            "y": y,
            "distance_m": round(distance, 3) if distance is not None else None,
            "samples": used,
            "window": total,
            "age": round(_now() - stamp, 2),
            "size": [int(depth_frame.shape[1]), int(depth_frame.shape[0])],
            "aligned": aligned,
        }

    def _rgbd(self, query):
        """Colour and raw depth of one moment, with the intrinsics to back-project them.

        One reply rather than three requests, so the pair is as close to simultaneous as
        the topics allow. ``step`` decimates both — 2 quarters the bytes over Spot's wifi and
        is still a point every few millimetres at a metre. Depth stays 16-bit millimetres,
        PNG-compressed: the colour-mapped JPEG is a picture, not a measurement.

        Reply: b"RGBD", three big-endian uint32 lengths (header, JPEG, PNG), then each.
        """
        colour, colour_stamp = _snapshot(_color)
        depth, depth_stamp = _snapshot(_depth)
        with _lock:
            intrinsics = _info["intrinsics"]
            aligned = bool(_depth["aligned"])
        if colour is None or depth is None:
            self._json(503, {"error": "no colour and depth yet — is realsense_ros2 publishing?"})
            return
        if intrinsics is None:
            self._json(503, {"error": f"no camera_info on {INFO_TOPIC} yet"})
            return
        step = max(1, min(8, int(float(query.get("step", ["2"])[0]))))
        height, width = depth.shape[:2]
        if colour.shape[:2] != (height, width):
            colour = cv2.resize(colour, (width, height), interpolation=cv2.INTER_AREA)
        depth = np.ascontiguousarray(depth[::step, ::step])
        colour = np.ascontiguousarray(colour[::step, ::step])
        sx = width / float(intrinsics["width"]) / step
        sy = height / float(intrinsics["height"]) / step
        now = _now()
        header = json.dumps({
            "fx": intrinsics["fx"] * sx, "fy": intrinsics["fy"] * sy,
            "cx": intrinsics["cx"] * sx, "cy": intrinsics["cy"] * sy,
            "width": int(depth.shape[1]), "height": int(depth.shape[0]),
            "aligned": aligned,
            "color_age": round(now - colour_stamp, 3), "depth_age": round(now - depth_stamp, 3),
        }).encode("utf-8")
        ok_jpg, jpg = cv2.imencode(".jpg", colour, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        ok_png, png = cv2.imencode(".png", depth.astype("uint16"), [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
        if not (ok_jpg and ok_png):
            raise RuntimeError("RGBD encode failed")
        body = b"RGBD" + struct.pack(">III", len(header), len(jpg), len(png)) + header + jpg.tobytes() + png.tobytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    # ── plumbing ──────────────────────────────────────────────────────────────

    def _image(self, frame, prepare):
        if frame is None:
            self._json(503, {"error": "no frame yet — is realsense_ros2 publishing?"})
            return
        body = _encode(prepare(frame))
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(body)))
        # The console polls this; a cached frame is a frozen camera.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    rclpy.init()
    node = rclpy.create_node("spot_realsense_bridge")
    _Camera(node)
    print(f"[realsense] subscribing to {COLOR_TOPIC} and {DEPTH_TOPIC} "
          f"(falling back to {RAW_DEPTH_TOPIC})", flush=True)
    if ROTATE and ROTATE not in _ROTATIONS:
        print(f"[realsense] RS_ROTATE={ROTATE} is not 0/90/180/270 — serving frames unrotated",
              flush=True)
    else:
        print(f"[realsense] rotating frames {ROTATE}deg on arrival", flush=True)

    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.daemon_threads = True
    print(f"[realsense] listening on {HOST}:{PORT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
