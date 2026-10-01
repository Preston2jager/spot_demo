"""Stitch Spot's depth sweeps into one live point cloud, on the CoreIO's GPU.

This robot carries no lidar payload. What it has is five depth cameras — front-left,
front-right, left, right and back — which between them see all the way around the body,
and which are the same sensors the recorded map's point clouds were built from. A sweep
is one frame from each; stitching is putting all five into the map's seed frame and
merging them with everything seen so far.

**Why here and not on the laptop.** Five depth frames is about 2.4 million candidate
points *per sweep*, and the useful product is a voxel grid a few hundred thousand points
big that grows as the robot walks. Unprojecting, filtering, transforming and deduplicating
that at a few hertz is exactly the shape of work the Xavier's GPU does for nothing, and
it means the wire only ever carries the *new* voxels — a few hundred KB — instead of the
whole accumulation every frame.

The client is `src/spot_cloud.py`, which is where the depth frames come from: it is the
process holding Spot's credentials, so it fetches the images and the transforms and posts
them here. This service never talks to the robot's API — the `yolo:latest` image it runs
in has no bosdyn, and the CoreIO has no route to the internet to install one.

    GET  /health              device, voxel size, how much is accumulated
    POST /sweep               one sweep in, the voxels it *added* back out
    GET  /cloud?since=<seq>   everything accumulated after that sequence number
    POST /clear               forget the accumulation and start again

Both binary replies are framed the same way as the request: a 4-byte big-endian header
length, that many bytes of JSON, then packed float32 xyz triplets.

Runs in the `yolo:latest` image (Python 3.8, torch 2.0.0+nv23.05, CUDA 11.4) alongside
`yolo_server.py`, and like it needs no `pip install` — torch and numpy are already there.
Deploy it with `python coreio/deploy_cloud.py`.
"""
import json
import os
import struct
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

try:  # pragma: no cover - the import error is the message, and only ever seen on the robot
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "cloud_server needs torch: run it inside the yolo image, e.g. "
        f"`bash ~/cloud_service/run_cloud_server.sh` ({exc})"
    )

HOST = os.environ.get("CLOUD_HOST", "127.0.0.1")
PORT = int(os.environ.get("CLOUD_PORT", "8783"))
# "cuda" is the whole point of running here; "cpu" is for a bench test off the robot.
# An unavailable CUDA falls back rather than failing: a live map on the ARM cores is
# slower but still a live map, and /health says which one answered.
DEVICE = os.environ.get("CLOUD_DEVICE", "cuda")
# Voxel edge in metres. 6 cm is about the resolution Spot's stereo depth is honest to at
# room range, and it is what keeps a walked lap of a lab inside the budget below.
VOXEL_M = float(os.environ.get("CLOUD_VOXEL_M", "0.06"))
# Upper bound on voxels kept. Past it the *oldest* are dropped, so a robot that walks all
# day keeps the room it is in rather than the one it left an hour ago. 600k voxels is
# 7 MB on the wire for a client that asks for everything, which is the number that
# actually matters.
BUDGET = int(os.environ.get("CLOUD_BUDGET", "600000"))

# Voxel coordinates are packed one per int64 — 21 bits an axis, signed by an offset —
# so "have I seen this voxel" is an integer lookup rather than a spatial query. 21 bits
# reaches +-1,048,575 voxels, which at 6 cm is +-62 km: not a limit anybody meets.
_BITS = 21
_OFFSET = 1 << (_BITS - 1)
_MASK = (1 << _BITS) - 1

_lock = threading.Lock()
_device = None
# Sorted, so that testing a sweep's voxels against the accumulation is a searchsorted
# rather than a scan. `_stamp` is the sequence number each voxel arrived at, which is
# what makes both "everything since seq N" and "drop the oldest" one line each.
_keys = None
_stamp = None
_seq = 0
_sweeps = 0
_last_ms = 0.0
_last_at = 0.0
# One (u, v) grid per image size, built once: every sweep unprojects the same lattice,
# and rebuilding it five times a second is the sort of waste that only shows up as the
# GPU never idling.
_grids = {}


def _pick_device():
    """The device to work on, honouring CLOUD_DEVICE but never failing over it."""
    want = DEVICE.strip().lower()
    if want in ("cpu",):
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda:0" if want in ("cuda", "gpu", "0") else want)
    print("[cloud] CUDA is not available — falling back to the CPU", flush=True)
    return torch.device("cpu")


def _grid(width, height):
    """The pixel lattice for one image size, cached on the device."""
    key = (width, height)
    grid = _grids.get(key)
    if grid is None:
        us = torch.arange(width, device=_device, dtype=torch.float32)
        vs = torch.arange(height, device=_device, dtype=torch.float32)
        grid = (us.unsqueeze(0).expand(height, width).reshape(-1),
                vs.unsqueeze(1).expand(height, width).reshape(-1))
        _grids[key] = grid
    return grid


def _unpack(body):
    """Split a request body into (header dict, [depth frame as uint16 bytes, ...])."""
    if len(body) < 4:
        raise ValueError("body too short")
    (header_len,) = struct.unpack_from(">I", body, 0)
    offset = 4
    header = json.loads(body[offset:offset + header_len].decode("utf-8"))
    offset += header_len
    blobs = []
    while offset < len(body):
        (blob_len,) = struct.unpack_from(">I", body, offset)
        offset += 4
        blobs.append(body[offset:offset + blob_len])
        offset += blob_len
    cameras = header.get("cameras", [])
    if len(cameras) != len(blobs):
        raise ValueError(
            f"header lists {len(cameras)} cameras but body carries {len(blobs)} frames"
        )
    return header, blobs


def _points_from(camera, blob, near_m, far_m):
    """One depth frame, as xyz points in the seed frame — or None if it has nothing.

    The unprojection is the pinhole one: a depth pixel is a distance along the sensor's
    z axis, not a range along the ray, so z comes straight off the image and x and y
    follow from the intrinsics. Zero means "no return" and is dropped rather than placed
    at the camera, which is what would otherwise put a solid blob at Spot's feet.
    """
    width = int(camera["width"])
    height = int(camera["height"])
    raw = zlib.decompress(blob) if camera.get("compressed") else blob
    expected = width * height * 2
    if len(raw) != expected:
        raise ValueError(
            "{}: expected {} bytes of depth, got {}".format(camera.get("name"), expected, len(raw))
        )
    # np.frombuffer first: torch.frombuffer is 1.10+, and the copy here is a memcpy of a
    # couple of hundred KB against the several megabytes of float work that follows.
    depth = torch.as_tensor(
        np.frombuffer(raw, dtype="<u2").astype(np.float32), device=_device
    ) / float(camera.get("depth_scale", 1000.0))

    valid = (depth > near_m) & (depth < far_m)
    if not bool(valid.any()):
        return None
    us, vs = _grid(width, height)
    z = depth[valid]
    x = (us[valid] - float(camera["cx"])) * z / float(camera["fx"])
    y = (vs[valid] - float(camera["cy"])) * z / float(camera["fy"])

    transform = torch.as_tensor(
        np.asarray(camera["seed_tform_sensor"], dtype=np.float32).reshape(4, 4), device=_device
    )
    local = torch.stack((x, y, z), dim=1)
    return local @ transform[:3, :3].T + transform[:3, 3]


def _keys_of(points):
    """The voxel each point falls in, packed one per int64."""
    voxel = torch.floor(points / VOXEL_M).to(torch.int64) + _OFFSET
    voxel.clamp_(0, _MASK)
    return (voxel[:, 0] << (2 * _BITS)) | (voxel[:, 1] << _BITS) | voxel[:, 2]


def _points_of(keys):
    """Voxel centres for packed keys — the inverse of `_keys_of`, to a half-voxel."""
    x = (keys >> (2 * _BITS)) & _MASK
    y = (keys >> _BITS) & _MASK
    z = keys & _MASK
    grid = torch.stack((x, y, z), dim=1).to(torch.float32) - float(_OFFSET)
    return (grid + 0.5) * VOXEL_M


def _merge(new_keys):
    """Add a sweep's voxels to the accumulation and return the ones that were new.

    New-only because that is what goes over the wire: a client that has been watching
    since the start already holds everything else, and re-sending the room every frame
    is the cost this service exists to avoid.
    """
    global _keys, _stamp, _seq

    _seq += 1
    fresh = torch.unique(new_keys)
    if _keys is None or _keys.numel() == 0:
        added = fresh
        _keys = fresh
        _stamp = torch.full_like(fresh, _seq)
    else:
        # `_keys` is kept sorted, so membership is a binary search: where a voxel would be
        # inserted, and whether the neighbour there is the same voxel.
        where = torch.searchsorted(_keys, fresh).clamp_(max=_keys.numel() - 1)
        added = fresh[_keys[where] != fresh]
        if added.numel():
            _keys = torch.cat((_keys, added))
            _stamp = torch.cat((_stamp, torch.full_like(added, _seq)))
            order = torch.argsort(_keys)
            _keys, _stamp = _keys[order], _stamp[order]

    if _keys.numel() > BUDGET:
        # Oldest out. Sorting by arrival and keeping the tail is stable and cheap, and it
        # degrades the way an operator expects: the room behind you fades, the room you
        # are standing in does not.
        keep = torch.argsort(_stamp)[-BUDGET:]
        _keys, _stamp = _keys[keep], _stamp[keep]
        order = torch.argsort(_keys)
        _keys, _stamp = _keys[order], _stamp[order]
        # A voxel evicted this sweep must not be reported as added, or a client would draw
        # a point the service no longer believes in.
        where = torch.searchsorted(_keys, added).clamp_(max=_keys.numel() - 1)
        added = added[_keys[where] == added]
    return added


def _payload(keys):
    """Packed float32 xyz for a set of voxel keys, ready for a WebGL buffer."""
    if keys is None or keys.numel() == 0:
        return b""
    return _points_of(keys).to("cpu").numpy().astype(np.float32).tobytes()


def _stats():
    return {
        "device": str(_device),
        "voxel": VOXEL_M,
        "budget": BUDGET,
        "seq": _seq,
        "total": int(_keys.numel()) if _keys is not None else 0,
        "sweeps": _sweeps,
        "last_ms": round(_last_ms, 1),
        "age": round(time.time() - _last_at, 2) if _last_at else None,
    }


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        """Quieten the default stderr access log; the handlers log what matters."""
        return

    def _send_json(self, status, payload):
        blob = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def _send_cloud(self, header, points):
        """A framed reply: 4-byte header length, JSON, then the float32 triplets."""
        head = json.dumps(header).encode("utf-8")
        body = struct.pack(">I", len(head)) + head + points
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path)
        if path.path == "/health":
            with _lock:
                self._send_json(200, dict(_stats(), status="ok"))
            return
        if path.path != "/cloud":
            self._send_json(404, {"error": "not found"})
            return
        try:
            since = int(parse_qs(path.query).get("since", ["0"])[0])
        except ValueError:
            self._send_json(400, {"error": "since must be an integer"})
            return
        with _lock:
            if _keys is None or _keys.numel() == 0:
                self._send_cloud(dict(_stats(), since=since, points=0), b"")
                return
            wanted = _keys if since <= 0 else _keys[_stamp > since]
            self._send_cloud(
                dict(_stats(), since=since, points=int(wanted.numel())), _payload(wanted)
            )

    def do_POST(self):
        global _sweeps, _last_ms, _last_at
        path = urlparse(self.path).path
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))

        if path == "/clear":
            with _lock:
                _clear()
                self._send_json(200, dict(_stats(), status="cleared"))
            return
        if path != "/sweep":
            self._send_json(404, {"error": "not found"})
            return

        try:
            header, blobs = _unpack(body)
        except Exception as exc:
            self._send_json(400, {"error": f"bad request: {exc}"})
            return

        near = float(header.get("near_m", 0.3))
        far = float(header.get("far_m", 6.0))
        started = time.time()
        try:
            with _lock:
                if header.get("reset"):
                    _clear()
                chunks = []
                for camera, blob in zip(header["cameras"], blobs):
                    points = _points_from(camera, blob, near, far)
                    if points is not None and points.numel():
                        chunks.append(points)
                if not chunks:
                    self._send_cloud(dict(_stats(), added=0, cameras=len(blobs)), b"")
                    return
                added = _merge(_keys_of(torch.cat(chunks)))
                payload = _payload(added)
                if _device.type == "cuda":
                    torch.cuda.synchronize()
                _sweeps += 1
                _last_ms = (time.time() - started) * 1000
                _last_at = time.time()
                stats = dict(_stats(), added=int(added.numel()), cameras=len(blobs))
        except Exception as exc:
            print(f"[cloud] sweep failed: {exc}", flush=True)
            self._send_json(500, {"error": f"sweep failed: {exc}"})
            return

        print(
            "[cloud] sweep {} — {} cameras, +{} voxels, {} total, {:.0f} ms".format(
                stats["seq"], len(blobs), stats["added"], stats["total"], stats["last_ms"]
            ),
            flush=True,
        )
        self._send_cloud(stats, payload)


def _clear():
    """Forget the accumulation. The caller holds the lock."""
    global _keys, _stamp, _seq, _sweeps
    _keys, _stamp, _seq, _sweeps = None, None, 0, 0


def main():
    global _device
    _device = _pick_device()
    if _device.type == "cuda":
        # Same reason yolo_server warms up before it answers: the first CUDA call pays for
        # context creation, and paying it inside the operator's first sweep looks like a
        # slow robot rather than a one-off.
        torch.zeros(1, device=_device).sum().item()
        print(f"[cloud] CUDA ready on {torch.cuda.get_device_name(0)}", flush=True)
    server = ThreadingHTTPServer((HOST, PORT), _Handler)
    server.daemon_threads = True
    print(
        f"[cloud] listening on {HOST}:{PORT} — device {_device}, voxel {VOXEL_M} m, budget {BUDGET} points",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
