# CoreIO services

Everything that runs on Spot's CoreIO (Jetson Xavier NX), and the laptop-side scripts that
put it there.

```
coreio/
  remote.py              HERE — the SSH helpers every deploy script uses
  vision/                the segment + classify service (the sorting demo's eyes)
    vocab/sort3.json       the demo's prompts, bins and grip materials — the file you edit
    bake_vocab.py          HERE — prompts -> weights (needs ultralytics 8.3.156)
    service/               ROBOT — vision_server.py (Python 3.8), Dockerfile, compose.yaml
  deploy_vision.py       HERE — build spot-vision:latest on the robot, install the Dockge stack
  service/               ROBOT — realsense_server.py + realsense_compose.yaml, cloud_server.py
  deploy_realsense.py    HERE — install the RealSense bridge as the spot-realsense Dockge stack
  deploy_cloud.py        HERE — the live-scan stitcher
  demo_web/ + deploy_web.py   the console itself as a container on the robot
```

The CoreIO has **no DNS and no default route**, so nothing there can be downloaded: weights,
code and images all arrive over the SSH session. Every robot-side service binds
CoreIO-localhost, because Spot forwards only 20022 and 21443 off the CoreIO; the console
reaches them through SSH channels.

| port | service |
|---|---|
| 8782 | RealSense bridge |
| 8783 | live-cloud stitcher |
| 8784 | vision (segment + classify) |

## Vision: segment, then classify

Two YOLOE models on the same frame:

1. **segment** — `yoloe-11s-seg-pf`, prompt-free. Separates every object into its own mask
   whatever it is. It decides *that* something is there, and its outline.
2. **classify** — `yoloe-11s-seg` with `vocab/sort3.json` baked in. Every prompt belongs to a
   group — `teddy_bear`, `bottle`, `other`, `ignore` — and has a grip material
   (`soft` / `medium` / `hard`).

Each mask is scored per group from the classify boxes that line up with it. The winner is
reported if it has enough evidence; if nothing claims the object it is still reported, as
`other` with `uncertain: true`. Two groups scoring close together are also uncertain.
`ignore` (people, furniture, the robot) is dropped before replying.

### Changing the categories

```bash
# once: a bake venv with the robot's ultralytics version
uv venv --python 3.12 runtime/bake-venv
uv pip install --python runtime/bake-venv/bin/python ultralytics==8.3.156 \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple

runtime/bake-venv/bin/python coreio/vision/bake_vocab.py   # -> runtime/weights/vision/
python coreio/deploy_vision.py                             # build on the robot, restart the stack
```

The bake needs ultralytics **8.3.156**, the version inside `yolo:latest` on the robot: a
checkpoint pickled by a newer ultralytics does not reliably load on an older one. It also
downloads MobileCLIP (~570 MB) once, to encode the prompts; the robot never needs it.

### Dockge

`deploy_vision.py` builds `spot-vision:latest` on the CoreIO (`FROM yolo:latest`, COPY only)
and writes `/data/stacks/spot-vision/compose.yaml`, so the service is started, stopped and
watched in Dockge with the other stacks:

```bash
ssh -L 5001:127.0.0.1:5001 spot_coreio_wifi   # then http://localhost:5001
```

The thresholds (`VISION_ACCEPT`, `VISION_CERTAIN`, `VISION_MARGIN`, `VISION_SEG_CONF`) are in
the stack's environment and can be changed in Dockge without a rebuild. The stack replaces
`spot-yolo` — deploying stops that container, and the two should not run together.

### Speed

Measured on the robot with the ROS 2 stack also running (load average ~25 on six cores):
five surround cameras **~1.7 s**, one hand frame segment-only **~0.4 s**. The CPU is the
bottleneck, not the GPU. A TensorRT export of both models is the next step if that matters.

### Protocol

`GET /health`, `GET /vocab`, and `POST /analyse` with a length-prefixed JSON header followed
by length-prefixed JPEGs:

```
[4B BE header length][header JSON][ [4B BE length][JPEG] ... ]
header = {"cameras": [...], "classify": true, "imgsz": 640}
```

The reply is `{"objects": [...], "counts": {...}, "elapsed_ms": ...}`; each object is
`{camera, group, label, seg_label, conf, seg_conf, uncertain, scores, material, xyxy,
polygon, cx, cy, area}`, in the original image's pixels. `cx, cy` is the point inside the
mask furthest from its edge — where a grasp should aim. `classify: false` skips stage 2,
which is what the approach servo uses to keep track of a target it has already chosen.
