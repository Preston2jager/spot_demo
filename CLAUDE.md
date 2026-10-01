# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Boston Dynamics Spot robot control: segment-then-classify vision on the CoreIO's Jetson, GraphNav
navigation, a three-bin sorting demo, and a single-page browser console. Python 3.12+, managed with `uv`.

## Setup

```bash
uv sync                  # install dependencies into .venv
source .venv/bin/activate
```

Robot credentials live in `credentials.json` at the project root (not committed):
```json
{ "hostname": "192.168.80.3", "username": "user", "password": "..." }
```

Run from the **project root** so `from src.xxx import ...` resolves:
```bash
python start.py                                  # console on :8080, Dockge forwarded to :8081
WEB_PORT=9000 python start.py
```

There is no terminal menu any more — `cli/` was removed. `SPOT_HOSTNAME` / `SPOT_USERNAME` /
`SPOT_PASSWORD` override `credentials.json` (the `demo_web` container, where Spot is `192.168.50.3`).

## Map Workflow

1. Record a map with `src/ref/recording_command_line.py` (fiducial must be in view at start).
2. Download it into `runtime/maps/<map_name>/`; inspect waypoints with `src/ref/check_map.py`.
3. Point `SPOT_MAP_DIR` at it (default `./runtime/maps/amp_3`). The console draws only the graph
   GraphNav is holding; a freshly booted Spot holds none, and the Map tab's *Load … onto Spot*
   (`/api/map/load`, the `load_map` handler) uploads this directory and localises.

The Boston Dynamics CLI tools in `src/ref/` are unmodified vendor copies; run them from inside that
directory so their `import graph_nav_util` resolves.

## Architecture

### `src/` — core library

| File | Class | Role |
|------|-------|------|
| `spot_agent.py` | `SpotAgent`, `MissionStopped` | Auth, lease, the stop latch, posture, arm/grasp primitives, navigation, speaker |
| `spot_grasp.py` | `GraspPlanner` | Approach → line up (1.0 s servo) → grasp → grip by material |
| `missions.py` | `run_*`, `SortHooks`, `SortPoints` | Mission handlers, including `run_sort_loop` |
| `vision.py` | `Vision`, `VisionClient`, `Scene`, `SceneObject` | CoreIO vision client, scans, camera geometry |
| `vision_view.py` | `Panorama`, `highlight` | Grey-scene/colour-object rendering; the body cameras as a 2×2 grid |
| `recon.py` | `Reconstructor`, `VoxelMap` | 3D Recon: hand + every body camera + RealSense depth fused on the laptop, front weighted most |
| `tracker.py` | `ObjectTracker`, `Track` | Detections registered in space across scans — what grasping chooses from |
| `coreio_link.py` | `CoreIOLink` | The SSH-channel HTTP transport every CoreIO service is reached over |
| `coreio_forward.py` | `PortForward` | `ssh -L` in-process: `start.py` puts Dockge (CoreIO :5001) on `DOCKGE_LOCAL_PORT` 8081 |
| `spot_realsense.py` | `RealSenseClient` | The CoreIO's RealSense: colour, depth, distances in metres |
| `spot_cloud.py` | `LiveCloud` | The live depth stitch on the CoreIO |
| `web/console.py` | `SpotWebConsole` | Both Flask apps, mission dispatch, Stop, teleop, the grasp handshake |
| `web/live.py` | `LiveView` | Every MJPEG view (`hand`, `cameras`, `rs_blend`) and the held choice |
| `web/logbus.py` | `LogBus` | loguru + SpotTracker → the control page's log, over SSE |
| `web/mapview.py` | — | GraphNav graph → the 3D map layout and point cloud |
| `spot_tracker.py` | `SpotTracker` | Decorator/context-manager for timed, coloured logging |
| `speaker.py` | `CoreIOSpeaker` | SSH/SFTP transport for the CoreIO's USB speaker |
| `janitor.py` | `Janitor` | Static I/O utilities: graph proto loading, temp audio, credentials |
| `config.py` / `log_config.py` | — | Credentials; loguru console + dated-file sinks |

### Stopping — the rule everything else obeys

**Stop freezes the robot where it is; it never sits, stands or stows.** `SpotAgent.halt()` sets a
latch and sends `RobotCommandBuilder.stop_command()` straight to the command client — no lease
check, no tracker, no lock a mission could be holding. While the latch is set, **every** command
the agent sends goes through `_command()`, which raises `MissionStopped` instead. `resume()` clears
it, and the console calls it only when the operator dispatches something new.

The console's `/api/stop` does `halt()` first, then raises `MissionStopped` inside the mission
thread (`PyThreadState_SetAsyncExc`). It works whether or not a mission is running. Esc on the
control page is Stop (or releases teleop when teleop is what is active), and the page retries
the request until it is acknowledged.

Rules that keep it working — follow them in new code:

- **Never send a robot command except through `SpotAgent._command()`.** A bare
  `cmd_client.robot_command` is a way for a stopped mission to move the robot again.
- **Every polling loop calls `agent.check_halt()`** (navigation, grasp feedback, servo, waits).
- `MissionStopped` subclasses `KeyboardInterrupt`. **Catch `Exception`, never `BaseException`.**
  `SpotTracker` swallows only `Exception` subclasses — it used to swallow everything, which is how a
  Stop landing inside a tracked method was eaten and the mission carried on.
- In `except Exception` blocks inside agent methods, re-raise `MissionStopped` explicitly first
  when the `try` covers `_command` calls (see `arm_in`).
- The safety fallback (`missions.safety_fallback`) on an **error** holds position
  (`hold_position()`, unlatched) and returns `False` — it does not stow, sit or power off.

### The ready pose

`SpotAgent.arm_ready()` — hand at (0.80, 0, 0.20) in the gravity-aligned body frame, gripper open,
pitch `ARM_READY_PITCH_DEG` — is the teleop pose and how **every action starts** (`_ready()` in the
console: stand, then `arm_ready()`), which is also how an arm frozen by a Stop or left by a failed
grasp gets back. Surveys call it too; it does nothing when the hand is already there. A failed
grasp calls `release_to_ready()` (open, back to the ready pose) instead of stowing.

### Posture

`SpotAgent.posture()` asks the robot: `off` (motors off), `sitting` / `standing` (body height above
the `gpe` frame, threshold 0.33 m), or `unknown`. `Stand` on a standing robot and `Sit` on a sitting
or powered-off robot do nothing and log that. `_stand_up()` — the first line of every moving
mission — stands the robot only if it is not already standing, because re-sending a stand resets
the body to its nominal pose.

### Lease

Ownership is decided by the owner's client name (`SpotAgent.owner_name`, built the way the SDK
builds it) before comparing lease sequences: the sequence comparison alone called our own lease
"taken" once commands had advanced it, and the retake returned the lease — a standing Spot with no
lease holder sits down. `_get_lease` now takes first and never returns the old keepalive's lease.

`start.py` opens **one** agent with `lease=False`: starting the console is watching, not driving.
`ensure_lease()` takes the lease on the first command and takes it back when another client (the
ROS 2 driver, the tablet) has it — only when the operator dispatches something. The header's `Lease`
pill shows the holder. `power_on_and_stand()` retakes and retries exactly once on
`STATUS_COMMAND_OVERRIDDEN`.

### Vision — segment, then classify, on the CoreIO

`coreio/vision/service/vision_server.py` runs in the `spot-vision` image (Dockge stack
`/data/stacks/spot-vision`, port **8784** on CoreIO-localhost). Two YOLOE models per frame:

1. **segment** — `yoloe-11s-seg-pf` (prompt-free): every object's mask, whatever it is.
2. **classify** — `yoloe-11s-seg` with `coreio/vision/vocab/sort3.json` baked in. Prompts belong to
   groups `teddy_bear` / `bottle` / `other` / `ignore` and carry a material `soft` / `medium` / `hard`.

Each mask is scored per group from overlapping classify boxes. Not enough evidence → still reported
as `other` with `uncertain: true`; close runner-up → uncertain; `ignore` → dropped. The grasp point
`cx, cy` is the point inside the mask furthest from its edge. `classify: false` in the request
skips stage 2 (the servo uses it). Thresholds are env vars in the stack's `compose.yaml`.

**Baking** (`coreio/vision/bake_vocab.py`) must run under **ultralytics 8.3.156** — the version in
the robot's `yolo:latest` — and writes `runtime/weights/vision/{segment.pt, classify.pt, vocab.json}`.
The text encoder is stripped from the checkpoint so the offline CoreIO never needs it.
`coreio/deploy_vision.py` uploads, `docker build`s on the robot (`FROM yolo:latest`, COPY only),
writes the stack and `docker-compose up -d`s it, stopping `spot_yolo` first. It also retires the
old YOLO-World `coreio-vision` stack (moved to `~/retired_stacks/`): that one reused the container
name `spot_vision` and mounted `~/vision_service`, so it ran this server on 8782 — RealSense's port.

On the laptop, `build_vision()` returns the shared `Vision`. `scan(agent, sources, depth=, wait=,
classify=)` captures frames, analyses them and publishes a `Scene` to listeners (`LiveView`).
`scan_views()` is the same in one request but returns one `Scene` per view — `cameras` (body) and
`hand` — sharing a `seq`, so an object both saw counts as one sighting. Every frame is turned
upright from its own transforms snapshot before it is sent (`_upright`: quarter turns for the body
mounts, the actual roll for the hand) and the replies are turned back (`_turn_back`).
`wait=False` is the preview: skipped if a scan is in flight or during the retry back-off. There is
**no local fallback** — the old YOLOv8/OpenVINO detectors are gone. Objects closer than
`VISION_SELF_RANGE_M` to a body camera (depth-checked) are Spot's own legs and are dropped, as is any body-camera outline covering where the reached-out
arm projects (`_arm_points`).

Measured with the ROS 2 stack running on the CoreIO: surround (5 cameras) ~1.7 s, hand
segment-only ~0.4 s (with the `s` segmenter). The bottleneck is the CoreIO's CPU.

### The sorting demo

`run_sort_loop(agent, vision, SortPoints, hooks)` — until stopped: walk to the pick point,
`survey()` (`arm_ready()`, then `SURVEY_SCANS` scans with depth of `survey_sources(hooks.surround())`:
every body camera or only the front pair — the console's *Surround vision* switch — plus the hand;
front only keeps tracks within `VISION_FRONT_HALF_DEG` of ahead), `hooks.choose(Survey)` → `Choice(track, group)`
or `None` (survey again), `GraspPlanner.pick(choice.scene, choice.obj, target=track.position)`,
forget the track, walk to the group's bin, `drop_in_bin()`, repeat.

**Objects are registered in space, not picked from one frame** (`tracker.ObjectTracker`, owned by
the shared `Vision` as `vision.tracker`). Every classified scan (preview or mission; not the
servo's `classified=False` scans) places each detection in the vision frame with `locate()` and
merges it into a track within `TRACK_RADIUS_M`; one scan may add one detection per camera to a
track (the front pair overlap); beyond `TRACK_MERGE_M` (12 cm) only if the track projects onto its
box. **Piles**: per pair of nearby tracks the cameras vote *apart* (one camera outlined both in one
scan) or *together* (one outline covers the other's position at its depth); tracks within
`TRACK_MERGE_M` merge into the older id only when together outvotes apart and they are not certainly
different bins (`get()` follows merged ids). The grasp servo's `_find` prefers the hand outline the
target projects inside over the nearest grasp point. Seen in `TRACK_MIN_SCANS` scans, or once at `TRACK_SURE_CONF` →
registered. Only classification evidence votes on what a track is: an unclaimed mask (`other`,
uncertain, label = segmenter's) adds position and a hit plus its below-threshold `scores`, never an
Other vote — so a bear recognised once stays a bear. `candidates()` gives the best ten (classified,
certain, confidence, then hits) seen since `TRACK_RECENT_SEC` before the survey. Tracks expire
after `TRACK_TTL_SEC` — or sooner when gone: a classified scan that should have seen a track (well
inside a frame, within `TRACK_GONE_RANGE_M` 2.5 m, depth there not nearer than it) and did not is a
miss; `TRACK_GONE_SCANS` (3) misses over `TRACK_GONE_SEC` (2 s), no sighting between, deletes it.

**Looking closer** (`missions.look_closer`, end of every `survey()`, so both sorting and grasp-here):
when `LOOK_MIN` (2) or more tracks a prompt claimed for teddy_bear/bottle (sure or uncertain,
registered or not) were seen in the last `LOOK_WINDOW_SEC` (15 s), within `LOOK_RANGE_M` and
`LOOK_HALF_DEG` of ahead, the hand gazes at each (best `LOOK_MAX`, swept by bearing), waits for
`wait_hand_still()`, and takes one classified hand scan with depth. `ObjectTracker.confirm` scores
it: a recognised sighting votes `TRACK_LOOK_WEIGHT` (3×) and adds its conf to `look_score`;
nothing there in clear view is −1. `candidates()` ranks a negative `look_score` last and a
hand-confirmed track ahead of equally certain unchecked ones. Then `arm_ready()`.

**Display names are the AI's own class** (`SceneObject.label`); the three bins (`group`) are only
the colour and what the sorting does with it.

- **Sort — you choose**: `SpotWebConsole._await_choice` holds the survey's last scan
  (`LiveView.hold`) and sets `LiveView.candidates`. The held picture is then drawn from
  `track_objects()`: the registered objects themselves, numbered by **track id** (the same number
  as the chips and the log's `#22`), one outline per camera — its detection in that scan, else
  the last outline the AI found in that camera if the camera has not moved since
  (`Track.outlines`, "remembered"), else a `project_box` of its position ("projected").
  `/api/vision/scene` adds `tracks` (metadata for the chips). It blocks on a queue, checking
  `check_halt()` every 250 ms. The page answers via `/api/sort/confirm` with `{scene, track,
  group}` (group may overrule the classifier), `{source: "hand"|"cameras"}` to choose from the
  other camera (a hand scan with depth is taken there), or `{rescan: true, source}`. A confirm
  for an older scan is rejected.
- **Sort — automatic**: `AutoChoice` takes the best registered track.

**`GraspPlanner` (1.6)**: `locate()` the object in the vision frame from the pixel and the depth
behind its mask (ground-plane ray fallback); walk to a standoff (`SORT_STANDOFF_M`); then every
`SORT_SERVO_PERIOD` (1.0 s): gaze the hand at the estimate, segment-only hand scan, re-identify the
target by projected position (not by label — labels flicker), blend in the new depth measurement,
and `step_towards()` a short-lived SE2 goal with the gaze synchronized. Two consecutive checks
within `SORT_RANGE_TOL_M` / `SORT_YAW_TOL_DEG` → `grasp_in_image(auto_walk=False)`. A missed grasp
(or a target lost while lining up) backs straight away from the object, still facing it, to
`GRASP_RANGE` + n×`SORT_BACKOFF_M` (0.35 m) and goes round the servo and grasp again —
`SORT_GRASP_RETRIES` (2) times; only the last try may fall back to auto-walk. Then `set_grip(material)` re-closes at `GRIP_TORQUE_*`
(the manipulation API grasp has no force setting) and `carry()`.

**Grasp here — no map** (`run_grasp_here`, menu `grab_confirm` / `grab_auto`): `survey()`,
keep registered objects within `GRAB_REACH_M` of the body, choose through the same hooks
(the chooser shows a single "Grasp this" for `grab_*`), `GraspPlanner.pick()` — odometry only, no
GraphNav — then `agent.set_down()` (`SET_DOWN_X` / `SET_DOWN_Z`) and stow. One object, then done.

Mission waypoints — `pick`, `bin_teddy_bear`, `bin_bottle`, `bin_other`, `home`, `patrol` — are
chosen in the points window (menu items with `setup`: sort, patrol, home — it opens over the map,
is draggable, hides when the view leaves the map, and its Start validates against Spot's map) or
by clicking the map, and saved to `runtime/web_selection.json`. Dispatch still refuses a sorting
mission until all four sorting points are set.

### The web console — one page

`SpotWebConsole.serve()` runs one Flask app on `WEB_PORT` (8080) with `werkzeug.make_server`.
`control.html`: the left column is Stop and the AI switch, then the `MENU` blocks — System
(`status_check`, `initialisation`), Posture (teleop, stand, sit), Actions. The rest is the display
area with tabs; the log is one of them (`complain()` also raises a toast). Templates:
`control.html` includes `_header.html` (status pills, two rows) and `_viewer.html`; static JS is
`common.js` (helpers, state poll, header) + `viewer.js` (tabs, AI overlay/chooser, map, live,
recon, toasts) + `control.js` (commands, points window, teleop, log). Plain dependency-free JS.

**Every dispatch runs `SpotAgent.status_check()` first** (time sync, lease, behaviour faults,
E-Stop, system faults, battery — no motion, no audio); problems are logged as warnings.
*Status Check* is the same call with `verbose=True`, plus re-probing vision and the RealSense.
*Initialisation* (formerly Reset) stands, announces, reports, sits and powers off.

Views (`web/live.py` `VIEWS`): `hand`, `cameras` (live `Panorama` of `SURROUND` as a 2×2 grid —
front pair, back, left, right; the front pair rotated 90° CW, right 180°) and `rs_blend`
(per-request `alpha`; 0 is the colour picture, 1 the depth map). `map`, `live` and `recon` are
WebGL on the page (`map3d.js`, with `setColorCloud` for recon). Every view is captured only while a
browser pulls it. **AI is a layer, not a view**: with `detecting` on, the preview rescans whichever
of `hand` / `cameras` is open and the server draws the latest scan over the live picture (dropped
after its TTL). `/api/vision/scene?view=` returns every object's outline in that view's pixels
plus `shown`; the page hit-tests them with an SVG overlay on both views. While the operator
chooses, the view being chosen from is held on that scan.

### 3D Recon — on the laptop

`recon.Reconstructor` runs while the tab polls `/api/recon/status`: one thread pulls hand +
frontleft/frontright colour and depth (`capture(..., depth=True)`), placed with each image's own
transforms snapshot; a second, slower one (`RECON_SIDE_PERIOD`) pulls left/right/back — coarser,
`RECON_FAR_SIDE_M`, weighted `RECON_SIDE_WEIGHT` so the front wins where they overlap, and kept
out of the calibration `reference`. Body-camera points inside Spot's footprint (`_BODY_BOX`) are
dropped. A third pulls the RealSense bridge's `/rgbd` (colour, raw aligned depth,
intrinsics), placed via the body pose and `RECON_RS_MOUNT`, and only while the body is still.
`capture()` drops a depth image whose camera pose differs from its colour image's (> 1.5 cm /
1.5°): the hand's ToF depth can be seconds older than the hand colour picture, which misplaced
everything while the arm moved — in recon, in `locate()` and in the grasp servo alike. The
RealSense mount is calibrated with *Calibrate RealSense* (`Reconstructor.calibrate`: ICP of one
RealSense frame onto the front/hand-only `reference` map, saved to `runtime/recon_rs_mount.json`):
first the floor (RANSAC in the RealSense's own depth) against gravity fixes pitch, roll and height
— anchored to the floor the front cameras measured — then **planar point-to-plane** ICP refines only yaw and x/y (point-to-point let the floor drag the fore-aft
offset back to the guess, leaving a "shell" in front of objects; it is refused when nothing vertical is in view).
Full 6-DoF ICP re-tilted it, because the front stereo is noisier than the RealSense's floor.
Depth edges are dropped, points weighted 1/range² and fused into `VoxelMap` (sorted int64 keys,
weighted centroids and colours); voxels hit fewer than `RECON_MIN_HITS` times are not sent. Free space is carved
(`VoxelMap.carve`, its own thread every 0.3 s, using each camera's latest frame): a voxel that depth frames keep seeing *through*
(measured surface farther by 5 cm + 3 cm·z²), with no hit in between, for `RECON_CARVE_SEC` (1 s) is
deleted — someone who walks away is gone in ~1–2 s; occluded voxels are never carved. A pixel with no reading
counts as seen through too, within the camera's fusion range, when that pixel normally measures (`_Coverage`:
share of frames with a reading, rising fast, falling over ~35 frames) and the 7×7 around it read nothing —
so open space where something stood clears, while what only the stereo dead zones see is kept.
Nothing runs on the CoreIO beyond serving frames. With AI on, `/api/recon/status` adds `objects`
(the registered tracks — up to `RECON_LABELS`, 40, not the chooser's ten — same vision frame) and the page pins a label on each; while the tab is open
the preview scans `survey_sources()` with depth instead of the camera views.

**One mission at a time**; a dispatch while something runs is refused. A handler returning
`False` is reported as "did not complete". **Teleop is a mode**: 8 Hz self-expiring velocity
commands, so a frozen browser stops the robot.

### RealSense

`coreio/service/realsense_server.py` subscribes to the `realsense_ros2` container's topics (it never
opens the device) and serves pictures and depth probes on 8782; deploy with
`python coreio/deploy_realsense.py`. It needs `--network host`, `ROS_DOMAIN_ID` 30 and the camera's
`CYCLONEDDS_URI` — a mismatch discovers nothing and waits forever. The camera is mounted inverted,
so the bridge rotates both streams 180° on arrival (`RS_ROTATE`), and `/rgbd` rotates the
`camera_info` intrinsics to match. Depth prefers the aligned topic;
readings are medianed over 9×9. When the camera node stops publishing the bridge keeps serving the
last frame — `/health` reports each stream's age and `stale`, shown as a "frozen" pill. The console
re-probes a missing bridge every 30 s.

### The map

Drawn only from the graph GraphNav holds (never silently from `SPOT_MAP_DIR`), with waypoint snapshot point clouds put
into the seed frame, cached under `runtime/clouds/<key>.f32` and filtered by `WEB_CLOUD_RANGE_M`.
`web/static/map3d.js` is hand-written WebGL. On the control page, clicking a waypoint offers
*Navigate here* and setting it as pick point, one of the three bins, home, or a patrol stop
(patrol appends; order is the mission).

### Speaker

Mission code only calls `agent.play("assets/audio/foo.mp3")`. One SSH session per agent; audio is
transcoded to WAV locally and pushed over SFTP to the CoreIO's `aplay`. Nothing about it is fatal:
failures are logged and `play()` returns a bool. `COREIO_SPEAKER_VOLUME` sets the card's mixer.

### Other patterns

**Janitor** centralises disk I/O (`load_graph`, `load_waypoint_snapshots`, `load_edge_snapshots`,
`load_credentials`). Snapshots are found in both `<map>/waypoint_snapshots/<id>` and `<map>/wp_<id>`.

**Logging** is configured in `src/log_config.py` and installed by `src/__init__.py`; `start.py` also
calls `setup_logging()` explicitly (idempotent) — never rely on a bare side-effect import.

**SpotTracker** intercepts `print()` within its scope and adds timing. Apply as
`@SpotTracker("label")` or `with SpotTracker("label"):`. It swallows `Exception` unless
`exit_on_fail=True`, and never swallows `KeyboardInterrupt`/`MissionStopped`.

**assets/** — speaker audio and the RMIT marks (served by `/assets/images/<file>`).

**runtime/** — data, not code: `maps/`, `logs/` (dated), `clouds/`, `weights/vision/` (baked models,
git-ignored), `web_selection.json`.

### `coreio/` — what goes onto the robot

The CoreIO has **no DNS and no default route**; everything arrives over SSH. Services bind
CoreIO-localhost because Spot forwards only 20022 and 21443 off it. Ports: RealSense 8782,
live cloud 8783, vision 8784.

- `coreio/remote.py` — SSH helpers (`connect`, `run`, `upload`, `stream`) every deploy script imports
  as `from remote import ...`.
- `coreio/vision/` — vocab, bake script, and `service/` (Python **3.8** code, Dockerfile, compose).
- `coreio/service/` — `realsense_server.py`, `cloud_server.py` (Python 3.8).
- `coreio/demo_web/` + `deploy_web.py` — the console as its own container on the robot, serving
  both ports; reached from a laptop with an SSH forward for 8080 and 8081.

Dockge (`/data/stacks`, UI on CoreIO port 5001, see `reference/core_io.md`) manages every stack;
new services go in as stacks, not bare `docker run`.

## Linting

Ruff is configured in `pyproject.toml`. `src/ref/` and `runtime/` are excluded — never "fix" code
there. `coreio/service/` and `coreio/vision/service/` are pinned to `py38` by their own
`.ruff.toml`. `ruff check .` must stay clean.
