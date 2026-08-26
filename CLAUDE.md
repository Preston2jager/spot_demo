# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Boston Dynamics Spot robot control application combining YOLOv8 object detection with autonomous navigation and grasping. Python 3.12+, managed with `uv`.

## Setup

```bash
uv sync                  # install dependencies into .venv
source .venv/bin/activate
```

Robot credentials live in `credentials.json` at the project root (not committed). Copy and fill in:
```json
{ "hostname": "192.168.80.3", "username": "user", "password": "..." }
```

Run scripts from the **project root** so `from src.xxx import ...` resolves correctly:
```bash
python spot_guard.py
python spot_patrol_only.py
python spot_go_home.py
```

## YOLOv8 → OpenVINO Export

```bash
yolo export model=yolov8m_rmit.pt format=openvino half=True imgsz=640 batch=1
yolo export model=yolov8m_rmit_2.pt format=openvino half=True imgsz=640 batch=1
```

Intel Ultra 125h benchmarks: CPU 200 ms/f, NPU 62 ms/f, GPU 28 ms/f.

## Map Workflow

1. Record a map with `src/ref/recording_command_line.py` (fiducial must be in view at start).
2. Download the map locally into `runtime/maps/<map_name>/`.
3. Inspect waypoints with `src/ref/check_map.py`.
4. Point `SPOT_MAP_DIR` at the map directory (default `./runtime/maps/amp_3`) and use waypoint names in navigation scripts.

The Boston Dynamics CLI tools in `src/ref/` are unmodified vendor copies; run them from inside that
directory so their `import graph_nav_util` resolves.

## Architecture

### `src/` — core library

| File | Class | Role |
|------|-------|------|
| `spot_agent.py` | `SpotAgent` | Context-manager controller: auth, lease, navigation, arm, detection |
| `spot_detector.py` | `SpotDetector` | YOLOv8 (PyTorch backend) |
| `spot_detector_openvino.py` | `SpotDetector` | YOLOv8 (OpenVINO backend — Intel GPU/NPU) |
| `spot_streamer.py` | `SpotStreamer` | Flask live camera dashboard on port 5555 |
| `spot_tracker.py` | `SpotTracker` | Decorator/context-manager for timed, coloured logging |
| `janitor.py` | `Janitor` | Static I/O utilities: graph proto loading, temp audio, credentials |
| `mcp_server.py` | — | FastMCP SSE server exposing `navigate_to_waypoint` |
| `config.py` | — | `load_robot_credentials()` reads `credentials.json` |
| `log_config.py` | — | `setup_logging()` installs the loguru console + dated-file sinks |

### Key patterns

**SpotAgent** is always used as a context manager — `with SpotAgent(...) as agent:`. It handles authentication, lease acquisition, power-on, and cleanup automatically.

**Janitor** centralises all disk I/O:
- `Janitor.load_graph(save_dir)` / `load_waypoint_snapshots` / `load_edge_snapshots` — used by `SpotAgent.upload_graph_and_snapshots`
- `Janitor.load_credentials(path)` — used by `config.py`

**Logging** is configured in `src/log_config.py` and installed by `src/__init__.py`, so importing anything from
`src` sets it up. `start.py` also calls `setup_logging()` explicitly (it is idempotent) — never rely on a bare
side-effect import for this, a linter will delete it.

**Two detector variants** share the same interface; swap the import to switch backends.

**SpotTracker** intercepts `print()` globally within its scope, adds colour and wall-clock timing. Apply as `@SpotTracker("label")` or `with SpotTracker("label"):`.

### Running the MCP server

```bash
# MCP server (SSE on port 8000 by default, or set MCP_TRANSPORT=stdio)
python run_mcp.py
```

**runtime/** — runtime data, not code. `runtime/maps/` holds GraphNav graph folders (`amp_3`); `runtime/logs/`
gets one `YYYY-MM-DD.log` per day (git-ignored), written by the loguru file sink in `src/log_config.py`.

**src/ref/** — verbatim copies of the Boston Dynamics GraphNav CLI examples (`recording_command_line.py`,
`graph_nav_command_line.py`, `graph_nav_util.py`, `check_map.py`). Reference only — nothing in `src/` imports them.

**Archive/** — deprecated experiments plus the retired `Maps/` folder and the `spot-sdk` checkout; do not rely
on code there.

## Linting

Ruff is configured in `pyproject.toml`. `Archive/`, `src/ref/`, and `runtime/` are excluded — never "fix" code
there. `ruff check .` must stay clean.
