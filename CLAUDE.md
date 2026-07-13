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

1. Record a map with `Maps/recording_command_line.py` (fiducial must be in view at start).
2. Download the map locally.
3. Inspect waypoints with `Maps/check_map.py`.
4. Use waypoint names in navigation scripts.

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

### Key patterns

**SpotAgent** is always used as a context manager — `with SpotAgent(...) as agent:`. It handles authentication, lease acquisition, power-on, and cleanup automatically.

**Janitor** centralises all disk I/O:
- `Janitor.load_graph(save_dir)` / `load_waypoint_snapshots` / `load_edge_snapshots` — used by `SpotAgent.upload_graph_and_snapshots`
- `Janitor.load_credentials(path)` — used by `config.py`

**Two detector variants** share the same interface; swap the import to switch backends.

**SpotTracker** intercepts `print()` globally within its scope, adds colour and wall-clock timing. Apply as `@SpotTracker("label")` or `with SpotTracker("label"):`.

### Running the MCP server

```bash
# MCP server (SSE on port 8000 by default, or set MCP_TRANSPORT=stdio)
python run_mcp.py
```

**Maps/** — pre-recorded GraphNav graph files (`amp_3`, `lv12_office`) and Boston Dynamics CLI tools.

**Archive/** — deprecated experiments; do not rely on code there.
