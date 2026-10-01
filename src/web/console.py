"""The web console: one robot session, one page on ``WEB_PORT`` (8080).

Missions, teleop, Stop, the points chosen on the map, the grasp confirmation, and one
display area for every camera, the AI overlay, the RealSense depth blend, the maps, the
laptop-side 3D reconstruction and the run log. Several browsers can have it open at once;
they all read the same state.

Five things are worth knowing.

**Stop outranks everything.** `/api/stop` freezes the robot where it stands before it does
anything else — `SpotAgent.halt()` latches the agent and sends a stop command straight to
the command service — then unwinds the mission thread with `MissionStopped`. It takes no
lock a mission could be holding and needs no mission to be running: a Stop during a sit, a
teleop step or a GraphNav walk is the same Stop. The latch stays until the operator
dispatches something new, so nothing a dying mission tries on its way out reaches the robot.

**Stance is asked, not assumed.** `Stand` on a standing robot and `Sit` on a sitting one do
nothing and say so — no stand-up before sitting, no reset to the nominal pose.

**Every command starts with a Status Check.** `SpotAgent.status_check()` — time sync,
lease, behaviour faults, E-Stop, system faults, battery — runs before each mission and
teleop. It moves nothing; the operator's own Status Check is the same call, reported in full.

**One mission at a time.** A dispatch while something runs is refused; Spot has one body.

**The grasp missions wait for a click.** The `confirm` variants scan, then block in
`choose()` until the operator picks an object (and may overrule its bin). The scan is held
on screen while it waits, so the picture cannot change under the click. The operator can
switch between the Body Cameras scan and a Hand Camera scan and choose from either.
"""
import ctypes
import json
import logging
import os
import queue
import threading
import time
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_from_directory
from loguru import logger
from werkzeug.serving import make_server

from .. import missions as mission_lib
from ..recon import Reconstructor
from ..spot_agent import MissionStopped
from ..spot_cloud import build_live_cloud
from ..spot_realsense import build_realsense, realsense_enabled
from ..vision import GROUPS, HAND, Scene, build_vision, survey_sources
from . import mapview
from .live import VIEWS, LiveView, view_of
from .logbus import LogBus

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SELECTION_FILE = _PROJECT_ROOT / "runtime" / "web_selection.json"
_RS_REPROBE_SEC = 30.0
_VISION_REPROBE_SEC = 20.0

STANCE_STAND = "stand"
STANCE_SIT = "sit"

# Three blocks on the page. `needs` lists what must be chosen on the map before a mission
# can be dispatched; `setup` opens the points window on the map instead of dispatching
# straight away, and its Start button dispatches once every point is filled in.
MENU = [
    {"group": "System", "items": [
        {"id": "status_check", "label": "Status Check",
         "hint": "Connection, lease, faults, E-Stop, battery, services — nothing moves"},
        {"id": "initialisation", "label": "Initialisation",
         "hint": "Stand, announce, report battery, then sit and power off"},
    ]},
    {"group": "Posture", "items": [
        {"id": "teleop", "kind": "teleop", "label": "Walk — WASD", "hint": "Hold keys to drive · Esc releases"},
        {"id": "stand", "label": "Stand", "stance": STANCE_STAND, "hint": "Get up and hold"},
        {"id": "sit", "label": "Sit", "stance": STANCE_SIT, "hint": "Stow, sit, power off"},
    ]},
    {"group": "Actions", "items": [
        {"id": "grab_confirm", "label": "Grasp here — you choose",
         "hint": "Scan around Spot; click an object nearby, Spot grasps it and sets it down"},
        {"id": "grab_auto", "label": "Grasp here — automatic",
         "hint": "The same, taking the most confident object within reach"},
        {"id": "sort_confirm", "label": "Sort — you choose", "graphnav": True, "needs": ["sort"],
         "setup": "sort", "hint": "Pick point → scan → you click → Spot bins it, in a loop"},
        {"id": "sort_auto", "label": "Sort — automatic", "graphnav": True, "needs": ["sort"],
         "setup": "sort", "hint": "The same loop, taking the most confident object each time"},
        {"id": "patrol", "label": "Patrol Loop", "graphnav": True, "needs": ["patrol"],
         "setup": "patrol", "hint": "Walk the patrol route in order until stopped"},
        {"id": "go_home", "label": "Go Home", "graphnav": True, "needs": ["home"],
         "setup": "home", "hint": "Walk to the home point"},
    ]},
]
_UNLISTED = {
    "navigate_to": {"label": "Navigate to waypoint"},
    "load_map": {"label": "Load map onto Spot"},
}

_SELECTION_KEYS = ("pick", "bin_teddy_bear", "bin_bottle", "bin_other", "home")


class SpotWebConsole:
    """Runs missions against one `SpotAgent` and serves the console page."""

    def __init__(self, agent: object, host: str = "0.0.0.0", port: int | None = None) -> None:
        self.agent = agent
        self.host = host
        self.port = int(port or os.getenv("WEB_PORT", "8080"))
        self.logs = LogBus()
        self.vision = build_vision()
        self.view = LiveView(agent, self.vision)
        self.selection = _load_selection()

        self._lock = threading.Lock()
        self._mission: str | None = None
        self._mission_started = 0.0
        self._thread: threading.Thread | None = None
        self._teleop = False
        self.stance = STANCE_SIT
        self._posture = "unknown"
        self.phase = ""
        self._battery: dict = {}
        self._lease: dict = {}
        self._cloud = mapview.CloudBuilder()
        self.live_cloud: object | None = None
        self.recon = Reconstructor(agent, lambda: self.view.realsense)
        # The grasp missions' surround switch: every body camera, or only the front pair.
        # Either way the hand camera looks too.
        self.surround = os.getenv("VISION_SURROUND", "1").strip().lower() not in {"0", "false", "no", "off"}
        # With AI on and the 3D Recon tab open, the preview scans what a survey would, so
        # the registered objects can be labelled in the reconstruction.
        self.view.label_scan = lambda: survey_sources(self.surround) if self.recon.active else None

        # The grasp confirmation handshake: the mission publishes a scene and waits; the
        # page answers through `_confirm`.
        self._pending: Scene | None = None
        self._answer: queue.Queue = queue.Queue(maxsize=1)

        self.app = Flask("spot_console", root_path=str(Path(__file__).parent))
        self._routes(self.app)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def serve(self) -> None:
        """Bring everything up and serve the page until interrupted."""
        self.logs.install()
        self.view.realsense = build_realsense()
        self.live_cloud = build_live_cloud()
        self.view.start()
        threading.Thread(target=self._status_loop, daemon=True, name="status").start()

        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        server = make_server(self.host, self.port, self.app, threaded=True)
        logger.success(f"Console at http://{_display_host(self.host)}:{self.port}")
        server.serve_forever()

    def _status_loop(self) -> None:
        """Poll battery, lease and posture for the headers, and re-probe missing services."""
        last_rs = last_vision = time.time()
        while True:
            self._battery = self.agent.battery()  # type: ignore[union-attr]
            self._lease = self.agent.lease_status()  # type: ignore[union-attr]
            self._posture = self.agent.posture()  # type: ignore[union-attr]
            now = time.time()
            if self.view.realsense is None and realsense_enabled() and now - last_rs > _RS_REPROBE_SEC:
                last_rs = now
                client = build_realsense(quiet=True)
                if client is not None:
                    self.view.realsense = client
            if not self.vision.status()["ready"] and now - last_vision > _VISION_REPROBE_SEC:
                last_vision = now
                self.vision.probe()
            time.sleep(3.0)

    # ── stance ────────────────────────────────────────────────────────────────

    def _stand_up(self) -> None:
        """Stand Spot up unless it is already standing — then leave it exactly as it is.

        Re-sending a stand to a standing robot resets its body to the nominal pose, which is
        a movement; the robot is asked first so that does not happen for nothing.
        """
        posture = self.agent.posture()  # type: ignore[union-attr]
        if posture != "standing":
            logger.info(f"Stance: standing up (robot reports {posture}).")
            self.agent.power_on_and_stand()  # type: ignore[union-attr]
        self.stance = STANCE_STAND

    # ── the sorting handshake ─────────────────────────────────────────────────

    def _sort_hooks(self, confirm: bool) -> mission_lib.SortHooks:
        console = self

        class Hooks(mission_lib.AutoChoice if not confirm else mission_lib.SortHooks):
            def phase(self, text: str) -> None:
                console.phase = text

            def surround(self) -> bool:
                return console.surround

            if confirm:
                def choose(self, scene: Scene) -> mission_lib.Choice | None:
                    return console._await_choice(scene)

        return Hooks()

    def _await_choice(self, offered: mission_lib.Survey) -> mission_lib.Choice | None:
        """Hold the survey's scan on screen and wait for the operator's click, rescan or Stop.

        What is offered is the registered objects (at most ten), not one scan's detections.
        The operator may switch to the survey's Hand Camera scan, or a fresh one taken here; it
        registers too, and the same objects are outlined there.
        """
        while not self._answer.empty():
            self._answer.get_nowait()
        tracker = self.vision.tracker
        hand: Scene | None = offered.hand
        self.view.candidates = lambda: tracker.candidates(since=offered.started)
        self._hold(offered.scene)
        logger.info(f"Choose an object — {len(offered.tracks)} registered: "
                    + ", ".join(f"#{t.id} {t.label} ×{t.hits}" for t in offered.tracks))
        try:
            while True:
                self.agent.check_halt()  # type: ignore[union-attr]
                try:
                    kind, *args = self._answer.get(timeout=0.25)
                except queue.Empty:
                    continue
                if kind == "view":
                    source, rescan = args
                    if source == "hand" and (rescan or hand is None):
                        self.phase = "scanning with the hand camera"
                        hand = self.vision.scan(self.agent, (HAND,), depth=True)
                        if hand is None:
                            logger.warning("The hand camera returned nothing — still on the previous scan.")
                            continue
                    elif source == "cameras" and rescan:
                        logger.info("Surveying again at the operator's request.")
                        return None
                    self._hold(hand if source == "hand" else offered.scene)
                    continue
                track_id, group = args
                track = tracker.get(track_id)
                if track is None or track.latest is None:
                    continue
                self.view.select(track.id)
                note = "" if group == track.group else f" (overruling '{track.group}')"
                logger.info(f"Operator chose {track.summary()} for the {GROUPS[group]['label']} bin{note}")
                return mission_lib.Choice(track, group)
        finally:
            self._pending = None
            self.view.candidates = None
            self.view.release()

    def _hold(self, scene: Scene) -> None:
        self.view.hold(scene)
        self._pending = scene
        self.phase = "waiting for you to choose an object"

    def _confirm(self, body: dict) -> tuple[dict, int]:
        scene = self._pending
        if scene is None:
            return {"error": "nothing is waiting for a choice"}, 409
        source = body.get("source")
        if source is not None or body.get("rescan"):
            source = source or view_of(scene)
            if source not in {"hand", "cameras"}:
                return {"error": f"cannot choose from '{source}'"}, 400
            self._offer(("view", source, bool(body.get("rescan"))))
            return {"source": source}, 200
        if int(body.get("scene", -1)) != scene.seq:
            return {"error": "that view is out of date — the robot has scanned again"}, 409
        track = self.vision.tracker.get(int(body.get("track", -1)))
        if track is None:
            return {"error": "that object is no longer registered"}, 404
        group = body.get("group") or track.group
        if group not in GROUPS:
            return {"error": f"unknown bin '{group}'"}, 400
        self._offer(("pick", track.id, group))
        return {"track": track.id, "group": group}, 200

    def _offer(self, answer: tuple) -> None:
        try:
            self._answer.put_nowait(answer)
        except queue.Full:
            pass

    # ── mission dispatch ──────────────────────────────────────────────────────

    def _start_mission(self, mission_id: str, **kwargs: object) -> tuple[dict, int]:
        handler = _HANDLERS.get(mission_id)
        if handler is None:
            return {"error": f"unknown mission '{mission_id}'"}, 404
        blocked = self._missing(_spec(mission_id))
        if blocked:
            return {"error": blocked}, 409
        with self._lock:
            if self._mission is not None:
                return {"error": f"'{self._mission}' is still running"}, 409
            if self._teleop:
                return {"error": "teleop is active — stop it first"}, 409
            self._mission = mission_id
            self._mission_started = time.time()
            self.phase = ""
            self._thread = threading.Thread(
                target=self._run, args=(mission_id, handler, kwargs), daemon=True, name=f"mission-{mission_id}"
            )
            self._thread.start()
        return {"started": mission_id}, 200

    def _missing(self, spec: dict) -> str:
        chosen = self.selection
        for need in spec.get("needs", []):
            if need == "sort":
                absent = [k for k in ("pick", "bin_teddy_bear", "bin_bottle", "bin_other") if not chosen.get(k)]
                if absent:
                    return "Choose the pick point and all three bins on the map first."
            if need == "home" and not chosen.get("home"):
                return "Choose a home point on the map first."
            if need == "patrol" and len(chosen.get("patrol") or []) < 2:
                return "A patrol needs at least two waypoints — add them from the map."
        return ""

    def _run(self, mission_id: str, handler: object, kwargs: dict) -> None:
        label = _spec(mission_id).get("label", mission_id)
        try:
            logger.info(f"▶ {label}")
            # A new command is the only thing that lifts a Stop.
            self.agent.resume()  # type: ignore[union-attr]
            if mission_id != "status_check":
                self.agent.status_check()  # type: ignore[union-attr]
            if handler(self, **kwargs) is False:  # type: ignore[operator]
                logger.warning(f"■ {label} did not complete")
            else:
                logger.success(f"■ {label} complete")
        except KeyboardInterrupt:
            logger.warning(f"■ {label} stopped — Spot is holding where it was")
        except Exception:
            logger.exception(f"■ {label} failed — holding position")
            self.agent.hold_position()  # type: ignore[union-attr]
        finally:
            self.view.mission_scans = False
            self.phase = ""
            with self._lock:
                self._mission = None
                self._thread = None

    def _stop(self) -> dict:
        """Freeze the robot, then unwind whatever was driving it."""
        self.agent.halt()  # type: ignore[union-attr]
        logger.warning("STOP — Spot is holding where it is")
        self._teleop = False
        with self._lock:
            thread, mission = self._thread, self._mission
        if thread is not None and thread.is_alive():
            raised = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(thread.ident), ctypes.py_object(MissionStopped)
            )
            if raised != 1:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(thread.ident), None)
                logger.warning("Stop: the mission thread did not take the interrupt — it will stop at its next command")
        return {"stopped": mission, "halted": True}

    # ── teleop ────────────────────────────────────────────────────────────────

    def _teleop_start(self) -> tuple[dict, int]:
        with self._lock:
            if self._mission is not None:
                return {"error": f"'{self._mission}' is still running"}, 409
            self._teleop = True
        try:
            self.agent.resume()  # type: ignore[union-attr]
            self.agent.status_check()  # type: ignore[union-attr]
            self._stand_up()
            self.agent.arm_ready()  # type: ignore[union-attr]
            logger.success("Teleop armed — W A S D to walk, Q E to turn, release to stop")
        except KeyboardInterrupt:
            self._teleop = False
            return {"error": "stopped"}, 409
        except Exception:
            self._teleop = False
            logger.exception("Teleop failed to start")
            return {"error": "teleop failed to start — see the log"}, 500
        return {"teleop": True}, 200

    def _teleop_stop(self) -> dict:
        self._teleop = False
        try:
            self.agent.arm_in()  # type: ignore[union-attr]
            logger.info("Teleop released — Spot is standing.")
        except KeyboardInterrupt:
            pass
        except Exception:
            logger.exception("Teleop shutdown failed")
        return {"teleop": False}

    # ── state ─────────────────────────────────────────────────────────────────

    def state(self) -> dict:
        with self._lock:
            mission, started = self._mission, self._mission_started
        pending = self._pending
        return {
            "mission": mission,
            "mission_label": _spec(mission).get("label") if mission else None,
            "elapsed": round(time.time() - started, 1) if mission else 0.0,
            "phase": self.phase,
            "teleop": self._teleop,
            "stance": self.stance,
            "posture": self._posture,
            "halted": self.agent.halted,  # type: ignore[union-attr]
            "battery": self._battery,
            "lease": self._lease,
            "graph": self.agent.graph is not None,  # type: ignore[union-attr]
            "selection": self.selection,
            "detecting": self.view.detecting,
            "surround": self.surround,
            "vision": self.vision.status(),
            "realsense": self.view.realsense is not None,
            "live_cloud": self.live_cloud is not None,
            "pending": pending.seq if pending is not None else None,
            "pending_view": view_of(pending) if pending is not None else None,
            "groups": {k: {"label": v["label"], "color": v["hex"]} for k, v in GROUPS.items()},
        }

    # ── routes ────────────────────────────────────────────────────────────────

    def _routes(self, app: Flask) -> None:
        @app.route("/")
        def index() -> str:
            return render_template("control.html", menu=MENU)

        @app.get("/assets/images/<path:filename>")
        def brand_asset(filename: str) -> Response:
            return send_from_directory(_PROJECT_ROOT / "assets" / "images", filename)

        @app.get("/api/state")
        def state() -> Response:
            return jsonify(self.state())

        @app.get("/video/<view>")
        def video(view: str) -> Response:
            if view not in VIEWS:
                return Response("unknown view", status=404)
            try:
                alpha = float(request.args.get("alpha", 0.5))
            except ValueError:
                alpha = 0.5
            return Response(self.view.frames(view, alpha), mimetype="multipart/x-mixed-replace; boundary=frame")

        @app.get("/api/vision/scene")
        def vision_scene() -> Response:
            view = request.args.get("view", "cameras")
            if view not in {"hand", "cameras"}:
                return jsonify({"error": f"no scans on '{view}'"}), 404
            payload = self.view.scene_description(view)
            if payload is None:
                return jsonify({"error": "no scan yet"}), 404
            return jsonify(payload)

        @app.get("/api/realsense/depth")
        def realsense_depth() -> Response:
            client = self.view.realsense
            if client is None:
                return jsonify({"error": "no RealSense bridge on this robot"}), 404
            try:
                u = float(request.args.get("u", 0.5))
                v = float(request.args.get("v", 0.5))
            except (TypeError, ValueError):
                return jsonify({"error": "u and v must be numbers"}), 400
            try:
                return jsonify(client.depth_at(u, v))
            except Exception as exc:
                return jsonify({"error": str(exc)}), 503

        @app.get("/api/realsense/health")
        def realsense_health() -> Response:
            client = self.view.realsense
            if client is None:
                return jsonify({"error": "no RealSense bridge on this robot"}), 404
            try:
                return jsonify(client.health())
            except Exception as exc:
                return jsonify({"error": str(exc)}), 503

        @app.get("/api/map")
        def graph_map() -> Response:
            graph = mapview.load_graph(self.agent)
            if graph is None:
                local = mapview.local_map()
                return jsonify({"error": "Spot has no map loaded", "local": local}), 404
            payload = mapview.layout(graph)
            payload["selection"] = self.selection
            payload["live"] = self.agent.graph is not None  # type: ignore[union-attr]
            return jsonify(payload)

        @app.get("/api/map/cloud/status")
        def cloud_status() -> Response:
            graph = mapview.load_graph(self.agent)
            if graph is None:
                return jsonify({"error": "no map available"}), 404
            return jsonify(self._cloud.request(self.agent, graph))

        @app.get("/api/map/cloud")
        def cloud_data() -> Response:
            payload = self._cloud.data()
            if payload is None:
                return jsonify(self._cloud.status()), 202
            return Response(payload, mimetype="application/octet-stream")

        @app.get("/api/map/live/status")
        def live_status() -> Response:
            client = self.live_cloud
            if client is None:
                return jsonify({"error": "no live cloud service on this robot"}), 404
            client.want(self.agent)
            return jsonify(client.status())

        @app.get("/api/map/live/cloud")
        def live_cloud_data() -> Response:
            client = self.live_cloud
            if client is None:
                return jsonify({"error": "no live cloud service on this robot"}), 404
            try:
                since = int(request.args.get("since", 0))
            except (TypeError, ValueError):
                return jsonify({"error": "since must be an integer"}), 400
            try:
                header, points = client.cloud_since(since)
            except Exception as exc:
                return jsonify({"error": str(exc)}), 503
            response = Response(points, mimetype="application/octet-stream")
            response.headers["X-Cloud-Seq"] = str(header.get("seq", since))
            response.headers["X-Cloud-Total"] = str(header.get("total", 0))
            return response

        @app.post("/api/stop")
        def stop() -> Response:
            return jsonify(self._stop())

        @app.post("/api/mission/<mission_id>")
        def mission(mission_id: str) -> Response:
            payload, status = self._start_mission(mission_id)
            return jsonify(payload), status

        @app.post("/api/navigate")
        def navigate() -> Response:
            waypoint = ((request.get_json(silent=True) or {}).get("waypoint") or "").strip()
            if not waypoint:
                return jsonify({"error": "no waypoint given"}), 400
            payload, status = self._start_mission("navigate_to", waypoint=waypoint)
            return jsonify(payload), status

        @app.post("/api/sort/confirm")
        def sort_confirm() -> Response:
            payload, status = self._confirm(request.get_json(silent=True) or {})
            return jsonify(payload), status

        @app.post("/api/map/load")
        def map_load() -> Response:
            if mapview.local_map() is None:
                return jsonify({"error": "no map on this laptop — check SPOT_MAP_DIR"}), 404
            payload, status = self._start_mission("load_map")
            return jsonify(payload), status

        @app.get("/api/recon/status")
        def recon_status() -> Response:
            self.recon.want()
            status = self.recon.status()
            # With AI on, the registered objects, to label in the reconstruction — they and
            # the cloud are both in Spot's vision frame.
            status["objects"] = [_recon_label(track) for track in self.vision.tracker.candidates()] \
                if self.view.detecting else []
            return jsonify(status)

        @app.get("/api/recon/cloud")
        def recon_cloud() -> Response:
            return Response(self.recon.cloud(), mimetype="application/octet-stream")

        @app.post("/api/recon/clear")
        def recon_clear() -> Response:
            self.recon.clear()
            return jsonify(self.recon.status())

        @app.post("/api/recon/calibrate")
        def recon_calibrate() -> Response:
            try:
                return jsonify(self.recon.calibrate())
            except Exception as exc:
                logger.warning(f"RealSense calibration: {exc}")
                return jsonify({"error": str(exc)}), 409

        @app.post("/api/recon/sources")
        def recon_sources() -> Response:
            return jsonify(self.recon.set_sources(request.get_json(silent=True) or {}))

        @app.post("/api/teleop/<action>")
        def teleop(action: str) -> Response:
            if action == "start":
                payload, status = self._teleop_start()
                return jsonify(payload), status
            if action == "stop":
                return jsonify(self._teleop_stop())
            if action == "drive":
                if not self._teleop:
                    return jsonify({"error": "teleop is not active"}), 409
                body = request.get_json(silent=True) or {}
                try:
                    self.agent.drive(  # type: ignore[union-attr]
                        v_x=float(body.get("vx", 0.0)), v_y=float(body.get("vy", 0.0)),
                        v_rot=float(body.get("vrot", 0.0)),
                    )
                except KeyboardInterrupt:
                    return jsonify({"error": "stopped"}), 409
                except Exception as exc:
                    return jsonify({"error": str(exc)}), 500
                return jsonify({"ok": True})
            return jsonify({"error": f"unknown teleop action '{action}'"}), 404

        @app.post("/api/detect/<action>")
        def detect(action: str) -> Response:
            if action not in {"on", "off"}:
                return jsonify({"error": f"unknown detect action '{action}'"}), 404
            self.view.detecting = action == "on"
            return jsonify({"detecting": self.view.detecting})

        @app.post("/api/vision/surround/<action>")
        def vision_surround(action: str) -> Response:
            if action not in {"on", "off"}:
                return jsonify({"error": f"unknown surround action '{action}'"}), 404
            self.surround = action == "on"
            logger.info("Grasp surveys: every body camera and the hand" if self.surround
                        else "Grasp surveys: the front pair and the hand only")
            return jsonify({"surround": self.surround})

        @app.get("/api/logs/stream")
        def log_stream() -> Response:
            return Response(self._log_events(), mimetype="text/event-stream",
                            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        @app.post("/api/map/live/clear")
        def live_cloud_clear() -> Response:
            client = self.live_cloud
            if client is None:
                return jsonify({"error": "no live cloud service on this robot"}), 404
            try:
                return jsonify(client.clear())
            except Exception as exc:
                return jsonify({"error": str(exc)}), 503

        @app.post("/api/map/live/localise")
        def live_cloud_localise() -> Response:
            with self._lock:
                if self._mission:
                    return jsonify({"error": "a mission is running — stop it first"}), 409
            self.agent.resume()  # type: ignore[union-attr]
            self.agent.status_check()  # type: ignore[union-attr]
            if _navigable(self):
                return jsonify({"localised": True})
            return jsonify({"localised": False,
                            "error": "could not localise — park Spot where it can see a fiducial"}), 409

        @app.post("/api/map/selection")
        def selection() -> Response:
            body = request.get_json(silent=True) or {}
            chosen = {key: (body.get(key) or "").strip() for key in _SELECTION_KEYS}
            chosen["patrol"] = _clean_route(body.get("patrol"))
            self.selection = chosen
            _save_selection(chosen)
            logger.info(
                f"Points set — pick '{chosen['pick'] or '—'}', bins teddy '{chosen['bin_teddy_bear'] or '—'}' "
                f"bottle '{chosen['bin_bottle'] or '—'}' other '{chosen['bin_other'] or '—'}', "
                f"home '{chosen['home'] or '—'}', patrol {' → '.join(chosen['patrol']) or '—'}"
            )
            return jsonify(chosen)

    def _log_events(self):
        subscriber = self.logs.subscribe()
        try:
            for entry in self.logs.backlog():
                yield f"data: {json.dumps(entry)}\n\n"
            while True:
                try:
                    entry = subscriber.get(timeout=15.0)
                except queue.Empty:
                    yield ": keepalive\n\n"
                    continue
                yield f"data: {json.dumps(entry)}\n\n"
        finally:
            self.logs.unsubscribe(subscriber)


# ── mission handlers ──────────────────────────────────────────────────────────


def _status_check(console: SpotWebConsole) -> bool:
    """The soft reset: check and tidy everything, move nothing, play nothing."""
    problems = console.agent.status_check(verbose=True)  # type: ignore[union-attr]
    vision = console.vision.status()
    if not vision["ready"] and not console.vision.probe():
        problems.append("CoreIO vision unavailable")
    else:
        logger.info(f"Vision: ready ({console.vision.status()['backend']})")
    if realsense_enabled():
        if console.view.realsense is None:
            console.view.realsense = build_realsense(quiet=True)
        if console.view.realsense is None:
            problems.append("RealSense bridge unavailable")
        else:
            logger.info("RealSense: bridge up")
    graph = console.agent.graph  # type: ignore[union-attr]
    logger.info("GraphNav: localised" if graph is not None else "GraphNav: not localised (only needed for map missions)")
    console.stance = STANCE_STAND if console.agent.posture() == "standing" else STANCE_SIT  # type: ignore[union-attr]
    if problems:
        logger.warning(f"Status check: {len(problems)} problem(s) — {'; '.join(problems)}")
        return False
    logger.success("Status check: all clear.")
    return True


def _initialisation(console: SpotWebConsole) -> None:
    """The full reset: stand, announce, report the battery, sit, power off."""
    console._stand_up()
    console.agent.play(os.getenv("RESET_ANNOUNCE_AUDIO"))  # type: ignore[union-attr]
    mission_lib.run_initialisation(console.agent)
    console.stance = STANCE_SIT


def _load_map(console: SpotWebConsole) -> bool:
    """Put `SPOT_MAP_DIR` on the robot if it holds no map, then try to localise. Nothing moves."""
    agent = console.agent
    if agent.get_current_graph() is None:  # type: ignore[union-attr]
        if agent.upload_graph_and_snapshots(mapview.local_map()) is None:  # type: ignore[union-attr]
            return False
    else:
        logger.info("Spot already holds a map — using it rather than uploading.")
    mapview.forget_graph()
    return _navigable(console)


def _stand(console: SpotWebConsole) -> None:
    if console.agent.posture() == "standing":  # type: ignore[union-attr]
        logger.info("Already standing — nothing to do.")
        console.stance = STANCE_STAND
        return
    console._stand_up()


def _sit(console: SpotWebConsole) -> None:
    """Sit and power off — or, if Spot is already down, say so and do nothing at all.

    It used to stand up first to have power for stowing the arm, so sitting a sitting robot
    got it up and put it back down again.
    """
    posture = console.agent.posture()  # type: ignore[union-attr]
    if posture in {"off", "sitting"} or (posture == "unknown" and console.stance == STANCE_SIT):
        logger.info(f"Already sitting ({'motors off' if posture == 'off' else 'motors on'}) — nothing to do.")
        console.stance = STANCE_SIT
        return
    mission_lib.run_sit_down(console.agent)
    console.stance = STANCE_SIT


def _ready(console: SpotWebConsole) -> None:
    """How every action starts: standing, the hand out in front with the gripper open.

    Also how one recovers — after a failed grasp, or a Stop that froze the arm wherever it
    was, the next action brings the hand back to this pose before it does anything else.
    """
    console._stand_up()
    console.agent.arm_ready()  # type: ignore[union-attr]


def _go_home(console: SpotWebConsole) -> bool:
    _ready(console)
    if not _navigable(console):
        return False
    return mission_lib.run_go_to(console.agent, console.selection["home"])


def _patrol(console: SpotWebConsole) -> bool:
    _ready(console)
    if not _navigable(console):
        return False
    return mission_lib.run_patrol_loop(console.agent, list(console.selection.get("patrol") or []))


def _navigate_to(console: SpotWebConsole, waypoint: str) -> bool:
    _ready(console)
    if not _navigable(console):
        return False
    return mission_lib.run_go_to(console.agent, waypoint)


def _sort(console: SpotWebConsole, confirm: bool) -> bool:
    _ready(console)
    # The mission's scans are drawn over the cameras whether or not AI is on, and the
    # preview stands aside so it never delays the grasp servo's hand scans.
    console.view.mission_scans = True
    if not _navigable(console):
        return False
    points = mission_lib.SortPoints(
        pick=console.selection["pick"],
        bins={group: console.selection[f"bin_{group}"] for group in GROUPS},
    )
    return mission_lib.run_sort_loop(console.agent, console.vision, points, console._sort_hooks(confirm))


def _grasp_here(console: SpotWebConsole, confirm: bool) -> bool:
    """One grasp near where Spot stands — no map, no navigation — then set it down."""
    _ready(console)
    console.view.mission_scans = True
    return mission_lib.run_grasp_here(console.agent, console.vision, console._sort_hooks(confirm))


def _navigable(console: SpotWebConsole) -> bool:
    if console.agent.ensure_navigation():  # type: ignore[union-attr]
        return True
    logger.warning("GraphNav unavailable — park Spot in front of a fiducial and retry.")
    return False


_HANDLERS = {
    "navigate_to": _navigate_to,
    "load_map": _load_map,
    "status_check": _status_check,
    "initialisation": _initialisation,
    "stand": _stand,
    "sit": _sit,
    "go_home": _go_home,
    "patrol": _patrol,
    "sort_confirm": lambda console: _sort(console, confirm=True),
    "sort_auto": lambda console: _sort(console, confirm=False),
    "grab_confirm": lambda console: _grasp_here(console, confirm=True),
    "grab_auto": lambda console: _grasp_here(console, confirm=False),
}


# ── helpers ───────────────────────────────────────────────────────────────────


def _recon_label(track: object) -> dict:
    """A registered object as the 3D Recon view labels it."""
    x, y, z = (round(float(v), 3) for v in track.position)  # type: ignore[attr-defined]
    group = track.group  # type: ignore[attr-defined]
    return {"id": track.id, "label": track.label, "group": group,  # type: ignore[attr-defined]
            "color": GROUPS[group]["hex"], "conf": round(track.conf, 2),  # type: ignore[attr-defined]
            "uncertain": track.uncertain, "x": x, "y": y, "z": z}  # type: ignore[attr-defined]


def _spec(mission_id: str | None) -> dict:
    for group in MENU:
        for item in group["items"]:
            if item["id"] == mission_id:
                return item
    return _UNLISTED.get(mission_id or "", {})


def _load_selection() -> dict:
    """The saved points, falling back to `.env` for the home point."""
    try:
        saved = json.loads(_SELECTION_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        saved = {}
    chosen = {key: str(saved.get(key) or "") for key in _SELECTION_KEYS}
    chosen["home"] = chosen["home"] or os.getenv("GO_HOME_WAYPOINT", "")
    chosen["patrol"] = _clean_route(saved.get("patrol"))
    return chosen


def _clean_route(route: object) -> list[str]:
    if not isinstance(route, list):
        return []
    return [str(name).strip() for name in route if str(name).strip()]


def _save_selection(selection: dict) -> None:
    try:
        _SELECTION_FILE.parent.mkdir(parents=True, exist_ok=True)
        _SELECTION_FILE.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning(f"Could not save the waypoint selection: {exc}")


def _display_host(host: str) -> str:
    return "localhost" if host in {"0.0.0.0", "::"} else host


__all__ = ["MENU", "SpotWebConsole"]
