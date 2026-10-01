import json
import os
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from bosdyn.api.graph_nav import map_pb2
from loguru import logger


class Janitor:
    """Static utility class for all file I/O: credentials, graph protos, and temp audio files."""

    @staticmethod
    def load_credentials(path: str | Path | None = None) -> dict:
        """Read robot hostname, username, and password from a JSON credentials file."""
        if path is None:
            path = Path(__file__).parent.parent / "credentials.json"
        try:
            with open(path) as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Credentials file not found at '{path}'. "
                "Create credentials.json with hostname, username, and password keys."
            )

    @staticmethod
    def load_graph(save_dir: str) -> map_pb2.Graph:
        """Parse and return a GraphNav graph protobuf from a local map directory."""
        graph_path = os.path.join(save_dir, "graph")
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Graph file not found: {graph_path}")
        with open(graph_path, "rb") as f:
            graph = map_pb2.Graph()
            graph.ParseFromString(f.read())
        return graph

    @staticmethod
    def _snapshot_path(save_dir: str, subdir: str, prefix: str, snapshot_id: str) -> str | None:
        """Find one snapshot file across both map layouts, or None if it is not there.

        `download_graph` writes ``<map>/waypoint_snapshots/<id>`` and
        ``<map>/edge_snapshots/<id>``; some older exports here are flat, with the id
        prefixed instead. Both are checked because a map that silently uploads without
        its snapshots still localises — and then navigates badly, which is a much harder
        failure to read than a missing file.
        """
        for candidate in (
            os.path.join(save_dir, subdir, snapshot_id),
            os.path.join(save_dir, f"{prefix}{snapshot_id}"),
        ):
            if os.path.exists(candidate):
                return candidate
        return None

    @staticmethod
    def load_waypoint_snapshots(save_dir: str, graph: map_pb2.Graph) -> list:
        """Load and return all waypoint snapshot protos referenced by the given graph."""
        snapshots = []
        for wp in graph.waypoints:
            if not wp.snapshot_id:
                continue
            path = Janitor._snapshot_path(save_dir, "waypoint_snapshots", "wp_", wp.snapshot_id)
            if path is None:
                logger.warning(f"Missing waypoint snapshot: {wp.snapshot_id}")
                continue
            with open(path, "rb") as f:
                snap = map_pb2.WaypointSnapshot()
                snap.ParseFromString(f.read())
                for img in snap.images:
                    img.shot.image.data = b""
            snapshots.append(snap)
        return snapshots

    @staticmethod
    def load_edge_snapshots(save_dir: str, graph: map_pb2.Graph) -> list:
        """Load and return all edge snapshot protos referenced by the given graph."""
        snapshots = []
        for edge in graph.edges:
            if not edge.snapshot_id:
                continue
            path = Janitor._snapshot_path(save_dir, "edge_snapshots", "edge_", edge.snapshot_id)
            if path is None:
                logger.warning(f"Missing edge snapshot: {edge.snapshot_id}")
                continue
            with open(path, "rb") as f:
                snap = map_pb2.EdgeSnapshot()
                snap.ParseFromString(f.read())
            snapshots.append(snap)
        return snapshots

    @staticmethod
    @contextmanager
    def temp_audio(suffix: str = ".oga") -> Generator[str, None, None]:
        """Yield a temporary file path for audio and delete it on context exit."""
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        path = tmp.name
        tmp.close()
        try:
            yield path
        finally:
            if os.path.exists(path):
                os.unlink(path)
