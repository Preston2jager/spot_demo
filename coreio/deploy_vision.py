"""Package the vision service as a Docker image on Spot's CoreIO and install it in Dockge.

Runs *here*, on the laptop, after `coreio/vision/bake_vocab.py` has written the weights:

    python coreio/deploy_vision.py               # upload, build spot-vision:latest, start the stack
    python coreio/deploy_vision.py --no-start    # build and install; start it from Dockge
    python coreio/deploy_vision.py --no-build    # only rewrite the stack file (e.g. new thresholds)

What happens on the robot:

1. `service/` and the three weight files go to `~/vision_service/` — under the account's
   home because /data is root-owned, and /home is a symlink onto the same volume.
2. `docker build -t spot-vision:latest` there. `FROM yolo:latest` and COPY only, because
   the CoreIO cannot reach a registry or a package index.
3. The stack is written to `/data/stacks/spot-vision/compose.yaml`, which is Dockge's stacks
   directory — so it appears in Dockge next to every other container on this machine and
   is started, stopped and watched from there.
4. `docker-compose up -d --force-recreate` in that directory, unless `--no-start`. The old
   `spot-yolo` stack is stopped first: both want most of the Xavier's GPU memory.
"""
import argparse
import shlex
import sys
from pathlib import Path

from remote import connect, remote_home, run, stream, upload

_HERE = Path(__file__).parent
_SERVICE = _HERE / "vision" / "service"
_WEIGHTS = _HERE.parent / "runtime" / "weights" / "vision"
_REMOTE_SUBDIR = "vision_service"
_STACKS_DIR = "/data/stacks"
_STACK = "spot-vision"
_IMAGE = "spot-vision:latest"
# The host has the standalone compose v2 binary, not the `docker compose` plugin.
_COMPOSE = "docker-compose"
# Stacks from earlier vision services (the YOLO-World `coreio-vision` project). They use the
# same container name, spot_vision, and bind-mount ~/vision_service — the directory this
# script uploads into — so one left running serves this code on its own port (8782, which
# is the RealSense bridge's) and this stack cannot start. Moved out of Dockge, not deleted.
_RETIRED_STACKS = ("coreio-vision",)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=_WEIGHTS,
                        help="directory holding segment.pt, classify.pt and vocab.json")
    parser.add_argument("--no-build", action="store_true", help="skip upload and image build")
    parser.add_argument("--no-start", action="store_true", help="install the stack but leave it stopped")
    args = parser.parse_args()

    needed = [args.weights / name for name in ("segment.pt", "classify.pt", "vocab.json")]
    missing = [str(path) for path in needed if not path.is_file()]
    if missing and not args.no_build:
        sys.exit("[deploy] missing " + ", ".join(missing) + "\n         run: coreio/vision/bake_vocab.py (see its docstring)")

    client = connect()
    try:
        if not run(client, "docker image inspect yolo:latest >/dev/null 2>&1 && echo yes", check=False).strip():
            sys.exit("[deploy] yolo:latest is not on this CoreIO — it is the base image, and cannot be pulled here")
        remote_app = f"{remote_home(client)}/{_REMOTE_SUBDIR}"
        stack_dir = f"{_STACKS_DIR}/{_STACK}"

        if not args.no_build:
            run(client, f"mkdir -p {shlex.quote(remote_app)}/weights")
            sftp = client.open_sftp()
            try:
                for name in ("vision_server.py", "Dockerfile"):
                    upload(sftp, _SERVICE / name, f"{remote_app}/{name}")
                for path in needed:
                    upload(sftp, path, f"{remote_app}/weights/{path.name}", skip_if_same_size=path.suffix == ".pt")
            finally:
                sftp.close()
            print(f"[deploy] building {_IMAGE} on the robot (COPY only, ~1 min)")
            stream(client, f"cd {shlex.quote(remote_app)} && docker build -t {_IMAGE} .", "docker build")

        if not run(client, f"test -d {_STACKS_DIR} && echo yes", check=False).strip():
            sys.exit(f"[deploy] {_STACKS_DIR} is not there — is Dockge installed on this CoreIO?")
        run(client, f"mkdir -p {stack_dir}")
        sftp = client.open_sftp()
        try:
            upload(sftp, _SERVICE / "compose.yaml", f"{stack_dir}/compose.yaml")
        finally:
            sftp.close()

        if args.no_start:
            print(f"[deploy] installed. Start '{_STACK}' in Dockge, or: cd {stack_dir} && {_COMPOSE} up -d")
            return

        # Stopped, not removed: it stays in Dockge, one click from coming back.
        run(client, "docker stop spot_yolo >/dev/null 2>&1 || true", check=False)
        _retire_old_stacks(client)
        print("[deploy] starting the stack")
        stream(client, f"cd {stack_dir} && {_COMPOSE} up -d --force-recreate", "docker-compose up")
    finally:
        client.close()

    print(
        "\n[deploy] the service loads two models and warms CUDA before answering (~30 s).\n"
        "         Dockge:   ssh -L 5001:127.0.0.1:5001 spot_coreio_wifi, then http://localhost:5001\n"
        "         logs:     docker logs -f spot_vision  (on the CoreIO) — wait for '[vision] ready'\n"
        "         console:  the Detector pill turns green once /health answers"
    )


def _retire_old_stacks(client) -> None:
    """Take down superseded vision stacks and move them to ~/retired_stacks."""
    for stack in _RETIRED_STACKS:
        path = f"{_STACKS_DIR}/{stack}"
        if not run(client, f"test -f {path}/compose.yaml && echo yes", check=False).strip():
            continue
        print(f"[deploy] retiring the old '{stack}' stack -> ~/retired_stacks/{stack}")
        run(client, f"cd {path} && {_COMPOSE} down", check=False)
        run(client, f"mkdir -p ~/retired_stacks && mv {path} ~/retired_stacks/{stack}", check=False)


if __name__ == "__main__":
    main()
