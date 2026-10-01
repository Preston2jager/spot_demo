"""Push the RealSense bridge onto Spot's CoreIO as a Dockge stack.

Runs *here*, on the laptop, and is the smallest of the deploy scripts because it ships no
weights and builds no image: the `realsense-coreio` image is already on the robot with
ROS 2, cv_bridge and OpenCV in it, so all that travels is one Python file and a compose.

    python coreio/deploy_realsense.py                # install the stack and (re)start it
    python coreio/deploy_realsense.py --no-start     # install only; start it from Dockge

1. `realsense_server.py` goes to `/data/stacks/spot-realsense/app/`, and
   `service/realsense_compose.yaml` becomes that stack's `compose.yaml`. Display range and
   rotation live in its environment: edit them in Dockge, not here.
2. A `spot_realsense` container that is not the stack's (the old `docker run` one) is
   removed, since it holds the name.
3. `docker-compose up -d --force-recreate` in the stack, unless `--no-start`.

The bridge reads `realsense2_camera`'s topics rather than the camera — see
`coreio/service/realsense_server.py` for why — so the `realsense-ros2` stack has to be
running for it to have anything to serve. That is checked here, because "no frame yet"
from the service and "nobody is publishing" look identical from the console.
"""
import argparse
import sys
from pathlib import Path

# Imported by module name, not as `coreio.remote`: this script is run as
# `python coreio/deploy_realsense.py`, so its own directory is what lands on sys.path.
from remote import connect, run, stream, upload

_SERVICE_DIR = Path(__file__).parent / "service"
_STACKS_DIR = "/data/stacks"
_STACK = "spot-realsense"
_CONTAINER = "spot_realsense"
_CAMERA_CONTAINER = "realsense_ros2"
_IMAGE = "realsense-coreio:v1"
# The host has the standalone compose v2 binary, not the `docker compose` plugin.
_COMPOSE = "docker-compose"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--no-start", action="store_true", help="install the stack but leave it stopped")
    args = parser.parse_args()

    client = connect()
    try:
        if not run(client, f"test -d {_STACKS_DIR} && echo yes", check=False).strip():
            sys.exit(f"[realsense] {_STACKS_DIR} is not there — is Dockge installed on this CoreIO?")
        if not run(client, f"docker image inspect {_IMAGE} >/dev/null 2>&1 && echo yes", check=False).strip():
            sys.exit(f"[realsense] no {_IMAGE} image on the robot, and the CoreIO cannot pull one.")

        stack_dir = f"{_STACKS_DIR}/{_STACK}"
        print(f"[realsense] installing the stack into {stack_dir}")
        run(client, f"mkdir -p {stack_dir}/app")
        sftp = client.open_sftp()
        try:
            upload(sftp, _SERVICE_DIR / "realsense_server.py", f"{stack_dir}/app/realsense_server.py")
            upload(sftp, _SERVICE_DIR / "realsense_compose.yaml", f"{stack_dir}/compose.yaml")
        finally:
            sftp.close()

        if args.no_start:
            print(f"[realsense] installed. Start '{_STACK}' in Dockge, or: cd {stack_dir} && {_COMPOSE} up -d")
            return

        # Said before starting rather than after failing: a bridge with no publisher comes
        # up perfectly happily and then reports "waiting" for as long as anyone watches it.
        running = run(
            client,
            f"docker ps --filter name=^{_CAMERA_CONTAINER}$ --format '{{{{.Names}}}}'",
            check=False,
        ).strip()
        if not running:
            print(
                f"[realsense] warning: {_CAMERA_CONTAINER} is not running, so nothing is publishing\n"
                f"            the camera topics. Start the realsense-ros2 stack in Dockge."
            )

        # "none" when the container exists without compose labels (the old `docker run` one),
        # empty when there is no container at all.
        project = run(
            client,
            f"docker inspect -f '{{{{or (index .Config.Labels \"com.docker.compose.project\") \"none\"}}}}' "
            f"{_CONTAINER} 2>/dev/null",
            check=False,
        ).strip()
        if project and project != _STACK:
            print(f"[realsense] removing the {_CONTAINER} container that is not the stack's")
            run(client, f"docker rm -f {_CONTAINER}", check=False)

        print("[realsense] starting the stack")
        stream(client, f"cd {stack_dir} && {_COMPOSE} up -d --force-recreate", "docker-compose up")
    finally:
        client.close()

    print(
        "\n[realsense] the bridge answers as soon as the first frame arrives.\n"
        "            Dockge:   ssh -L 5001:127.0.0.1:5001 spot_coreio_wifi, then http://localhost:5001\n"
        f"            logs:     docker logs -f {_CONTAINER}  (on the CoreIO)\n"
        "            health:   the console's RealSense pill, or wget -qO- 127.0.0.1:8782/health there"
    )


if __name__ == "__main__":
    main()
