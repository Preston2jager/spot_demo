"""Push the live point-cloud stitcher onto Spot's CoreIO.

Runs *here*, on the laptop. Like the RealSense bridge it ships one Python file and no
weights: the stitcher runs in the `yolo:latest` image that is already on the robot, which
has torch and CUDA in it and is the only way to get either — the CoreIO has no DNS and no
default route, so it cannot pull an image or pip a wheel.

    python coreio/deploy_cloud.py                 # upload and (re)start
    python coreio/deploy_cloud.py --voxel 0.04    # finer grid, more points
    python coreio/deploy_cloud.py --device cpu    # bench it on the ARM cores
    python coreio/deploy_cloud.py --no-start      # upload only

It runs as its own container (`spot_cloud`) beside `spot_yolo` rather than inside it: one
image, two services, so restarting the stitcher never interrupts a detection mid-fetch.
"""
import argparse
import shlex
import sys
from pathlib import Path

# Imported by module name, not as `coreio.remote`: this script is run as
# `python coreio/deploy_cloud.py`, so its own directory is what lands on sys.path.
from remote import connect, remote_home, run, upload

_SERVICE_DIR = Path(__file__).parent / "service"
_REMOTE_SUBDIR = "cloud_service"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", default="8783", help="port on the CoreIO's localhost (default: 8783)")
    parser.add_argument("--image", default="yolo:latest",
                        help="image to run in — it needs torch with CUDA (default: yolo:latest)")
    parser.add_argument("--device", default="cuda",
                        help="cuda or cpu; cuda falls back on its own if unavailable")
    parser.add_argument("--voxel", default="", help="voxel edge in metres (default: 0.06)")
    parser.add_argument("--budget", default="", help="how many voxels to keep (default: 600000)")
    parser.add_argument("--remote-dir", help="install path on the CoreIO (default: ~/cloud_service)")
    parser.add_argument("--no-start", action="store_true", help="upload only; do not (re)start the container")
    args = parser.parse_args()

    client = connect()
    try:
        remote_app = args.remote_dir or f"{remote_home(client)}/{_REMOTE_SUBDIR}"
        print(f"[cloud] installing into {remote_app}")
        run(client, f"mkdir -p {shlex.quote(remote_app)}")

        sftp = client.open_sftp()
        try:
            upload(sftp, _SERVICE_DIR / "cloud_server.py", f"{remote_app}/cloud_server.py")
            upload(sftp, _SERVICE_DIR / "run_cloud_server.sh", f"{remote_app}/run_cloud_server.sh")
        finally:
            sftp.close()

        if args.no_start:
            print(f"[cloud] uploaded; start it with: bash {remote_app}/run_cloud_server.sh")
            return

        if not run(client, f"docker image inspect {shlex.quote(args.image)} >/dev/null 2>&1 && echo yes",
                   check=False).strip():
            sys.exit(
                f"[cloud] no {args.image} image on the robot, and the CoreIO cannot pull one.\n"
                f"        yolo:latest is part of the CoreIO's own setup (spot-vision is built on it);\n"
                f"        point --image at another image that has torch with CUDA."
            )

        print("[cloud] (re)starting the spot_cloud container...")
        env = (
            f"APP_DIR={shlex.quote(remote_app)} "
            f"CLOUD_IMAGE={shlex.quote(args.image)} "
            f"CLOUD_PORT={shlex.quote(args.port)} "
            f"CLOUD_DEVICE={shlex.quote(args.device)}"
        )
        if args.voxel:
            env += f" CLOUD_VOXEL_M={shlex.quote(args.voxel)}"
        if args.budget:
            env += f" CLOUD_BUDGET={shlex.quote(args.budget)}"
        print(run(client, f"{env} bash {remote_app}/run_cloud_server.sh").strip())
    finally:
        client.close()

    print(
        "\n[cloud] the stitcher answers immediately; it has nothing to load.\n"
        "        watch it:  ssh -p 20022 spot@192.168.80.3 docker logs -f spot_cloud\n"
        "        check it:  python -c \"from dotenv import load_dotenv; load_dotenv(); \\\n"
        "                   from src.spot_cloud import LiveCloud; c=LiveCloud(); \\\n"
        "                   c.connect(); print(c.health()); c.close()\"\n"
        "        then open the console's Live tab — it starts sweeping when you do."
    )


if __name__ == "__main__":
    main()
