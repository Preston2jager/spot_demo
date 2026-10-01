"""Put the web console onto Spot's CoreIO as the `demo_web` image.

Runs *here*, on the laptop. The sibling of `coreio/deploy.py`, which does the same job
for the YOLO service, and it reuses that script's SSH plumbing rather than growing its
own — same `COREIO_*` credentials, same reasoning about where things may be installed.

    python coreio/deploy_web.py                  # upload, build on the robot, start
    python coreio/deploy_web.py --ship-image     # build here, ship the image, start
    python coreio/deploy_web.py --no-build       # upload only
    python coreio/deploy_web.py --no-start       # upload and build, start it yourself

Everything that goes up lives in `coreio/demo_web/`, plus one `app.tar.gz` packed here
from the manifest below. The image carries the code, the GraphNav map and the speaker
audio — and no weights: detection belongs to the `spot_yolo` service, which owns the GPU
and the TensorRT engine.

**The CoreIO has no route to the internet, so the build borrows this laptop's.**

`coreio/proxy_tunnel.py` opens a reverse SSH forward for the duration of the build: the
robot's `127.0.0.1:3128` comes out here, and apt, pip and wget inside the build use it.
Nothing has to be installed on the robot and nothing is left running afterwards. The
image is built `FROM ubuntu:22.04`, which is already on the CoreIO — a base that is *not*
cached would have to be pulled, and a pull is the docker daemon's own request, which
ignores the build's proxy and would need dockerd reconfigured and restarted.

`--ship-image` is the alternative for a laptop that cannot reach the robot's network and
the internet at the same time: the image is built here for linux/arm64, saved, and loaded
on the robot. It needs `docker buildx` with qemu binfmt on this machine.
"""
import argparse
import contextlib
import os
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from proxy_tunnel import ReverseProxy

# Imported by module name, not as `coreio.remote`: this script is run as
# `python coreio/deploy_web.py`, so its own directory is what lands on sys.path.
from remote import connect, remote_home, run, upload

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
# Everything uploaded lives in this one folder, generated files included: app.tar.gz and
# demo_web.env are written *into* it rather than beside it, so "what goes to the robot"
# is a directory listing and the docker build context is that same directory.
_CONTEXT_DIR = _HERE / "demo_web"
_BUILD_DIR = _CONTEXT_DIR / "build"
_REMOTE_SUBDIR = "demo_web"
_IMAGE = "demo_web:latest"

# What the container needs to be the application. Deliberately a list rather than "the
# repo minus some ignores": Archive/ is dead experiments, src/ref/ is a vendor copy
# nothing imports, and runtime/weights/ is the one thing that must *not* travel.
_PAYLOAD = [
    "src",
    "start.py",
    "runtime/maps",
    "assets",
]
_PAYLOAD_EXCLUDE_DIRS = {"__pycache__", ".ruff_cache", ".pytest_cache"}
# Verbatim Boston Dynamics CLI examples. Nothing in `src/` imports them, and they pull in
# a GUI stack the container has no use for.
_PAYLOAD_EXCLUDE_PATHS = ("src/ref",)

# Copied into the container's env file from the laptop's .env when present, so a single
# deploy carries the site's waypoints and speaker settings across.
_INHERITED = [
    "COREIO_USER", "COREIO_PASSWORD",
    "COREIO_AUDIO_DEVICE", "COREIO_SPEAKER_VOLUME", "RESET_ANNOUNCE_AUDIO",
    "GO_HOME_WAYPOINT", "GRIP_TORQUE_SOFT", "GRIP_TORQUE_MEDIUM", "GRIP_TORQUE_HARD",
    "COREIO_VISION_PORT", "COREIO_VISION_TIMEOUT", "WEB_CAPTURE_FPS", "WEB_SCAN_INTERVAL",
    "COREIO_RS_ENABLED", "COREIO_RS_PORT", "COREIO_RS_TIMEOUT",
]


def pack_app() -> Path:
    """Tar the application into the upload folder, so one directory is all that ships."""
    archive = _CONTEXT_DIR / "app.tar.gz"

    def keep(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        if info.name.startswith(_PAYLOAD_EXCLUDE_PATHS):
            return None
        return None if set(Path(info.name).parts) & _PAYLOAD_EXCLUDE_DIRS else info

    print(f"[web] packing {', '.join(_PAYLOAD)}")
    with tarfile.open(archive, "w:gz") as tar:
        for entry in _PAYLOAD:
            source = _PROJECT_ROOT / entry
            if not source.exists():
                sys.exit(f"[web] missing from the payload: {source}")
            tar.add(source, arcname=entry, filter=keep)
    print(f"[web] app.tar.gz is {archive.stat().st_size >> 20} MB")
    return archive


def container_env(args: argparse.Namespace) -> str:
    """Build the container's env file from the template plus this machine's settings.

    The credentials are read here rather than mounted: `credentials.json` is not in the
    image, and the robot's address is different from inside the CoreIO anyway.
    """
    from dotenv import dotenv_values

    local = dotenv_values(_PROJECT_ROOT / ".env")
    template = dotenv_values(_CONTEXT_DIR / "demo_web.env.example")

    settings = {key: (value or "") for key, value in template.items()}
    settings.update({key: local[key] for key in _INHERITED if local.get(key)})

    creds = {}
    credentials_file = _PROJECT_ROOT / "credentials.json"
    if credentials_file.is_file():
        import json

        creds = json.loads(credentials_file.read_text())

    settings["SPOT_HOSTNAME"] = args.robot_host
    settings["SPOT_USERNAME"] = local.get("SPOT_USERNAME") or creds.get("username", "")
    settings["SPOT_PASSWORD"] = local.get("SPOT_PASSWORD") or creds.get("password", "")
    settings["WEB_PORT"] = str(args.port)
    settings["WEB_CONTROL_PORT"] = str(args.port + 1)
    # Inside the CoreIO the speaker's SSH host is the CoreIO itself, whatever .env says
    # about reaching it from the laptop — and so is its port. 20022 is Spot's forward of
    # the CoreIO's sshd, published on the *robot*; from inside the CoreIO nothing listens
    # there and only :22 does. Both are overridden rather than inherited, because a .env
    # written from the laptop's point of view is right for the laptop and wrong here.
    settings["COREIO_HOST"] = "127.0.0.1"
    settings["COREIO_PORT"] = "22"

    if not settings["SPOT_USERNAME"] or not settings["SPOT_PASSWORD"]:
        print("[web] warning: no Spot credentials found — fill them into demo_web.env on the robot")

    lines = ["# Written by coreio/deploy_web.py — edit on the robot or redeploy.\n"]
    lines += [f"{key}={value}\n" for key, value in settings.items()]
    return "".join(lines)


def ship_image(client: object, remote_app: str) -> None:
    """Build the image here for arm64 and load it on the robot.

    The escape hatch for a CoreIO with no working proxy. `buildx` with qemu is what makes
    an aarch64 image buildable on an x86 laptop; without the binfmt handlers registered
    the build fails on the first `RUN`, which is worth saying plainly rather than letting
    it look like a broken Dockerfile.
    """
    _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    tarball = _BUILD_DIR / "demo_web.tar"
    print("[web] building linux/arm64 here — this is slow under emulation")
    build = subprocess.run(
        [
            "docker", "buildx", "build", "--platform", "linux/arm64",
            "-t", _IMAGE, "-o", f"type=docker,dest={tarball}", str(_CONTEXT_DIR),
        ],
        check=False,
    )
    if build.returncode != 0:
        sys.exit(
            "[web] the arm64 build failed. Check `docker buildx ls` shows a builder, and "
            "that qemu is registered:\n"
            "      docker run --privileged --rm tonistiigi/binfmt --install arm64"
        )

    size = tarball.stat().st_size
    print(f"[web] uploading the image ({size >> 20} MB) — this takes a while over wifi")
    sftp = client.open_sftp()
    try:
        started = time.time()
        sftp.put(str(tarball), f"{remote_app}/demo_web.tar")
        print(f"[web] uploaded in {time.time() - started:.0f}s")
    finally:
        sftp.close()

    print("[web] loading the image on the robot...")
    print(run(client, f"docker load -i {shlex.quote(remote_app)}/demo_web.tar").strip())
    run(client, f"rm -f {shlex.quote(remote_app)}/demo_web.tar", check=False)


def build_on_robot(client: object, remote_app: str, args: argparse.Namespace) -> None:
    """Build the image on the CoreIO, lending it this laptop's internet while it runs.

    The tunnel is opened around the build and taken down after: the robot has no route of
    its own, and leaving it one it did not ask for is not this script's business.
    """
    proxy_url = f"http://127.0.0.1:{args.proxy_port}"
    context = (
        contextlib.nullcontext() if args.no_tunnel
        else ReverseProxy(client, port=args.proxy_port)
    )
    with context:
        print("[web] building on the robot — apt, pip and wget go through this laptop")
        _, stdout, stderr = client.exec_command(
            f"APP_DIR={shlex.quote(remote_app)} PROXY={shlex.quote(proxy_url)} "
            f"bash {shlex.quote(remote_app)}/build_image.sh",
            timeout=7200,
        )
        # Streamed rather than collected: a twenty-minute build that prints nothing is
        # indistinguishable from a hang.
        for line in iter(stdout.readline, ""):
            if line.strip():
                print(f"[web]   {line.rstrip()}")
        if stdout.channel.recv_exit_status() != 0:
            sys.exit(
                "[web] the build failed:\n"
                + stderr.read().decode(errors="replace").strip()
                + "\n      If this laptop has no internet of its own, the tunnel has nothing\n"
                  "      to forward — check `ip route show default` here."
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", type=int, default=int(os.getenv("WEB_PORT", "8080")),
                        help="port the console listens on, on the CoreIO (default: 8080)")
    parser.add_argument("--robot-host", default="192.168.50.3",
                        help="Spot's API address as seen from the CoreIO (default: 192.168.50.3)")
    parser.add_argument("--proxy-port", type=int, default=3128,
                        help="port on the CoreIO the build-time proxy is tunnelled to (default: 3128)")
    parser.add_argument("--no-tunnel", action="store_true",
                        help="do not open the reverse proxy; assume something already serves --proxy-port")
    parser.add_argument("--remote-dir", help="install path on the CoreIO (default: ~/demo_web)")
    parser.add_argument("--ship-image", action="store_true",
                        help="build the image here for arm64 and load it there, instead of building on the robot")
    parser.add_argument("--no-build", action="store_true", help="upload only; do not build an image")
    parser.add_argument("--no-start", action="store_true", help="do not start the container afterwards")
    args = parser.parse_args()

    archive = pack_app()
    env_file = _CONTEXT_DIR / "demo_web.env"
    env_file.write_text(container_env(args), encoding="utf-8")

    client = connect()
    try:
        remote_app = args.remote_dir or f"{remote_home(client)}/{_REMOTE_SUBDIR}"
        print(f"[web] installing into {remote_app}")
        run(client, f"mkdir -p {shlex.quote(remote_app)}/logs")

        sftp = client.open_sftp()
        try:
            for name in ("Dockerfile", ".dockerignore", "requirements.txt", "entrypoint.sh",
                         "build_image.sh", "run_demo_web.sh"):
                upload(sftp, _CONTEXT_DIR / name, f"{remote_app}/{name}")
            upload(sftp, archive, f"{remote_app}/app.tar.gz")
            upload(sftp, env_file, f"{remote_app}/demo_web.env")
            # Credentials live in this file; it should not be world-readable on a robot
            # several people share.
            sftp.chmod(f"{remote_app}/demo_web.env", 0o600)
        finally:
            sftp.close()

        if args.no_build:
            print(f"[web] uploaded. Build it with: bash {remote_app}/build_image.sh")
            return

        if args.ship_image:
            ship_image(client, remote_app)
        else:
            build_on_robot(client, remote_app, args)

        if args.no_start:
            print(f"[web] built. Start it with: bash {remote_app}/run_demo_web.sh")
            return

        print(run(client, f"APP_DIR={shlex.quote(remote_app)} bash {remote_app}/run_demo_web.sh").strip())
    finally:
        client.close()

    host = os.getenv("COREIO_HOST", "192.168.80.3")
    port = os.getenv("COREIO_PORT", "20022")
    user = os.getenv("COREIO_USER", "spot")
    print(
        f"\n[web] the console is on the CoreIO: display :{args.port}, control :{args.port + 1}.\n"
        f"      Spot forwards only 20022 and 21443 off the CoreIO, so from a laptop on\n"
        f"      Spot's wifi both are reached through the SSH session you already have:\n\n"
        f"        ssh -N -L {args.port}:127.0.0.1:{args.port} -L {args.port + 1}:127.0.0.1:{args.port + 1} "
        f"-p {port} {user}@{host}\n"
        f"        open http://localhost:{args.port} (display) and http://localhost:{args.port + 1} (control)\n\n"
        f"      On a network with a route to the CoreIO itself, http://<coreio>:{args.port}\n"
        f"      works directly — the container runs on the host network.\n"
        f"      Watch it:  ssh -p {port} {user}@{host} docker logs -f demo_web\n"
        f"      Stop it:   ssh -p {port} {user}@{host} docker stop demo_web"
    )


if __name__ == "__main__":
    main()
