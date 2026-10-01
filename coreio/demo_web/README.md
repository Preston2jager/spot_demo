# `demo_web` — the console, running on Spot

Everything in this folder goes to the robot. `coreio/deploy_web.py` runs on the laptop,
packs `src/`, `start.py`, the GraphNav map and the speaker audio into `app.tar.gz`
alongside these files, uploads the lot to `~/demo_web/`, and builds the image there.

```
coreio/demo_web/
  Dockerfile           the image
  .dockerignore        the upload folder is the build context; this trims it to the image's inputs
  requirements.txt     only what the console imports — no ultralytics, no OpenVINO
  entrypoint.sh        prints what it is about to connect to, then runs start.py
  build_image.sh       runs ON the robot: docker build, through the proxy
  run_demo_web.sh      runs ON the robot: docker run, on demand
  demo_web.env.example the committed template for the container's environment
  app.tar.gz           generated: the application payload            (git-ignored)
  demo_web.env         generated: the real environment, with credentials (git-ignored, uploaded 0600)
  build/               generated: the saved image, for --ship-image  (git-ignored)
```

## Deploying

```bash
python coreio/deploy_web.py
```

That packs, uploads, builds on the robot, and starts it. The container is **not** left
running: no `--restart` policy and `--rm`, so it exists for as long as a demo does.

```bash
ssh -p 20022 spot@192.168.80.3 bash '~/demo_web/run_demo_web.sh'   # start
ssh -p 20022 spot@192.168.80.3 docker stop demo_web                # stop
ssh -p 20022 spot@192.168.80.3 docker logs -f demo_web             # watch
```

## Getting an image onto a machine with no internet

The CoreIO has no DNS and no default route, so there are exactly two ways.

**Build on the robot** (the default) needs the squid tunnel from `core_io_setup.md`
running: that is what apt and pip talk to. It also needs the docker *daemon* proxied,
because pulling the `python:3.12-slim` base is the daemon's request rather than the
build's — if `docker pull python:3.12-slim` fails on the robot but
`curl -x http://127.0.0.1:3128 https://pypi.org` works, that is the missing piece.

**Ship the image** asks nothing of the robot's network:

```bash
python coreio/deploy_web.py --ship-image
```

It builds `linux/arm64` on the laptop under emulation, saves the image, uploads it, and
`docker load`s it there. Slow — emulated builds and a few hundred MB over Spot's wifi —
but it always works. Needs buildx with qemu registered:

```bash
docker run --privileged --rm tonistiigi/binfmt --install arm64
```

## Reaching the console

The container runs on the **host network**, so it binds the CoreIO's own ports 8080 and 8081.

Spot forwards only **20022** and **21443** off the CoreIO — a listener on any other port
is unreachable from a laptop on Spot's wifi. So from there, ride the SSH session:

```bash
ssh -N -L 8080:127.0.0.1:8080 -L 8081:127.0.0.1:8081 -p 20022 spot@192.168.80.3
# then open http://localhost:8080 (display) and http://localhost:8081 (control)
```

On a network with a route to the CoreIO itself, `http://<coreio>:8080` works directly.

## What the container talks to

| | Address from inside the CoreIO | Why |
|---|---|---|
| Spot's API | `192.168.50.3` (`SPOT_HOSTNAME`) | the payload network; `192.168.80.3` is the *laptop's* view and reaches nothing from here |
| Vision service | `127.0.0.1:8784`, no tunnel (`COREIO_VISION_DIRECT=1`) | `spot_vision` is a container on the same host network |
| Speaker | `127.0.0.1:22` over SSH | playback needs `aplay` on the CoreIO host, which this container cannot call directly |

## No weights in this image

Vision is the `spot_vision` service's job — it owns the GPU and the models, and is deployed
separately with `python coreio/deploy_vision.py`. This image carries no weights and no
ultralytics; if the vision service is down, the console's Vision pill says so.

The container serves both pages: the display on `WEB_PORT` (8080) and the control page on
`WEB_CONTROL_PORT` (8081). Forward both through the SSH session.
