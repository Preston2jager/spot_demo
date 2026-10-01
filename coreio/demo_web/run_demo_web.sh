#!/usr/bin/env bash
# Start the demo_web console on the CoreIO. Run this ON the robot.
#
# The image is not meant to be running all the time — no `--restart` policy, and `--rm`
# so a stopped container leaves nothing behind. Start it for a demo, stop it after:
#
#   bash ~/demo_web/run_demo_web.sh          # start, detached
#   bash ~/demo_web/run_demo_web.sh -f       # start in the foreground, logs on screen
#   docker stop demo_web                     # stop
#   docker logs -f demo_web                  # watch it
#
# --network host, for three reasons at once: the console has to reach Spot's API on the
# payload network, the YOLO service listens on CoreIO-localhost, and the console's own
# port then needs no publishing to be reachable.
set -euo pipefail

APP_DIR=${APP_DIR:-$HOME/demo_web}
IMAGE=${IMAGE:-demo_web:latest}
ENV_FILE=${ENV_FILE:-$APP_DIR/demo_web.env}

# A `[ ... ] || [ ... ] && x` chain would exit the script under `set -e` when neither
# test matches, which is the common case.
FOREGROUND=0
case "${1:-}" in
  -f|--foreground) FOREGROUND=1 ;;
esac

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "no $IMAGE image here — build it first: bash $APP_DIR/build_image.sh" >&2
  exit 1
fi

docker rm -f demo_web >/dev/null 2>&1 || true

# Logs are bind-mounted out so a run survives the container being thrown away.
mkdir -p "$APP_DIR/logs"

COMMON=(
  --name demo_web
  --rm
  --network host
  --env-file "$ENV_FILE"
  -v "$APP_DIR/logs":/app/runtime/logs
  "$IMAGE"
)

if [ "$FOREGROUND" = "1" ]; then
  exec docker run -it "${COMMON[@]}"
fi

docker run -d "${COMMON[@]}"
PORT=$(grep -E '^WEB_PORT=' "$ENV_FILE" | cut -d= -f2 || true)
CONTROL=$(grep -E '^WEB_CONTROL_PORT=' "$ENV_FILE" | cut -d= -f2 || true)
echo "demo_web started — display :${PORT:-8080}, control :${CONTROL:-8081} — docker logs -f demo_web"
