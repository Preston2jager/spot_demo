#!/usr/bin/env bash
# Build the demo_web image on the CoreIO. Run this ON the robot.
#
# The CoreIO has no DNS and no default route, so this needs a proxy at $PROXY that can
# actually reach the internet. `python coreio/deploy_web.py` opens one for the duration
# of the build — a reverse SSH tunnel out through the laptop — so the ordinary way to
# run this script is to let the deploy script do it.
#
# --network host is not optional: without it the RUN steps get their own loopback and
# 127.0.0.1:3128 is nothing at all.
#
# There is no curl on this machine; wget is what it has.
set -euo pipefail

APP_DIR=${APP_DIR:-$HOME/demo_web}
IMAGE=${IMAGE:-demo_web:latest}
PROXY=${PROXY:-http://127.0.0.1:3128}

echo "[build] proxy $PROXY"
if ! http_proxy="$PROXY" https_proxy="$PROXY" \
     wget -q --spider --timeout=15 --tries=1 https://pypi.org/simple/; then
  echo "[build] no internet through $PROXY — start the tunnel first:" >&2
  echo "        python coreio/deploy_web.py     (it opens one for the build)" >&2
  exit 1
fi
echo "[build] proxy reaches pypi"

docker build \
  --network host \
  --build-arg HTTP_PROXY="$PROXY" \
  --build-arg HTTPS_PROXY="$PROXY" \
  -t "$IMAGE" \
  "$APP_DIR"

echo "[build] $IMAGE ready — start it with: bash $APP_DIR/run_demo_web.sh"
