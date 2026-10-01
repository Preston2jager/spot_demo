#!/usr/bin/env bash
# Start the console, having said out loud what it is about to connect to.
#
# The four lines below are the ones that go wrong: the robot address is different from
# inside the CoreIO than from the laptop, and the detector is a localhost port here
# rather than an SSH tunnel. Printing them costs nothing and turns "it does not work"
# into one glance at `docker logs demo_web`.
set -euo pipefail

echo "[demo_web] robot        : ${SPOT_HOSTNAME:-<from credentials.json>}"
echo "[demo_web] display port : ${WEB_PORT:-8080}   control port : ${WEB_CONTROL_PORT:-8081}"
echo "[demo_web] vision       : ${COREIO_VISION_HOST:-127.0.0.1}:${COREIO_VISION_PORT:-8784}" \
     "(direct=${COREIO_VISION_DIRECT:-0})"
echo "[demo_web] map          : ${SPOT_MAP_DIR:-./runtime/maps/amp_3}"

exec python /app/start.py
