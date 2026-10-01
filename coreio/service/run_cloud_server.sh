#!/usr/bin/env bash
# Start the live point-cloud stitcher on Spot's CoreIO.
#
# Run this ON the CoreIO (ssh -p 20022 spot@192.168.80.3). It rides in the same
# yolo:latest image the detector uses — torch, CUDA and numpy are already in there, and
# the CoreIO cannot pip anything — but as its own container, with its own name, logs and
# restart policy, so restarting one service never takes the other down.
#
# Binds CoreIO-localhost only, like every other service here: Spot forwards just 20022
# and 21443, so the console reaches it through an SSH channel and a port of its own would
# be unreachable anyway.
set -euo pipefail

# Under $HOME, not /data: /data is root-owned, while /home is a symlink onto the same
# encrypted volume, so this is persistent and needs no privilege.
APP_DIR=${APP_DIR:-$HOME/cloud_service}
IMAGE=${CLOUD_IMAGE:-yolo:latest}
PORT=${CLOUD_PORT:-8783}
DEVICE=${CLOUD_DEVICE:-cuda}

docker rm -f spot_cloud >/dev/null 2>&1 || true

docker run -d --name spot_cloud \
  --restart unless-stopped \
  --runtime nvidia \
  --network host \
  -e CLOUD_HOST=127.0.0.1 \
  -e CLOUD_PORT="${PORT}" \
  -e CLOUD_DEVICE="${DEVICE}" \
  ${CLOUD_VOXEL_M:+-e CLOUD_VOXEL_M="${CLOUD_VOXEL_M}"} \
  ${CLOUD_BUDGET:+-e CLOUD_BUDGET="${CLOUD_BUDGET}"} \
  -v "${APP_DIR}":/app:ro \
  "${IMAGE}" \
  python3 /app/cloud_server.py

echo "spot_cloud started on 127.0.0.1:${PORT} — follow it with: docker logs -f spot_cloud"
