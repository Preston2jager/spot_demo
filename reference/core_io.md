# Notes on 16th Sep 2026 by Sang

## Wired access to the Core IO (there is no DHCP on the payload port)
- The Core IO is powered independently of Spot — it boots and runs with the robot switched off.
  What does *not* work with the robot off is the usual `-p 20022 spot@192.168.80.3` route, because
  that port is forwarded by Spot's own computer.
- Plugging a laptop straight into the Core IO gives link but no address: its `eth0` is a static
  `192.168.50.5/24` with **no DHCP server and no gateway**, so NetworkManager sits in
  `dhcp4 … no lease` forever. Give the laptop a static address on the same subnet instead.
  ```bash
  nmcli connection modify "Wired connection 2" \
    ipv4.method manual \
    ipv4.addresses 192.168.50.100/24 \
    ipv4.gateway "" \
    ipv4.never-default yes \
    ipv4.route-metric 200 \
    ipv6.method link-local

  nmcli connection up "Wired connection 2"
  ```
  - `never-default` matters: without it the wired link steals the default route and the laptop
    loses internet, since the Core IO has no gateway to offer.
  - To undo it: `nmcli connection modify "Wired connection 2" ipv4.method auto ipv4.addresses ""`
- Added both routes to `~/.ssh/config` so they are one word each:
  ```bash
  # Wired straight into the Core IO — works with Spot powered off.
  Host spot_coreio
      HostName 192.168.50.5
      User spot
      Port 22

  # Through Spot's wifi AP — needs Spot powered on.
  Host spot_coreio_wifi
      HostName 192.168.80.3
      User spot
      Port 20022

  Host spot_coreio spot_coreio_wifi
      IdentityFile ~/.ssh/id_ed25519
      IdentitiesOnly yes
      StrictHostKeyChecking accept-new
      ServerAliveInterval 30
      ServerAliveCountMax 3
  ```

## Container management moved to Dockge
- The four project containers no longer start at boot. From now on the only non-BD container that
  comes up on its own is **Dockge**, and everything else is started and stopped from its web UI.
  Boston Dynamics' own containers (`coreio_web_server`, `coreio_nginx`, `coreio_influx`) are
  untouched and still managed by `bd-docker-containers.service`.
- Dockge is at **http://192.168.50.5:5001** over the wire. Over wifi, Spot forwards only 20022 and
  21443, so tunnel it:
  ```bash
  ssh -L 5001:127.0.0.1:5001 spot_coreio_wifi
  # then open http://localhost:5001
  ```
  - Use the following to log in
    ```bash
    Username: spot
    Password: bim1191
    ```
  - This account lives in `/data/dockge-data/dockge.db` on the host, not in the container, so it
    survives `docker-compose down && up` and any rebuild of the image. Deleting that directory is
    how you reset a forgotten password: Dockge goes back to asking for a new admin account on the
    next visit.
- The Core IO cannot pull an image (no DNS, no default route), so the arm64 image was shipped in
  from the laptop. Do the same for any future image:
  ```bash
  docker pull --platform linux/arm64 louislam/dockge:1
  docker save --platform linux/arm64 louislam/dockge:1 -o /tmp/dockge-arm64.tar
  gzip -1 -c /tmp/dockge-arm64.tar | ssh spot_coreio 'gunzip -c | docker load'
  ```
- Stacks live in `/data/stacks`, Dockge's own database in `/data/dockge-data`.
  - `DOCKGE_STACKS_DIR` **must be the same path inside and outside the container** — Dockge shells
    out to `docker compose` against the host daemon, so the paths it passes have to mean the same
    thing on both sides. Upstream's default `/opt/stacks` is impossible here: the Core IO's `/` is
    a read-only image (`df` shows it 100% full, which is by design, not a fault).
  ```
  /data/stacks/
  ├── dockge/           restart: unless-stopped   <- the only one that starts at boot
  ├── spot-yolo/        restart: "no"
  ├── spot-ros2/        restart: "no"
  ├── spot-web/         restart: "no"   <- was not running before; now one click away
  ├── realsense-ros2/   restart: "no"
  └── spot-navigation/  restart: "no"
  ```
- Three of these were already Compose services, so their `compose.yaml` is a **verbatim copy** of the
  file that was running them. The originals are left in place and untouched at
  `/data/ros2_files/dockerfiles/`. `spot_yolo` was a plain `docker run`, and was translated from
  `coreio/service/run_yolo_server.sh`.
- Every edit made while copying, and nothing else:
  - `restart: unless-stopped` → `restart: "no"` on all four, so a reboot leaves them down.
  - The `build:` sections were dropped from `realsense-ros2` and `spot-navigation`. Those builds
    need the `127.0.0.1:3128` proxy from the reverse-proxy tunnel below, the images are already
    built on this machine, and leaving `build:` in only invites a failed rebuild on some later
    click. **Rebuild from the original files, not from the stacks.**
  - `spot-navigation`: `./nav2.yaml` made absolute, since the stack no longer sits beside it.
  - `spot-navigation`: the log line `ROS_DOMAIN_ID=${ROS_DOMAIN_ID}` was being substituted by
    Compose at *parse* time from the host environment, which is why it always printed empty. It is
    `$${ROS_DOMAIN_ID}` now so the container's own shell expands it. Cosmetic — the environment
    variable itself was always correct.
  - `spot-web`: `depends_on: [spot_ros2]` removed, because Compose `depends_on` cannot reach into
    another stack. **Start the `spot-ros2` stack first if `spot-web` needs the driver.**
  - Verified by creating each container from its stack with `docker-compose create` and diffing
    `docker inspect` against the containers that had been running: image, command, network mode,
    privileged, nvidia runtime, GPU reservations, mounts, environment, tty and workdir all match
    byte for byte. Only the restart policy and that one echo line differ.
- Per-stack notes are written as comments at the top of each `compose.yaml`, which is what Dockge's
  editor shows. Worth reading before starting `spot-ros2`: that driver **takes Spot's lease** when it
  navigates, and Spot recognises exactly one holder, so running it alongside the web console means
  whichever client spoke last owns the robot. Leave the stack stopped when ROS 2 is not in use.
- To put a stack back to starting at boot, change its `restart` in Dockge and redeploy:
  ```bash
  # in the stack's compose.yaml
  restart: unless-stopped
  ```

## Careful: `deploy.py` will undo the Dockge setup for spot_yolo
- `coreio/deploy.py` finishes by running `coreio/service/run_yolo_server.sh`, which does its own
  `docker run -d --name spot_yolo --restart unless-stopped …`. Running it again therefore recreates
  the container **outside Dockge and with boot auto-start back on**.
- Until that script is changed, after any `python coreio/deploy.py` do:
  ```bash
  ssh spot_coreio 'docker rm -f spot_yolo; cd /data/stacks/spot-yolo && docker-compose up -d'
  ```
- Note the stack serves the TensorRT engine (`/weights/yolov8m.engine`), which is what was loaded
  before. An engine is built against this exact GPU, driver and TensorRT version — never copy one in
  from elsewhere. Rebuild it on the robot with `python coreio/deploy.py --tensorrt`.

## Things found while looking, but not changed
- `velodyne_service` and `spotcore_lidar_ip_config` fail and restart forever (restart counter was
  past 20). Both are looking for a Velodyne VLP-16, which this robot does not have. Harmless, but
  they fill the journal. To silence them:
  ```bash
  sudo systemctl disable --now velodyne_service spotcore_lidar_ip_config
  ```
- `coreio-feature-servicer` had restarted 121 times with `UnableToConnectToRobotError`. It is
  `Restart=always` and dials `192.168.50.3`, so it does this whenever Spot is powered off. Expected;
  it settles as soon as the robot comes up.
- `systemd-remount-fs` fails on every boot with `cannot remount /dev/root read-write`. That is the
  read-only root image again, not a fault.
- Docker is holding 30.6 GB of images, 25.0 GB of it reclaimable. `/data` has 329 GB free so there is
  no hurry, but `docker image prune -a` is the first thing to run if space ever gets tight.
- Boot takes 4 min 42 s, of which `bd-docker-containers.service` is 1 min 52 s. Normal — BD gives
  that unit a ten-minute timeout.

# Notes on 28th Aug 2026 by Sang
I have disabled entity_detection.service on CoreIO, its taking too much resources on the dog.
To enbale this service in the futre use:
```bash
sudo systemctl enable entity_detection.service
sudo systemctl start entity_detection.service
```

# Getting internet connection to Core IO (Reverse Proxy tunneling method)
## Connecting to the Core IO
- Use the following command to ssh into the Core IO
  ```bash
  ssh -o IdentitiesOnly=yes -p 20022 spot@192.168.80.3
  ```
  - If connecting through VS Code, the `config` file should contain the following.
    ```bash
    Host spot-dog
      HostName 192.168.80.3
      User spot
      Port 20022
      IdentitiesOnly yes
    ```
  - Use the following as the password
    ```bash
    3gydvg6mnegsgzr57s
    ```
