[Go to Home](/readMe.md)
# Setting up the docker container in CoreIO
- Follow the instruction for [reverse proxy tunneling](../../Documentation/reverse_proxy_tunneling.md)

- Start a minimal interactive container on Core I/O
    ```bash
    sudo docker run -it --name spot_manual_build \
        --network=host \
        -e HTTP_PROXY=http://127.0.0.1:3128 \
        -e HTTPS_PROXY=http://127.0.0.1:3128 \
        -e http_proxy=http://127.0.0.1:3128 \
        -e https_proxy=http://127.0.0.1:3128 \
        ubuntu:22.04 bash
    ```
- Set apt proxy inside the container
  ```bash
  cat > /etc/apt/apt.conf.d/95proxy <<EOF
  Acquire::http::Proxy "http://127.0.0.1:3128";
  Acquire::https::Proxy "http://127.0.0.1:3128";
  EOF
  ```
  - Test everything is OK
    ```bash
    apt-get update
    ```

- Install base tools
  ```bash
  apt-get install -y \
    locales \
    curl \
    git \
    wget \
    ca-certificates \
    gnupg \
    lsb-release \
    python3-pip \
    software-properties-common
  ```
  ```bash
  locale-gen en_US.UTF-8
  update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
  export LANG=en_US.UTF-8
  export LC_ALL=en_US.UTF-8
  ```

- Set pip proxy
  ```bash
  python3 -m pip config set global.proxy http://127.0.0.1:3128
  ```

- Add ROS 2 Humble repository
  ```bash
  curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg

  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" \
    > /etc/apt/sources.list.d/ros2.list

  apt-get update
  ```

- Install ROS and required dependencies
  ```bash
  apt-get install -y --no-install-recommends \
    python-is-python3 \
    python3-argcomplete \
    python3-colcon-common-extensions \
    python3-colcon-mixin \
    python3-rosdep \
    libpython3-dev \
    python3-tk \
    qttools5-dev \
    ros-humble-ros-base \
    ros-dev-tools \
    ros-humble-controller-manager \
    ros-humble-xacro \
    ros-humble-hardware-interface \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-launch-pytest \
    ros-humble-controller-interface \
    ros-humble-forward-command-controller \
    ros-humble-bondpy \
    ros-humble-bondcpp \
    ros-humble-cv-bridge \
    ros-humble-depth-image-proc \
    ros-humble-image-transport \
    ros-humble-tl-expected \
    ros-humble-joint-state-publisher \
    ros-humble-joint-state-publisher-gui \
    ros-humble-example-interfaces \
    ros-humble-rviz2 \
    python3-protobuf \
    python3-pytest-cov \
    python3-typing-extensions \
    libopencv-dev \
    libprotobuf-dev \
    libprotoc-dev \
    protobuf-compiler \
    libgrpc++-dev \
    libgrpc++1 \
    libprotobuf23 \
    libcli11-dev
  ```
  - Verify CLI11 exists
    ```bash
    dpkg -l | grep cli11
    find /usr -name "*CLI11*Config.cmake" -o -name "*cli11*config.cmake"
    ```
    - You should see something like `/usr/lib/aarch64-linux-gnu/cmake/CLI11/CLI11Config.cmake`

- Initialise rosdep
  ```bash
  rosdep init || true
  rosdep update
  ```

- Clone spot_ros2
  ```bash
  mkdir -p /ros_ws/src
  cd /ros_ws/src

  git clone https://github.com/rai-opensource/spot_ros2.git .
  git submodule update --init --recursive
  ```

- Run the Spot installer manually
  - Test GitHub access
    ```bash
    wget -O /tmp/test.html https://github.com
    ```
  - Start the installation
    ```bash
    ARCH=$(dpkg --print-architecture)
    echo "Building driver with $ARCH"

    /ros_ws/src/install_spot_ros2.sh --$ARCH
    ```

- Commit the changes
  ```bash
  sudo docker commit spot_manual_build spot-ros2:v1
  ```
  - Test the new image
    ```bash
    sudo docker run -it --rm --network=host spot-ros2:v1 bash
    ```
  - Run following inside the container
    ```bash
    source /opt/ros/humble/setup.bash
    source /ros_ws/install/setup.bash
    ros2 pkg list | grep spot
    ```

- Build the workspace
  ```bash
  cd /ros_ws

  source /opt/ros/humble/setup.bash
  colcon build --symlink-install
  ```