# The demo_web image: the Spot web console, built to run on Spot's CoreIO.
#
# Python 3.12 because `src/` uses `typing.Self` and PEP 604 annotations that are
# evaluated at import time — 3.10 cannot even load the modules. The Jetson's own
# `yolo:latest` image is Python 3.8, which is exactly why the console gets an image of
# its own rather than sharing that one.
#
# Nothing about CUDA appears here. Detection is not this container's job: it asks the
# `spot_yolo` service next door, which owns the GPU and the TensorRT engine. That is the
# brief's "only the TensorRT weights are hosted by the CoreIO" — this image carries the
# code and the map, and no weights at all.
#
# `ubuntu:22.04` rather than `python:3.12-slim`, for one blunt reason: it is already on
# the robot. A `FROM` that is not cached locally has to be pulled, and a pull is the
# docker *daemon's* request — it ignores the build's proxy settings and would need
# dockerd reconfigured and restarted, which on this machine means bouncing Boston
# Dynamics' own extension containers. So the base is one that needs no pull, and Python
# arrives as a standalone build over the same proxy apt and pip use.
FROM ubuntu:22.04

# Build-time only. The CoreIO has no DNS and no default route, so a build there reaches
# an index solely through the reverse tunnel `coreio/proxy_tunnel.py` opens — which is
# why this must be built with `--network host`, or 127.0.0.1 is the container's own
# loopback and the proxy is invisible. ARGs rather than ENV, so the proxy is never baked
# into the runtime environment as a dangling reference to a tunnel that is long gone.
ARG HTTP_PROXY=""
ARG HTTPS_PROXY=""
ENV http_proxy=$HTTP_PROXY \
    https_proxy=$HTTPS_PROXY \
    HTTP_PROXY=$HTTP_PROXY \
    HTTPS_PROXY=$HTTPS_PROXY \
    DEBIAN_FRONTEND=noninteractive

# ffmpeg is for the speaker: CoreIO ships only `aplay`, so announcements are transcoded
# to PCM WAV on this side before being pushed over SFTP. libgl/libglib are what the
# OpenCV wheel links against even headless.
RUN if [ -n "$HTTP_PROXY" ]; then \
        printf 'Acquire::http::Proxy "%s";\nAcquire::https::Proxy "%s";\n' \
            "$HTTP_PROXY" "${HTTPS_PROXY:-$HTTP_PROXY}" > /etc/apt/apt.conf.d/95proxy; \
    fi \
 && apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates wget ffmpeg libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# A relocatable CPython from python-build-standalone — the same builds `uv python install`
# uses. Pinned by URL: an image that silently changes its interpreter between rebuilds is
# not an image anyone can debug.
ARG PYTHON_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20260825/cpython-3.12.14%2B20260825-aarch64-unknown-linux-gnu-install_only_stripped.tar.gz"
RUN wget -q -O /tmp/python.tar.gz "$PYTHON_URL" \
 && tar -xzf /tmp/python.tar.gz -C /opt \
 && rm /tmp/python.tar.gz \
 && ln -sf /opt/python/bin/python3 /usr/local/bin/python \
 && ln -sf /opt/python/bin/python3 /usr/local/bin/python3 \
 && python -VV

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
 && python -m pip install --no-cache-dir -r /tmp/requirements.txt \
 && rm /tmp/requirements.txt

# The application, packed by coreio/deploy_web.py: src/, start.py, the GraphNav map and
# the speaker audio. No weights, no Archive/, no src/ref/.
COPY app.tar.gz /tmp/app.tar.gz
RUN mkdir -p /app && tar -xzf /tmp/app.tar.gz -C /app && rm /tmp/app.tar.gz

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Drop the build proxy and the apt override: at run time there is no tunnel and nothing
# to fetch, and a stale proxy setting would only make failures confusing.
RUN rm -f /etc/apt/apt.conf.d/95proxy
ENV http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    WEB_PORT=8080 \
    WEB_CONTROL_PORT=8081

WORKDIR /app
EXPOSE 8080 8081
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
