# spot-vision — the segment+classify service, packaged as its own image.
#
# Built ON the CoreIO by coreio/deploy_vision.py. The base is the `yolo:latest` image that
# is already there (ultralytics 8.3.156, torch 2.0 for JetPack 5, CUDA 11.4), and the
# build does nothing but COPY: the CoreIO has no DNS and no default route, so any step that
# reached for a registry or a package index would fail.
#
# The weights are baked into the image rather than mounted, so what Dockge starts is one
# self-contained thing and a stack can never pair the service with somebody else's weights.
FROM yolo:latest

ENV YOLO_CONFIG_DIR=/tmp \
    PYTHONUNBUFFERED=1 \
    VISION_WEIGHTS_DIR=/weights

COPY weights/segment.pt weights/classify.pt weights/vocab.json /weights/
COPY vision_server.py /app/vision_server.py

WORKDIR /app
CMD ["python3", "/app/vision_server.py"]
