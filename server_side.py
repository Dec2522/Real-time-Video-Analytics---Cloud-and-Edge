from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import time
import psutil

app = Flask(__name__)
model = YOLO("yolo11n.pt")   # loaded once at startup, not per request

@app.route("/detect", methods=["POST"])
def detect():
    t0 = time.time()
    # Receive raw JPEG bytes from the edge
    jpg_bytes = request.data
    # Decode back into an image
    arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR) # is colour needed
    # Run inference
    results = model.track(frame, persist=True, conf=0.3, verbose=False)
    # Pull out detections into plain JSON-able data
    boxes = results[0].boxes
    dets = []
    inference_ms = (time.time() - t0) * 1000
    if boxes.id is not None:
        for tid, cls, conf in zip(
            #boxes.xyxy.tolist(),
            boxes.id.int().tolist(),
            boxes.cls.int().tolist(),
            boxes.conf.tolist(),
        ):
            dets.append({
                #"box": box,            # [x1, y1, x2, y2]
                "id": tid,
                "label": results[0].names[cls],
                "conf": round(conf, 2),
            })


        return jsonify({
        "detections": dets,
        "metrics": {
            "inference_ms": round(inference_ms, 1),
            "cpu_percent": psutil.cpu_percent(),
            "mem_percent": psutil.virtual_memory().percent,
            "mem_used_mb": round(psutil.virtual_memory().used / 1024 / 1024, 1),
        }
    })

if __name__ == "__main__":
    # localhost for local testing; change host to the Nebula IP on the VM "127.0.0.1"
    app.run(host="10.0.0.2", port=8000)