from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)
model = YOLO("yolo11n.pt")   # loaded once at startup, not per request

@app.route("/detect", methods=["POST"])
def detect():
    # 1. Receive raw JPEG bytes from the edge
    jpg_bytes = request.data
    # 2. Decode back into an image
    arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR) # is colour needed
    # 3. Run inference
    results = model.track(frame, persist=True, conf=0.3, verbose=False)
    # 4. Pull out detections into plain JSON-able data
    boxes = results[0].boxes
    dets = []
    if boxes.id is not None:
        for box, tid, cls, conf in zip(
            boxes.xyxy.tolist(),
            boxes.id.int().tolist(),
            boxes.cls.int().tolist(),
            boxes.conf.tolist(),
        ):
            dets.append({
                "box": box,            # [x1, y1, x2, y2]
                "id": tid,
                "label": results[0].names[cls],
                "conf": round(conf, 2),
            })
    return jsonify({"detections": dets})

if __name__ == "__main__":
    # localhost for local testing; change host to the Nebula IP on the VM
    app.run(host="127.0.0.1", port=8000)