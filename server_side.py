from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import psutil
import time
from collections import deque
import threading

app = Flask(__name__)
model = YOLO("yolo11n.pt")

# --- rolling stores for the dashboard (last N samples) ---
edge_metrics_history = deque(maxlen=300)    # pushed from the edge
cloud_metrics_history = deque(maxlen=300)   # sampled locally
lock = threading.Lock()

# --- background thread: sample cloud resources once a second ---
def sample_cloud_metrics():
    proc = psutil.Process()
    while True:
        with lock:
            cloud_metrics_history.append({
                "ts": time.time(),
                "cpu_percent": psutil.cpu_percent(),
                "cpu_per_core": psutil.cpu_percent(percpu=True),
                "mem_percent": psutil.virtual_memory().percent,
                "mem_used_mb": round(psutil.virtual_memory().used / 1e6, 1),
                "net_sent_mb": round(psutil.net_io_counters().bytes_sent / 1e6, 1),
                "net_recv_mb": round(psutil.net_io_counters().bytes_recv / 1e6, 1),
                "proc_cpu": proc.cpu_percent(),
                "proc_mem_mb": round(proc.memory_info().rss / 1e6, 1),
                "load_avg": psutil.getloadavg()[0],
            })
        time.sleep(1)

threading.Thread(target=sample_cloud_metrics, daemon=True).start()


@app.route("/detect", methods=["POST"])
def detect():
    t0 = time.time()
    jpg_bytes = request.data
    arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    results = model.track(frame, persist=True, conf=0.3, verbose=False,
                          classes=[2, 3, 5, 7])
    inference_ms = (time.time() - t0) * 1000

    boxes = results[0].boxes
    dets = []
    if boxes.id is not None:
        for tid, cls, conf in zip(
            boxes.id.int().tolist(),
            boxes.cls.int().tolist(),
            boxes.conf.tolist(),
        ):
            dets.append({"id": tid, "label": results[0].names[cls],
                         "conf": round(conf, 2)})

    return jsonify({
        "detections": dets,
        "inference_ms": round(inference_ms, 1),   # cloud-side compute time for THIS frame
    })


@app.route("/metrics", methods=["POST"])
def receive_metrics():
    """Edge pushes its metrics here."""
    data = request.get_json()
    with lock:
        edge_metrics_history.append(data)
    return jsonify({"status": "ok"})


@app.route("/metrics/data")
def metrics_data():
    """Dashboard reads combined history from here."""
    with lock:
        return jsonify({
            "edge": list(edge_metrics_history),
            "cloud": list(cloud_metrics_history),
        })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)