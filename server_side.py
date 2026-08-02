import argparse
import os

from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2 
import numpy as np
import psutil
import time
from collections import deque, defaultdict
import threading

app = Flask(__name__)

MODEL_WEIGHTS = "yolo11n.pt"
HISTORY_LEN = 300          # samples retained per series for the dashboard
STREAM_TIMEOUT_S = 15      # no traffic for this long => stream counted as finished

# Inference backends. `pytorch` is the unoptimised baseline, the other two are
# the CPU optimisation, kept switchable so they can be compared
BACKENDS = ("pytorch", "openvino", "onnx")
DEFAULT_IMGSZ = 640


# Change response metadata headers to allow any website to fetch it
@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp

# Rolling storage of edge metrics, keyed by steam ID
edge_metrics_history = defaultdict(lambda: deque(maxlen=HISTORY_LEN)) 
# Cloud metrics history, shared across streams
cloud_metrics_history = deque(maxlen=HISTORY_LEN)
stream_info = {}            
# Avoid concurrent writes the shared data storage
lock = threading.Lock()     

# --- --- Detector --- ---
# A single detector is shared across all streams.
# The model is thread-safe, but per-stream tracking of object IDs is not.
# To stop track IDs from different videos blending, the per-stream tracker state is swapped in and out around each call.
# Why? Treat resource as a pool rather than a dedication per stream. And a model per stream is exspensive.
#
# Inference is deliberately serialised under model_lock rather than run concurrently:
# on CPU the runtime already parallelises each convolution across all cores, so
# letting N streams infer at once oversubscribes the cores instead of scaling.
# Concurrency is exploited either side of this lock (decode, encode, transport).
model = None                      # assigned in main() once the backend is known
model_lock = threading.Lock()
stream_trackers = {}              # stream_id - saved predictor.trackers list

# Inference resolution. Passed explicitly on every call so that the exported
# backends (which bake in a fixed input shape) and the pytorch baseline are
# compared at exactly the same resolution.
imgsz = DEFAULT_IMGSZ

# Describes the active backend, surfaced to the dashboard so a run can be
# attributed to the runtime that produced it.
runtime_info = {"backend": "pytorch", "weights": MODEL_WEIGHTS,
                "imgsz": DEFAULT_IMGSZ, "int8": False, "torch_threads": None}

inflight = 0  # number of requests being processed
inflight_lock = threading.Lock()


def export_path(weights, backend):
    """Where ultralytics writes the exported artefact for this backend."""
    stem = os.path.splitext(weights)[0]
    return f"{stem}_openvino_model" if backend == "openvino" else f"{stem}.onnx"


def load_model(weights, backend, size, int8=False, data=None):
    """Load `weights` under the chosen backend, exporting on first use.

    The export is cached on disk, so only the first run of a given
    (weights, backend, imgsz) combination pays the conversion cost.
    """
    if backend == "pytorch":
        return YOLO(weights)

    target = export_path(weights, backend)
    if not os.path.exists(target):
        print(f"[server] exporting {weights} -> {backend} (imgsz={size}, int8={int8}); first run only")
        kwargs = {"format": backend, "imgsz": size}
        if backend == "onnx":
            # Runtime rejected the default newer version of opset, so have forced an older one.
            kwargs["opset"] = 20
        if backend == "openvino" and int8:
            # INT8 needs a calibration set; ultralytics falls back to a small
            # default COCO subset (downloaded on demand) when data is omitted.
            kwargs["int8"] = True
            if data:
                kwargs["data"] = data
        YOLO(weights).export(**kwargs)

    print(f"[server] loading {target}")
    return YOLO(target)


def _swap_in_trackers(stream_id):
    """Point the shared predictor at this stream's tracker state. Called with model_lock held.

    Relies on model having attributes predictor, tracker, reset. 
    YOLO model used does, can't guarentee for others
    """
    predictor = getattr(model, "predictor", None)
    # Avoid first call failure of 
    if predictor is None:
        return None
    saved = stream_trackers.get(stream_id)
    if saved is not None:
        predictor.trackers = saved
    else:
        # If first frame of a stream, reset the trackers of the left over state fom previous stream
        for t in predictor.trackers:
            t.reset()
    return predictor


def _swap_out_trackers(stream_id, predictor):
    """Stash this stream's tracker state back. Called with model_lock held."""
    if predictor is not None:
        stream_trackers[stream_id] = predictor.trackers


def active_streams(now=None):
    """Show what streams are still active."""
    now = now or time.time()
    return [sid for sid, info in stream_info.items()
            if now - info["last_seen"] <= STREAM_TIMEOUT_S]


# --- background thread: sample cloud resources once a second ---
def sample_cloud_metrics():
    proc = psutil.Process() # the current process
    while True:
        with lock:
            # add cloud metrics
            cloud_metrics_history.append({
                "ts": time.time(),
                "cpu_percent": psutil.cpu_percent(),
                "cpu_per_core": psutil.cpu_percent(percpu=True),
                "mem_percent": psutil.virtual_memory().percent,
                "mem_used_mb": round(psutil.virtual_memory().used / 1e6, 1), 
                "net_sent_mb": round(psutil.net_io_counters().bytes_sent / 1e6, 1), 
                "net_recv_mb": round(psutil.net_io_counters().bytes_recv / 1e6, 1),
                # Process i.e. the codes cost
                "proc_cpu": proc.cpu_percent(), # Sums across cores
                "proc_mem_mb": round(proc.memory_info().rss / 1e6, 1),
                # Queue - the avg number of processes wanting CPU
                "load_avg": psutil.getloadavg()[0],
                # Concurrency
                "inflight_requests": inflight,
                "active_streams": len(active_streams()),
            })
        time.sleep(1) # sample metrics every second

# Video detection end point
@app.route("/detect", methods=["POST"])
def detect():
    global inflight
    stream_id = request.args.get("stream", "0")

    t0 = time.time()
    jpg_bytes = request.data # Get data
    arr = np.frombuffer(jpg_bytes, dtype=np.uint8) # convert to numpy array
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR) # decode into coloured image
    decode_ms = (time.time() - t0) * 1000 # log decode latency

    with inflight_lock:
        inflight += 1
    try:
        t1 = time.time()
        # One model, one lock: streams queue here rather than inferring in parallel.
        with model_lock:
            predictor = _swap_in_trackers(stream_id)
            # run YOLO with track for persistent object IDs across frames
            results = model.track(frame, persist=True, conf=0.3, verbose=False,
                                  imgsz=imgsz,          # identical across backends
                                  classes=[2, 3, 5, 7]) # Classes = vehicles we want to detect
            # predictor only exists after the first call - re-fetch on frame 1
            _swap_out_trackers(stream_id, predictor or getattr(model, "predictor", None))
        inference_ms = (time.time() - t1) * 1000 # log inference latency
    finally:
        with inflight_lock:
            inflight -= 1

    boxes = results[0].boxes
    dets = []
    # If there are detections:
    # One line per object - ID: Label: Confidence:
    if boxes.id is not None:
        for tid, cls, conf in zip(
            boxes.id.int().tolist(),
            boxes.cls.int().tolist(),
            boxes.conf.tolist(),
        ):
            dets.append({"id": tid, "label": results[0].names[cls],
                         "conf": round(conf, 2)})

    # Count labels
    labels = [d["label"] for d in dets]
    counts = {l: labels.count(l) for l in set(labels)}

    with lock:
        info = stream_info.setdefault(stream_id, {"frames": 0, "video": None, "last_seen": 0})
        info["frames"] += 1
        info["last_seen"] = time.time()

    return jsonify({
        "stream_id": stream_id,
        "detections": dets,
        "counts": counts,                          # per-label counts for THIS frame
        "decode_ms": round(decode_ms, 1),          # cloud-side JPEG decode time
        "inference_ms": round(inference_ms, 1),    # cloud-side compute time for THIS frame
        "backend": runtime_info["backend"],        # so the edge CSV records which runtime served it
    })


@app.route("/metrics", methods=["POST"])
def receive_metrics():
    """Edge pushes its metrics here."""
    data = request.get_json()
    stream_id = str(data.get("stream_id", "0"))
    data["stream_id"] = stream_id
    with lock:
        edge_metrics_history[stream_id].append(data)
        info = stream_info.setdefault(stream_id, {"frames": 0, "video": None, "last_seen": 0})
        info["last_seen"] = time.time()
        if data.get("video"):
            info["video"] = data["video"]
    return jsonify({"status": "ok"})


@app.route("/metrics/data")
def metrics_data():
    """Dashboard reads combined history from here.

    `edge` is keyed by stream id - each stream pushes on its own cadence, so the
    dashboard draws them as independent series rather than one merged timeline.
    """
    with lock:
        now = time.time()
        live = set(active_streams(now))
        return jsonify({
            "edge": {sid: list(recs) for sid, recs in edge_metrics_history.items()},
            "cloud": list(cloud_metrics_history),
            "runtime": runtime_info,
            "streams": [
                {
                    "id": sid,
                    "video": info["video"],
                    "frames": info["frames"],
                    "last_seen": info["last_seen"],
                    "active": sid in live,
                }
                for sid, info in sorted(stream_info.items())
            ],
        })


def main():
    global model, imgsz

    parser = argparse.ArgumentParser(description="Cloud inference service.")
    parser.add_argument("--backend", choices=BACKENDS, default="pytorch",
                        help="inference runtime: pytorch = unoptimised baseline, "
                             "openvino/onnx = CPU-optimised (exported on first use)")
    parser.add_argument("--weights", default=MODEL_WEIGHTS,
                        help="model weights, e.g. yolo11n.pt / yolo11s.pt / yolo11m.pt")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                        help="inference resolution, applied to every backend so the "
                             "comparison is like for like")
    parser.add_argument("--int8", action="store_true",
                        help="openvino only: INT8 quantisation (needs a calibration set)")
    parser.add_argument("--data", default=None,
                        help="dataset yaml used to calibrate --int8")
    parser.add_argument("--threads", type=int, default=None,
                        help="cap the runtime's CPU threads; pin this when comparing "
                             "backends or sweeping stream counts")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.int8 and args.backend != "openvino":
        parser.error("--int8 applies to --backend openvino")

    if args.threads is not None:
        if args.threads < 1:
            parser.error("--threads must be >= 1")
        import torch
        torch.set_num_threads(args.threads)
        # OpenVINO and ONNX Runtime read their thread counts from the environment
        # rather than from torch, so set those too.
        os.environ["OMP_NUM_THREADS"] = str(args.threads)

    imgsz = args.imgsz
    runtime_info.update({"backend": args.backend, "weights": args.weights,
                         "imgsz": args.imgsz, "int8": args.int8,
                         "torch_threads": args.threads})

    model = load_model(args.weights, args.backend, args.imgsz, args.int8, args.data)

    # Warm the model once so the first real request isn't paying lazy-init cost,
    # and so predictor/trackers exist before any stream swaps state in.
    model.track(np.zeros((540, 960, 3), dtype=np.uint8), persist=True,
                imgsz=args.imgsz, verbose=False)
    stream_trackers.clear()

    threading.Thread(target=sample_cloud_metrics, daemon=True).start()

    print(f"[server] backend={args.backend} weights={args.weights} "
          f"imgsz={args.imgsz} int8={args.int8} threads={args.threads or 'auto'}")
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
