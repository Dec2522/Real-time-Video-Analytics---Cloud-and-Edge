import argparse
import copy
import csv
import os
import shutil

from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import psutil
import time
from collections import deque, defaultdict
import threading

# Count crossings rather than unique object IDs - YOLO often re-IDs the same object, given an inflated count.
from line_counter import (LineCounter, parse_line, line_for, VIDEO_LINES,
                          DEFAULT_COOLDOWN)

app = Flask(__name__)

# Default detection configuration - chosen by offline profiling.
# Can be overrided with command line args
MODEL_WEIGHTS = "yolo11s.pt"
DEFAULT_BACKEND = "openvino"
DEFAULT_INT8 = True
DEFAULT_IMGSZ = 160

HISTORY_LEN = 300
STREAM_TIMEOUT_S = 15

BACKENDS = ("pytorch", "openvino", "onnx")

# Elastic pool dormancy thresholds (seconds)
# Dormancy rather than dropping as variable gating defers inference, but this shouldn't result in a start up cost again
IDLE_TO_DORMANT_S = 5
IDLE_TO_RELEASE_S = 300

REPROVISION_RETRIES = 2

# --- Line counting ---===============================================
DEFAULT_COUNT_MIN_AGE = 2

# Cloud metrics
CSV_DIR = "results"
CLOUD_CSV_PREFIX = ["ts", "elapsed_s", "cpu_percent"]
CLOUD_CSV_SUFFIX = [
    "mem_percent", "mem_used_mb", "net_sent_mb", "net_recv_mb",
    "proc_cpu", "proc_mem_mb", "load_avg",
    "inflight_requests", "active_streams",
    "crossings_total",
    "backend", "weights", "imgsz", "int8",
    "active_workers", "dormant_workers", "cores_assigned", "cores_total",
]


# Cross-Origin Resource sharing - allow dashborad to reach API
@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp

# Edge. cloud, and meta info observeability. Limited length shown in the Dashboard.
edge_metrics_history = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
cloud_metrics_history = deque(maxlen=HISTORY_LEN)
stream_info = {}
lock = threading.Lock()

# Built in main
elastic_pool = None
# Pre-compiled model variants
# Pre-warmed model instances, handed out one per worker. Built once at boot.
model_pool = None

# Most streams the pre-warmed pool is sized for. Past this the pool still
# serves, but a new worker has to build its instance on demand.
DEFAULT_MAX_STREAMS = 8

# Line counting state
count_min_age = DEFAULT_COUNT_MIN_AGE
count_enabled = True
stream_counters = {}
counter_last_frame = {}
counter_last_seen = {}
counter_lock = threading.Lock()

# YOLO detections
CONF = 0.3
VEHICLE_CLASSES = [2, 3, 5, 7] # limit to vehicles (2: car, 3: motorbike, 5: bus, 6: truck)


# Custom botsort required as default IoU too tight when skipping frames - Counts existing boxes as new ones as they may have moved too far.
TRACKER_CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "botsort_custom.yaml")
if not os.path.exists(TRACKER_CFG):
    raise FileNotFoundError(
        f"Required tracker config not found: {TRACKER_CFG}. "
    )


runtime_info = {"backend": DEFAULT_BACKEND, "weights": MODEL_WEIGHTS,
                "imgsz": DEFAULT_IMGSZ, "int8": DEFAULT_INT8}

# Queue depth
inflight = 0
inflight_lock = threading.Lock()

# CPU affinity
_infer_threads = None
_pin_cpus = True
_patched = False

CAN_PIN = hasattr(os, "sched_setaffinity")


def _patch_thread_caps(backend):
    """Force inference onto specified threads.

    Ultralytics doesn't give a way to pass thread caps to openvino/onnx,
    so monkey-patch at runtime.
    """
    global _patched
    if _patched:
        return
    _patched = True

    if backend == "openvino":
        import openvino as ov
        orig_compile = ov.Core.compile_model

        def compile_model(self, model, device_name=None, config=None, **kwargs):
            cfg = dict(config or {})
            if _infer_threads:
                cfg["INFERENCE_NUM_THREADS"] = int(_infer_threads)
            cfg["ENABLE_CPU_PINNING"] = False
            try:
                if device_name is None:
                    return orig_compile(self, model, config=cfg, **kwargs)
                return orig_compile(self, model, device_name, cfg, **kwargs)
            except Exception:
                cfg.pop("ENABLE_CPU_PINNING", None)
                if device_name is None:
                    return orig_compile(self, model, config=cfg, **kwargs)
                return orig_compile(self, model, device_name, cfg, **kwargs)

        ov.Core.compile_model = compile_model

    elif backend == "onnx":
        import onnxruntime
        orig_init = onnxruntime.InferenceSession.__init__

        def init(self, path_or_bytes, sess_options=None, providers=None, **kwargs):
            if sess_options is None and _infer_threads:
                sess_options = onnxruntime.SessionOptions()
                sess_options.intra_op_num_threads = int(_infer_threads)
                sess_options.inter_op_num_threads = 1
            orig_init(self, path_or_bytes, sess_options=sess_options,
                      providers=providers, **kwargs)

        onnxruntime.InferenceSession.__init__ = init


def ultralytics_export_path(weights, backend, int8=False):
    """Reproduce file name ultralytics creates so the compiled model
    can be found by `load_model`
    """
    stem = os.path.splitext(weights)[0]
    if backend == "openvino":
        return f"{stem}_int8_openvino_model" if int8 else f"{stem}_openvino_model"
    return f"{stem}.onnx"


def export_path(weights, backend, size, int8=False):
    """Add image size to model path"""
    base = ultralytics_export_path(weights, backend, int8)
    if backend == "openvino":
        return f"{base}_imgsz{size}"
    return f"{os.path.splitext(base)[0]}_imgsz{size}.onnx"


def load_model(weights, backend, size, int8=False, data=None):
    """Return YOLO model for defiend config.

    If it exists, retrieve, if not, create."""
    # Return standard model
    if backend == "pytorch":
        return YOLO(weights)

    # Set threads to backend
    _patch_thread_caps(backend)
    # Location for saved model
    target = export_path(weights, backend, size, int8)
    # Create if doesn't exist
    if not os.path.exists(target):
        print(f"[server] exporting {weights} -> {backend} (imgsz={size}, int8={int8}); first run only")
        kwargs = {"format": backend, "imgsz": size}
        if backend == "onnx":
            kwargs["opset"] = 20
        if backend == "openvino" and int8:
            kwargs["int8"] = True
            if data:
                kwargs["data"] = data
        YOLO(weights).export(**kwargs)

        # Retrieve model and rename
        produced = ultralytics_export_path(weights, backend, int8)
        if os.path.exists(produced):
            if os.path.isdir(target):
                shutil.rmtree(target)
            elif os.path.exists(target):
                os.remove(target)
            shutil.move(produced, target)

    # Load model
    print(f"Loading {target}")
    return YOLO(target)


def warm_instance(model, size):
    """YOLO lazy-compiles the first inferences, so needs a dummy frame to warm up."""
    model.track(np.zeros((540, 960, 3), dtype=np.uint8), persist=True,
                imgsz=size, verbose=False, tracker=TRACKER_CFG)
    reset_trackers(model)


def reset_trackers(model):
    """Clear a model instance's tracker state before the next stream."""
    predictor = getattr(model, "predictor", None)
    for t in getattr(predictor, "trackers", None) or ():
        t.reset()


def threads_for_worker(idx, n_workers, cores):
    """Divide cores as evenly as possible. Extras go to the low-idx workers."""
    if n_workers >= cores:
        return 1
    base = cores // n_workers
    extra = cores % n_workers
    return base + (1 if idx < extra else 0)


def core_slice_uneven(idx, n_workers, cores):
    """Return CPU IDs for a worker given n workers and n cores."""
    if n_workers >= cores:
        return [idx % cores]
    layout = [threads_for_worker(i, n_workers, cores) for i in range(n_workers)]
    start = sum(layout[:idx])
    return list(range(start, start + layout[idx]))


def working_set(cores, max_streams):
    """Peak instances needed at each thread count across n=1..max_streams.
    e.g. 8 cores, max_streams=8 is {8:1, 4:2, 3:2, 2:4, 1:8}  (17 instances).
    """
    need = {}
    for n in range(1, max_streams + 1):
        counts = {}
        for i in range(n):
            t = threads_for_worker(i, n, cores)
            counts[t] = counts.get(t, 0) + 1
        for t, c in counts.items():
            need[t] = max(need.get(t, 0), c)
    return need




class ModelPool:
    """Warm model instances, grouped by thread count, which workers take and return
    when they are resized or dropped.

    Warmed to avoid the first inference cost and allow rapid resizing.

    Each worker has it's own model instance. Before workers used a tempalte which resulted
    in the sharing the same model thus competing and queuing.
    """

    def __init__(self, weights, backend, size, int8, data):
        self.spec = (weights, backend, size, int8, data)
        self.size = size
        self.idle = {}                      # warm instances
        self.lock = threading.Lock()
        self.build_lock = threading.Lock()  # separate lock for compiling

    def _build(self, threads):
        """Compile and warm one model instance for defined thread count."""
        global _infer_threads
        with self.build_lock:
            previous = _infer_threads
            try:
                _infer_threads = threads
                model = load_model(*self.spec)
                warm_instance(model, self.size)
            finally:
                _infer_threads = previous
        return model

    def prebuild(self, cores, max_streams):
        """Compile every instance needed for number of cores and max streams."""
        need = working_set(cores, max_streams)
        total = sum(need.values())
        print(f"Pre-warming {total} model instance(s) for up to "
              f"{max_streams} streams on {cores} cores: "
              + ", ".join(f"{c}x{t}t" for t, c in sorted(need.items(), reverse=True))) # t = threads, c = count
        proc = psutil.Process()
        rss0 = proc.memory_info().rss   # RAM being used
        for t in sorted(need, reverse=True):
            self.idle[t] = [self._build(t) for _ in range(need[t])]
        rss1 = proc.memory_info().rss
        # Print memory usage info
        print(f"Pre-warmed {total} instances, "
              f"RSS +{(rss1 - rss0) / 1e6:.0f}MB "
              f"({(rss1 - rss0) / max(total, 1) / 1e6:.0f}MB each), "
              f"total {rss1 / 1e6:.0f}MB")

    def acquire(self, threads):
        """For a thread, return a warm isntance."""
        with self.lock:
            return self.idle[threads].pop()

    def release(self, threads, model):
        """Hand an instance back, cleared of the departing stream's tracks."""
        reset_trackers(model)
        with self.lock:
            self.idle[threads].append(model)

    def stats(self):
        with self.lock:
            return {"models_idle": sum(len(b) for b in self.idle.values())}


class Worker:
    """One worker has one model instance and one stream pinned to it."""

    def __init__(self, idx, model, threads, cores):
        self.idx = idx
        self.lock = threading.Lock()   # serialises this worker's own inferences
        self.stream = None
        self.cores = cores             # CPUs this worker is entitled to
        self.threads = threads
        self.dormant = False
        self.last_inference = time.time()
        self.gen = 0                   # times this stream has been resized
        self._pinned = threading.local()
        self._shutdown = False
        self.model = model

    def _pin_this_thread(self):
        # If already pinned exit
        if getattr(self._pinned, "done", False):
            return
        # If pinning is enabled and using Linux 
        if self.cores and _pin_cpus and CAN_PIN:
            os.sched_setaffinity(0, self.cores)
        self._pinned.done = True

    def track(self, frame, stream_id):
        """Run inference on a frame. Return detections and timing info."""
        self._pin_this_thread()
        t_wait = time.time()
        with self.lock:
            if self._shutdown:
                raise RuntimeError("worker shut down")
            t_infer = time.time()
            # Inference
            results = self.model.track(frame, persist=True, conf=CONF, verbose=False,
                                       imgsz=imgsz,
                                       classes=VEHICLE_CLASSES,
                                       tracker=TRACKER_CFG)
            inference_ms = (time.time() - t_infer) * 1000

        return results, inference_ms, (t_infer - t_wait) * 1000

    def adopt_tracks_from(self, other):
        """Copy the old worker's tracker ('other') state so track IDs continue across a resize."""
        self.model.predictor.trackers = copy.deepcopy(other.model.predictor.trackers)

    def shutdown(self):
        with self.lock:
            self._shutdown = True
            self.stream = None


class ElasticPool:
    """
    A pool of workers that can dynamically adjust their resource allocation based on demand.
    """

    def __init__(self, cores):
        self.cores = cores
        self.workers = {}
        self.lock = threading.Lock()

    def _active_ids(self):
        """Streams whose workers are counted in the current CPU split."""
        return [sid for sid, w in self.workers.items() if not w.dormant]

    def _rebalance(self):
        """Resize each active worker to its share under the current available CPU.
        Dormant workers keep their old sizing but don't count against anyone.
        """
        active_ids = self._active_ids()
        n = len(active_ids)
        if n == 0:
            return
        for new_idx, sid in enumerate(active_ids):
            w = self.workers[sid]
            new_t = threads_for_worker(new_idx, n, self.cores)
            # If different threads, reprovision the worker with a new model
            if w.threads != new_t:
                self._reprovision(sid, new_idx, n)
            # If same, update the index and core slice
            else:
                w.idx = new_idx
                w.cores = core_slice_uneven(new_idx, n, self.cores)
                w._pinned = threading.local()

    def _reprovision(self, sid, idx, n_workers):
        """For a stream, replace its worker with a new model and thread count,
        while carrying over the tracker state. The old worker is shut down and its
        model returned to the pool.
        """
        old = self.workers[sid]
        # Compute new threads
        new_threads = threads_for_worker(idx, n_workers, self.cores)
        cores_for_worker = core_slice_uneven(idx, n_workers, self.cores)
        # Initialise
        new = Worker(idx, model_pool.acquire(new_threads), new_threads,
                     cores_for_worker)
        new.stream = old.stream
        new.dormant = old.dormant
        new.last_inference = old.last_inference
        new.gen = old.gen + 1
        # Carry over tracker state
        new.adopt_tracks_from(old)
        old.shutdown()
        # Swap stream to new worker
        self.workers[sid] = new
        # Release old model
        model_pool.release(old.threads, old.model)
        print(f"Stream {sid}: reprovisioned to {new_threads}t on {cores_for_worker}"
              f"{' (dormant)' if new.dormant else ''}")

    def ensure_capacity(self, stream_id):
        """Return the worker for `stream_id`. Wake it if dormant, create it if
        this is the first sight of the stream.
        """
        with self.lock:
            w = self.workers.get(stream_id)
            if w is not None:
                if w.dormant:
                    # Wake if dormant and resize worker thread allocation
                    w.dormant = False
                    w.last_inference = time.time()
                    print(f"Stream {stream_id}: waking dormant worker")
                    self._rebalance()
                    w = self.workers[stream_id]
                return w

            # If new stream
            active_before = self._active_ids()
            n_active_after = len(active_before) + 1
            # Resize existing workers
            for new_idx, sid in enumerate(active_before):
                other = self.workers[sid]
                new_t = threads_for_worker(new_idx, n_active_after, self.cores)
                if other.threads != new_t:
                    self._reprovision(sid, new_idx, n_active_after) # model swap
                else:
                    # same model but different cores 
                    other.idx = new_idx
                    other.cores = core_slice_uneven(new_idx, n_active_after, self.cores)
                    other._pinned = threading.local()

            # Create new worker for new stream
            new_idx = n_active_after - 1
            new_threads = threads_for_worker(new_idx, n_active_after, self.cores)
            cores_for_worker = core_slice_uneven(new_idx, n_active_after, self.cores)
            new_worker = Worker(new_idx, model_pool.acquire(new_threads),
                                new_threads, cores_for_worker)
            new_worker.stream = stream_id
            self.workers[stream_id] = new_worker
            print(f"Stream {stream_id}: new worker {new_idx}, "
                  f"{new_threads}t on {cores_for_worker}")
            return new_worker

    def make_dormant(self, stream_id):
        """Changing flow gate means a stream may not infer for a while. Resource is taken but unused, 
        so instead make the worker dormant and reallocate its resource. Dormancy keeps a warm model so 
        that when a stream resumes the restart is fast.
        """
        with self.lock:
            w = self.workers.get(stream_id)
            if w is None or w.dormant:
                return
            w.dormant = True
            print(f"Stream {stream_id}: worker marked dormant")
            self._rebalance()

    def release(self, stream_id):
        """Stream ended, drop worker."""
        with self.lock:
            w = self.workers.pop(stream_id, None)
            if w is None:
                return
            w.shutdown()
            model_pool.release(w.threads, w.model)
            print(f"Stream {stream_id}: worker released")
            self._rebalance()

    def touch(self, stream_id):
        """Update model last_inference so 'reaper' knows a stream is still active."""
        w = self.workers.get(stream_id)
        if w is not None:
            w.last_inference = time.time()


def update_crossings(stream_id, video, frame_num, dets):
    """Update the line count for a stream."""
    if not count_enabled:
        return None

    with counter_lock:
        now = time.time()
        last_frame = counter_last_frame.get(stream_id)
        last_seen = counter_last_seen.get(stream_id)
        restarted = ((last_frame is not None and frame_num < last_frame)
                     or (last_seen is not None and now - last_seen > STREAM_TIMEOUT_S))
        # If a restarted stream declare old count and reset the counter
        if restarted:
            done = stream_counters.pop(stream_id, None)
            if done is not None:
                print(f"Stream {stream_id} restarted - previous run counted "
                      f"{done.total} crossing(s) over {last_frame} frames")
        counter_last_frame[stream_id] = frame_num
        counter_last_seen[stream_id] = now

        counter = stream_counters.get(stream_id)
        if counter is None:
            # lines are predefined for each video - this would be part of an offline profiling step
            line = line_for(video)
            counter = LineCounter(tuple(line), count_min_age, None, DEFAULT_COOLDOWN,
                                  once_per_track=True)
            stream_counters[stream_id] = counter
        # run new detections through counter
        counter.update(frame_num, dets)
        return {"in": counter.counts["in"], "out": counter.counts["out"],
                "total": counter.total, "unique": len(counter.unique)}


def crossings_snapshot():
    """Get line counts for all streams. Used for metrics and dashboard."""
    with counter_lock:
        per_stream = {sid: {"in": c.counts["in"], "out": c.counts["out"],
                            "total": c.total, "unique": len(c.unique),
                            "line": list(c.line)}
                      for sid, c in stream_counters.items()}
    return per_stream, sum(v["total"] for v in per_stream.values())


def pool_snapshot():
    """Per-worker view of the pool for the dashboard: which streams sit on
    which worker, whether it is active or dormant, and the cores it holds.
    """
    now = time.time()
    cores_total = psutil.cpu_count(logical=True) or 1

    with elastic_pool.lock:
        snap = [(sid, w.idx, w.stream, w.dormant, w.threads,
                list(w.cores or []), w.last_inference, w.gen)
                for sid, w in elastic_pool.workers.items()]
    # Dict shape for each worker
    workers = [{
            "owner": sid,
            "slot": idx,
            "gen": gen,
            "stream": stream,
            "state": "dormant" if dormant else "active",
            "threads": threads,
            "cores": cores,
            "cores_stale": dormant,
            "idle_s": round(now - last, 1),
    } for sid, idx, stream, dormant, threads, cores, last, gen in snap]
    # Active first, then by slot, so the layout does not jump around.
    workers.sort(key=lambda w: (w["state"] != "active", w["slot"], str(w["owner"])))
    active = [w for w in workers if w["state"] == "active"]
    return {
        "cores_total": cores_total,
        "cores_assigned": sum(w["threads"] or 0 for w in active),
        "active_workers": len(active),
        "dormant_workers": len(workers) - len(active),
        **model_pool.stats(),
        "idle_to_dormant_s": IDLE_TO_DORMANT_S,
        "idle_to_release_s": IDLE_TO_RELEASE_S,
        "pinning": bool(_pin_cpus and CAN_PIN),
        "workers": workers,
    }


def active_streams(now=None):
    """Different to elastic pool active streams - this marks a stream is
    still active if it is sending metrics, even if its worker is dormant / no 
    inference is happening. Used for dashboard."""
    now = now or time.time()
    return [sid for sid, info in stream_info.items()
            if now - info["last_seen"] <= STREAM_TIMEOUT_S]


def run_tag(args):
    """Make metrics file name for this run. Used for offline line analysis, not the dashboard."""
    stem = os.path.splitext(os.path.basename(args.weights))[0]
    parts = [time.strftime("%Y%m%d-%H%M%S"), args.backend, stem, f"imgsz{args.imgsz}"]
    if args.int8:
        parts.append("int8")
    return "_".join(parts)


def sample_cloud_metrics(csv_path=None):
    """Sample cloud metrics and pool snapshot every second and write to in-memory deque and CSV."""
    proc = psutil.Process()
    header = writer = csv_file = None
    # Set up CSV if asked for
    if csv_path:
        n_cores = len(psutil.cpu_percent(percpu=True))
        header = (CLOUD_CSV_PREFIX + [f"cpu_core{i}" for i in range(n_cores)]
                  + CLOUD_CSV_SUFFIX)
        csv_file = open(csv_path, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(header)
        csv_file.flush()
        print(f"[server] cloud metrics -> {csv_path}")

    t_start = time.time()
    while True:
        now = time.time()
        # cloud metrics
        record = {
            "ts": now,
            "cpu_percent": psutil.cpu_percent(),
            "cpu_per_core": psutil.cpu_percent(percpu=True),
            "mem_percent": psutil.virtual_memory().percent,
            "mem_used_mb": round(psutil.virtual_memory().used / 1e6, 1),
            "net_sent_mb": round(psutil.net_io_counters().bytes_sent / 1e6, 1),
            "net_recv_mb": round(psutil.net_io_counters().bytes_recv / 1e6, 1),
            "proc_cpu": proc.cpu_percent(),
            "proc_mem_mb": round(proc.memory_info().rss / 1e6, 1),
            "load_avg": psutil.getloadavg()[0],
            "inflight_requests": inflight,
            "crossings_total": crossings_snapshot()[1],
        }
        # Pool snapshot
        pool_state = pool_snapshot()
        record.update({k: pool_state[k] for k in
                       ("active_workers", "dormant_workers",
                        "cores_assigned", "cores_total")})
        # Write to in-memory
        with lock:
            record["active_streams"] = len(active_streams(now))
            cloud_metrics_history.append(record)
        # Write to CSV
        if writer:
            flat = dict(record, elapsed_s=round(now - t_start, 1),
                        weights=runtime_info["weights"],
                        backend=runtime_info["backend"],
                        imgsz=runtime_info["imgsz"],
                        int8=runtime_info["int8"])
            flat.update({f"cpu_core{i}": v for i, v in enumerate(record["cpu_per_core"])})
            writer.writerow([flat.get(c) for c in header])
            csv_file.flush()

        time.sleep(1)


def stream_reaper():
    """Active workers idle past N seconds are marked dormant, and dormant workers idle past M seconds are released."""
    while True:
        time.sleep(2)
        now = time.time()
        with elastic_pool.lock:
            items = [(sid, w.dormant, w.last_inference)
                     for sid, w in elastic_pool.workers.items()]
        for sid, dormant, last in items:
            idle = now - last
            if not dormant and idle > IDLE_TO_DORMANT_S:
                elastic_pool.make_dormant(sid)
            elif dormant and idle > IDLE_TO_RELEASE_S:
                elastic_pool.release(sid)


@app.route("/detect", methods=["POST"])
def detect():
    """Main request handler. Accepts a JPEG frame, runs inference, and returns detections and counts."""
    global inflight
    stream_id = request.args.get("stream", "0")
    req_frame = request.args.get("frame", type=int) # frame number for line counting
    video = request.args.get("video")

    # Decode JPEG to BGR frame
    # Timed for decode time metric.
    t0 = time.time()
    jpg_bytes = request.data
    arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    decode_ms = (time.time() - t0) * 1000

    # Run inference on the frame. If a worker is swapped mid-request, retry.
    for attempt in range(REPROVISION_RETRIES + 1):
        worker = elastic_pool.ensure_capacity(stream_id)

        with inflight_lock:
            inflight += 1
        try:
            results, inference_ms, queue_wait_ms = worker.track(frame, stream_id)
            break
        except RuntimeError:
            if attempt == REPROVISION_RETRIES:
                raise
            print(f"Stream {stream_id}: worker swapped mid-request, retrying")
        finally:
            with inflight_lock:
                inflight -= 1

    # Update last inference time
    elastic_pool.touch(stream_id)

    # Get YOLO tensors and convert to JSON compatible dicts
    boxes = results[0].boxes
    dets = []
    if boxes.id is not None:
        for tid, cls, conf, (cx, cy, bw, bh) in zip(
            boxes.id.int().tolist(),
            boxes.cls.int().tolist(),
            boxes.conf.tolist(),
            boxes.xywhn.tolist(),
        ):
            dets.append({"id": tid, "label": results[0].names[cls],
                         "conf": round(conf, 2),
                         "cx": round(cx, 4), "cy": round(cy, 4),
                         "w": round(bw, 4), "h": round(bh, 4)})

    labels = [d["label"] for d in dets]
    counts = {l: labels.count(l) for l in set(labels)}

    # Stream info for dashboard. Snap shot of elastic pool wouldn't give history of killed streams.
    with lock:
        info = stream_info.setdefault(stream_id, {"frames": 0, "video": None, "last_seen": 0})
        info["frames"] += 1
        info["last_seen"] = time.time()
        if video and not info["video"]:
            info["video"] = video
        served = info["frames"]

    # Detections to line counter
    crossings = update_crossings(stream_id, video, req_frame if req_frame is not None
                                 else served, dets)

    # Return metrics back to the edge. 
    # I wanted to do this so I could measure end-to-end and keep all metrics in one place
    # and a future extension is for the edge to act on the detections.
    return jsonify({
        "stream_id": stream_id,
        "detections": dets,
        "counts": counts,
        "count_in": crossings["in"] if crossings else None,
        "count_out": crossings["out"] if crossings else None,
        "count_total": crossings["total"] if crossings else None,
        "count_unique": crossings["unique"] if crossings else None,
        "decode_ms": round(decode_ms, 1),
        "inference_ms": round(inference_ms, 1),
        "queue_wait_ms": round(queue_wait_ms, 1),
        "backend": runtime_info["backend"],
        "worker_id": worker.idx,
        "infer_threads": worker.threads,
        "served_imgsz": runtime_info["imgsz"],
        "served_weights": runtime_info["weights"],
        "served_conf": CONF,
        "served_int8": runtime_info["int8"],
    })


@app.route("/disconnect", methods=["POST"])
def disconnect():
    """Forced disconnet of a stream."""
    stream_id = request.args.get("stream", "0")
    elastic_pool.release(stream_id)
    return jsonify({"status": "released", "stream": stream_id})


@app.route("/metrics", methods=["POST"])
def receive_metrics():
    """Endpoint for edge to push metrics to the cloud."""
    # Parse and coerce stream id
    data = request.get_json()
    stream_id = str(data.get("stream_id", "0"))
    data["stream_id"] = stream_id
    # Write to a per stream deque for dashboard to show history
    with lock:
        edge_metrics_history[stream_id].append(data)
        info = stream_info.setdefault(stream_id, {"frames": 0, "video": None, "last_seen": 0})
        info["last_seen"] = time.time()
        if data.get("video"):
            info["video"] = data["video"]
    return jsonify({"status": "ok"})


@app.route("/metrics/data")
def metrics_data():
    """Dashboard fetches data from here"""
    crossings, crossings_total = crossings_snapshot()
    pool_state = pool_snapshot()
    with lock:
        now = time.time()
        live = set(active_streams(now))
        # convert deque lists to JSON
        return jsonify({
            "edge": {sid: list(recs) for sid, recs in edge_metrics_history.items()},
            "cloud": list(cloud_metrics_history),
            "runtime": runtime_info,
            "pool": pool_state,
            "crossings_total": crossings_total,
            "streams": [
                {
                    "id": sid,
                    "video": info["video"],
                    "frames": info["frames"],
                    "last_seen": info["last_seen"],
                    "active": sid in live,
                    "crossings": crossings.get(sid),
                }
                for sid, info in sorted(stream_info.items())
            ],
        })


def main():
    global imgsz, _infer_threads, _pin_cpus, count_min_age, elastic_pool
    global model_pool
    global IDLE_TO_DORMANT_S, IDLE_TO_RELEASE_S

    parser = argparse.ArgumentParser(
        description="Cloud inference service. Each stream gets its own worker, and "
                    "the CPU is redivided between them as streams arrive and leave.")
    parser.add_argument("--backend", choices=BACKENDS, default=DEFAULT_BACKEND,
                        help=f"inference runtime (default {DEFAULT_BACKEND})")
    parser.add_argument("--weights", default=MODEL_WEIGHTS,
                        help=f"model weights (default {MODEL_WEIGHTS})")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                        help=f"inference resolution (default {DEFAULT_IMGSZ})")
    parser.add_argument("--no-int8", dest="int8", action="store_false",
                        help="export and run the model at full precision. INT8 is on "
                             "by default and applies to --backend openvino only")
    parser.set_defaults(int8=DEFAULT_INT8)
    parser.add_argument("--data", default=None,
                        help="calibration dataset for the INT8 export. Without it "
                             "Ultralytics falls back to its own and will try to "
                             "download it, which fails on a host with no internet")
    parser.add_argument("--max-streams", type=int, default=DEFAULT_MAX_STREAMS,
                        metavar="N",
                        help=f"how many concurrent streams to pre-warm model "
                             f"instances for (default {DEFAULT_MAX_STREAMS}). Each "
                             f"worker holds its own instance, and the thread cap is "
                             f"fixed at compile time, so the pool holds a warm one of "
                             f"every size the splits for 1..N streams can need. Going "
                             f"past N still works but the first resize to an "
                             f"un-warmed size pays a compile")
    parser.add_argument("--pin", action="store_true",
                        help="pin each worker to its slice of the cores. Linux only, "
                             "and it binds the request thread rather than the "
                             "inference runtime's own pool, so expect a tendency "
                             "rather than hard isolation")
    parser.add_argument("--idle-dormant", type=float, default=IDLE_TO_DORMANT_S,
                        metavar="SECONDS",
                        help=f"seconds without a /detect before a stream's worker "
                             f"goes dormant (default {IDLE_TO_DORMANT_S}).")
    parser.add_argument("--idle-release", type=float, default=IDLE_TO_RELEASE_S,
                        metavar="SECONDS",
                        help=f"seconds dormant before the worker is destroyed "
                             f"(default {IDLE_TO_RELEASE_S}). The default is long "
                             f"enough that a test run never reaches it - drop it to "
                             f"~20 to observe release as well as dormancy")
    parser.add_argument("--count-min-age", type=int, default=DEFAULT_COUNT_MIN_AGE)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--no-csv", dest="write_csv", action="store_false")
    args = parser.parse_args()

    # Validate args
    # INT8 only supported by openvino, so disable if another backend is passed.
    if args.int8 and args.backend != "openvino":
        args.int8 = False

    if args.imgsz < 32 or args.imgsz % 32:
        parser.error("--imgsz must be a multiple of 32 (the model's stride)")

    if args.max_streams < 1:
        parser.error("--max-streams must be >= 1")

    if args.idle_dormant <= 0 or args.idle_release <= 0:
        parser.error("--idle-dormant and --idle-release must be > 0")
    if args.idle_release < args.idle_dormant:
        parser.error("--idle-release must be >= --idle-dormant")

    # Override defaults with args
    IDLE_TO_DORMANT_S = args.idle_dormant
    IDLE_TO_RELEASE_S = args.idle_release
    count_min_age = args.count_min_age

    # Does the video passed have a line configured?
    configured = sorted(v for v, line in VIDEO_LINES.items() if line)
    print(f"Counting lines set for: {', '.join(configured)}")

    cores = psutil.cpu_count(logical=True) or 1
    _infer_threads = cores
    _pin_cpus = args.pin

    # Cap thread pools to number of cores.
    # Pytorch doesn't respect OMP_NUM_THREADS, so set it directly.
    if args.backend == "pytorch":
        import torch
        torch.set_num_threads(cores)
    os.environ["OMP_NUM_THREADS"] = str(cores)

    # Compile and warm every instance the pool will need
    imgsz = args.imgsz
    runtime_info.update({"backend": args.backend, "weights": args.weights,
                         "imgsz": args.imgsz, "int8": args.int8})

    model_pool = ModelPool(args.weights, args.backend, args.imgsz,
                           args.int8, args.data)
    model_pool.prebuild(cores, args.max_streams)
    elastic_pool = ElasticPool(cores)

    csv_path = None
    if args.write_csv:
        csv_path = args.csv or os.path.join(CSV_DIR, f"cloud_{run_tag(args)}.csv")
        parent = os.path.dirname(csv_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    # Start background threads for metrics and reaper
    threading.Thread(target=sample_cloud_metrics, args=(csv_path,), daemon=True).start()
    threading.Thread(target=stream_reaper, daemon=True).start()

    # Print config
    print(f"backend={args.backend} weights={args.weights} "
          f"imgsz={args.imgsz} int8={args.int8} cores={cores} "
          f"pinning={'on' if args.pin and CAN_PIN else 'off'}")
    # Start the Flask server
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()