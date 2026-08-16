import argparse
import copy
import csv
import math
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

app = Flask(__name__)

MODEL_WEIGHTS = "yolo11n.pt"
HISTORY_LEN = 300          # samples retained per series for the dashboard
STREAM_TIMEOUT_S = 15      # no traffic for this long => stream counted as finished

CSV_DIR = "results"        # cloud metrics CSVs land here, one per run

# Cloud metrics CSV layout. The per-core columns sit between these two blocks and
# are generated at open time, since the core count isn't known until runtime.
# The run's config is repeated on every row so a CSV can be analysed on its own
# without needing the filename parsed - matches how the edge CSV carries `backend`.
CLOUD_CSV_PREFIX = ["ts", "elapsed_s", "cpu_percent"]
CLOUD_CSV_SUFFIX = [
    "mem_percent", "mem_used_mb", "net_sent_mb", "net_recv_mb",
    "proc_cpu", "proc_mem_mb", "load_avg",
    "inflight_requests", "active_streams",
    # `threads` is the per-worker inference thread cap; `workers` is how many
    # model instances are sharing the box. threads*workers is the core budget.
    "backend", "weights", "imgsz", "int8", "threads", "workers",
]

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

# --- --- Detector pool --- ---
# One model instance per worker, rather than one shared model behind a lock.
#
# The old design serialised every inference under a single model_lock. That lock
# was not only a throughput choice - it was load-bearing for correctness, because
# one YOLO instance owns one predictor, which owns one tracker list. Two streams
# inferring at once on that predictor would blend their track IDs, so per-stream
# tracker state had to be swapped in and out around every call.
#
# Giving each worker its own YOLO instance removes both problems at once: the
# predictor, and therefore the tracker state, is private to the worker. Nothing
# needs swapping and nothing needs a global lock, so streams infer genuinely in
# parallel. Oversubscription is handled instead by capping each worker's model to
# a slice of the cores (see _patch_thread_caps), so the workers partition the CPU
# rather than each trying to use all of it.
pool = []                         # list[Worker], built in main()
stream_worker = {}                # stream_id -> Worker, fixed for the run
assign_lock = threading.Lock()    # guards first-sight worker assignment

# Detection settings live in one place so every worker is identical and results
# stay comparable with the earlier single-model runs.
CONF = 0.3
VEHICLE_CLASSES = [2, 3, 5, 7]    # Classes = vehicles we want to detect

# Inference resolution
imgsz = DEFAULT_IMGSZ

# Describes the active backend, surfaced to the dashboard so a run can be
# attributed to the runtime that produced it.
runtime_info = {"backend": "pytorch", "weights": MODEL_WEIGHTS,
                "imgsz": DEFAULT_IMGSZ, "int8": False, "torch_threads": None,
                "workers": 1, "infer_threads": None}

inflight = 0  # number of requests being processed
inflight_lock = threading.Lock()

# Thread cap applied to the NEXT model loaded. The patches installed by
# _patch_thread_caps read this at load time, which is how each worker ends up
# with its own bounded thread pool.
_infer_threads = None
_pin_cpus = True      # set from --no-pin; see Worker._apply_affinity
_patched = False

# Whether this platform can pin threads to cores at all (Linux: yes).
CAN_PIN = hasattr(os, "sched_setaffinity")


def core_slice(idx, threads, cores):
    """The cores worker `idx` runs on: `threads` of them, starting where the
    previous worker left off.

    4 workers x 2 threads on 8 cores -> {0,1} {2,3} {4,5} {6,7}: no overlap, so
    the workers genuinely run side by side instead of contending. When
    workers*threads exceeds the core count (3 x 3 on 8) the last slice wraps and
    shares a core with the first - deliberate, since the alternative is leaving
    a worker with fewer threads than it was compiled for.
    """
    start = (idx * threads) % cores
    return sorted({(start + j) % cores for j in range(min(threads, cores))})


def _patch_thread_caps(backend):
    """Make the per-worker thread cap actually reach the inference runtime.

    Ultralytics offers no way to pass runtime config through `YOLO(...)`, so the
    relevant constructor is wrapped once, here. This matters more than it looks:
    setting OMP_NUM_THREADS (as this server used to) does nothing to OpenVINO,
    which ships a TBB build and ignores it. The only reliable knob is the
    INFERENCE_NUM_THREADS property, supplied at compile time.
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
            # OpenVINO's own ENABLE_CPU_PINNING is deliberately NOT set here.
            # Ultralytics builds a separate ov.Core per model, so each worker's
            # plugin believes it owns the whole machine and pins to the same
            # low-numbered cores - four workers all land on cores 0-1. Core
            # placement is done with OS affinity in Worker instead, where the
            # allocation is explicit and non-overlapping.
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

    # pytorch: torch.set_num_threads() is process-global, so it cannot differ per
    # worker. It is applied once in main() instead - no loss, since every worker
    # gets the same cap anyway.


def ultralytics_export_path(weights, backend, int8=False):
    """Where ultralytics itself writes the export - no imgsz in its naming."""
    stem = os.path.splitext(weights)[0]
    if backend == "openvino":
        # The exporter suffixes int8 runs differently. Without this the cached
        # export is never found, so --int8 re-exports every launch and then tries
        # to load a directory that was never written.
        return f"{stem}_int8_openvino_model" if int8 else f"{stem}_openvino_model"
    return f"{stem}.onnx"


def export_path(weights, backend, size, int8=False):
    """Where this build is cached - keyed on imgsz as well as backend.

    An exported artefact has a STATIC input shape, so one exported at 640 cannot
    serve a 960 request; it just quietly runs at 640. Caching under ultralytics'
    imgsz-free name meant the first export won permanently and every later
    --imgsz was silently ignored on openvino/onnx.
    """
    base = ultralytics_export_path(weights, backend, int8)
    if backend == "openvino":
        return f"{base}_imgsz{size}"
    return f"{os.path.splitext(base)[0]}_imgsz{size}.onnx"


def load_model(weights, backend, size, int8=False, data=None):
    """Load `weights` under the chosen backend, exporting on first use.

    The export is cached on disk, so only the first run of a given
    (weights, backend, imgsz, int8) combination pays the conversion cost. Called
    once per worker: the export on disk is shared, the compiled instance is not.
    """
    if backend == "pytorch":
        return YOLO(weights)

    _patch_thread_caps(backend)

    target = export_path(weights, backend, size, int8)
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

        # Rehome under the imgsz-keyed name, before the next resolution's export
        # overwrites the path ultralytics just wrote to.
        produced = ultralytics_export_path(weights, backend, int8)
        if os.path.exists(produced):
            if os.path.isdir(target):
                shutil.rmtree(target)
            elif os.path.exists(target):
                os.remove(target)
            shutil.move(produced, target)

    print(f"[server] loading {target}")
    return YOLO(target)


def effective_imgsz(model):
    """The input size a loaded model actually runs at, read back after warmup.

    Read defensively - it is a diagnostic, and no ultralytics version guarantees
    where this lives.
    """
    predictor = getattr(model, "predictor", None)
    for obj in (getattr(predictor, "model", None), predictor,
                getattr(predictor, "args", None), model):
        size = getattr(obj, "imgsz", None)
        if isinstance(size, (list, tuple)) and len(size) >= 2:
            return [int(size[0]), int(size[1])]
        if isinstance(size, int) and size > 0:
            return [size, size]
    return None


class Worker:
    """One model instance and the streams pinned to it.

    A worker is the unit of parallelism: its model, predictor and tracker state
    are private, so two workers never interact. `lock` therefore only contends
    when more streams are running than there are workers and two of them landed
    on the same one.
    """

    def __init__(self, idx, weights, backend, size, int8=False, data=None, cores=None):
        self.idx = idx
        self.lock = threading.Lock()
        self.streams = set()          # stream ids routed here
        self.trackers = {}            # stream_id -> that stream's tracker list
        self.pristine = None          # clean tracker list, forked for each new stream
        self.cores = cores            # the cores this worker is confined to
        self._pinned = threading.local()

        # NOT loaded under an affinity mask, deliberately. OpenVINO runs on TBB,
        # which keeps ONE thread pool per process: separate compiled models get
        # separate task arenas but share the underlying worker threads. Those
        # threads are created during the first compile, so masking the loading
        # thread pins the whole shared pool to the first worker's cores and every
        # worker then contends for them. In-process workers can be given a thread
        # budget (INFERENCE_NUM_THREADS) but not private cores.
        self.model = load_model(weights, backend, size, int8, data)

    def _apply_affinity(self):
        """Confine the calling thread to this worker's cores."""
        if self.cores and _pin_cpus and CAN_PIN:
            os.sched_setaffinity(0, self.cores)

    def _release_affinity(self):
        """Hand the calling thread back the whole machine."""
        if _pin_cpus and CAN_PIN:
            os.sched_setaffinity(0, range(psutil.cpu_count(logical=True) or 1))

    def _pin_this_thread(self):
        """Pin the current request thread to this worker's cores, once.

        Inference runs partly on the calling thread, so without this the Flask
        thread would float across the machine while the pool threads stayed put.
        Cheap: one syscall the first time a given thread serves this worker.
        """
        if not getattr(self._pinned, "done", False):
            self._apply_affinity()
            self._pinned.done = True

    def warmup(self, size):
        """Run one dummy frame so predictor and trackers exist before real traffic.

        This also avoids the first real request paying lazy-init cost, and takes
        the clean tracker snapshot. When a worker has to serve a second stream,
        that stream starts from a deep copy of this snapshot - which is what
        keeps its track IDs independent instead of inheriting, or resetting,
        another stream's state.
        """
        self.model.track(np.zeros((540, 960, 3), dtype=np.uint8), persist=True,
                         imgsz=size, verbose=False)
        predictor = getattr(self.model, "predictor", None)
        if predictor is not None:
            for t in predictor.trackers:
                t.reset()
            try:
                self.pristine = copy.deepcopy(predictor.trackers)
            except Exception as e:
                print(f"[server] worker {self.idx}: tracker snapshot failed ({e}); "
                      f"isolation degrades if this worker serves >1 stream")
        self.trackers.clear()

    def _swap_in(self, stream_id, predictor):
        """Point this worker's predictor at `stream_id`'s tracker state.

        Only reached when a worker serves more than one stream. Each stream gets
        its own list of tracker objects - forked from the pristine snapshot, not
        aliased to the predictor's current list - so their IDs stay separate.
        """
        saved = self.trackers.get(stream_id)
        if saved is None:
            if self.pristine is not None:
                saved = copy.deepcopy(self.pristine)
            else:
                # No snapshot to fork from: fall back to resetting in place.
                saved = predictor.trackers
                for t in saved:
                    t.reset()
        predictor.trackers = saved

    def track(self, frame, stream_id):
        """Track one frame. Returns (results, inference_ms, queue_wait_ms)."""
        self._pin_this_thread()
        t_wait = time.time()
        with self.lock:
            t_infer = time.time()
            # Fast path: a worker serving a single stream owns that stream's
            # tracker outright, so there is nothing to swap.
            shared = len(self.streams) > 1
            predictor = getattr(self.model, "predictor", None)
            if shared and predictor is not None:
                self._swap_in(stream_id, predictor)

            # run YOLO with track for persistent object IDs across frames
            results = self.model.track(frame, persist=True, conf=CONF, verbose=False,
                                       imgsz=imgsz,            # identical across backends
                                       classes=VEHICLE_CLASSES)
            inference_ms = (time.time() - t_infer) * 1000

            if shared:
                # predictor only exists after the first call - re-fetch on frame 1
                predictor = predictor or getattr(self.model, "predictor", None)
                if predictor is not None:
                    self.trackers[stream_id] = predictor.trackers

        # queue wait is time spent behind another stream on this worker: ~0 once
        # workers >= streams, which is the whole point of the pool
        return results, inference_ms, (t_infer - t_wait) * 1000


def worker_for(stream_id):
    """Route a stream to its worker, assigning on first sight.

    Assignment is round-robin over the pool and never changes for the life of the
    run. Stickiness is not an optimisation: a stream's tracker lives inside one
    worker, so moving a stream mid-run would restart its IDs from scratch and
    inflate the unique-object count.
    """
    with assign_lock:
        worker = stream_worker.get(stream_id)
        if worker is None:
            worker = pool[len(stream_worker) % len(pool)]
            stream_worker[stream_id] = worker
            worker.streams.add(stream_id)
            note = "  (sharing - more streams than workers)" if len(worker.streams) > 1 else ""
            print(f"[server] stream {stream_id} -> worker {worker.idx}{note}")
        return worker


def active_streams(now=None):
    """Show what streams are still active."""
    now = now or time.time()
    return [sid for sid, info in stream_info.items()
            if now - info["last_seen"] <= STREAM_TIMEOUT_S]


def run_tag(args):
    """Filename stem describing this run's config, so runs don't overwrite each other."""
    stem = os.path.splitext(os.path.basename(args.weights))[0]
    parts = [time.strftime("%Y%m%d-%H%M%S"), args.backend, stem, f"imgsz{args.imgsz}"]
    if args.int8:
        parts.append("int8")
    # workers and per-worker threads define the concurrency config, so both go in
    # the filename - a sweep over stream counts otherwise produces identical stems
    parts.append(f"w{args.workers}")
    if args.infer_threads:
        parts.append(f"t{args.infer_threads}")
    return "_".join(parts)


# --- background thread: sample cloud resources once a second ---
def sample_cloud_metrics(csv_path=None):
    proc = psutil.Process() # the current process

    header = writer = csv_file = None
    if csv_path:
        # One column per logical core. Sampled once up front to fix the width -
        # the header has to be written before the first row.
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
        # Sampled outside the lock: these are syscalls, and /detect contends for
        # the same lock on every frame.
        now = time.time()
        record = {
            "ts": now,
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
        }
        with lock:
            record["active_streams"] = len(active_streams(now))
            # add cloud metrics
            cloud_metrics_history.append(record)

        if writer:
            # Flat view of the record for CSV: per-core list spread across columns,
            # run config appended. `ts` is absolute wall clock on both sides, so
            # cloud and edge CSVs join on it directly.
            flat = dict(record, elapsed_s=round(now - t_start, 1),
                        weights=runtime_info["weights"],
                        backend=runtime_info["backend"],
                        imgsz=runtime_info["imgsz"],
                        int8=runtime_info["int8"],
                        threads=runtime_info["infer_threads"],
                        workers=runtime_info["workers"])
            flat.update({f"cpu_core{i}": v for i, v in enumerate(record["cpu_per_core"])})
            writer.writerow([flat.get(c) for c in header])
            # Flushed every row - this thread is a daemon, so Ctrl-C kills the
            # process without unwinding and anything still buffered is lost.
            csv_file.flush()

        time.sleep(1) # sample metrics every second

# Video detection end point
@app.route("/detect", methods=["POST"])
def detect():
    global inflight
    stream_id = request.args.get("stream", "0")

    t0 = time.time()
    jpg_bytes = request.data # Get data
    arr = np.frombuffer(jpg_bytes, dtype=np.uint8) # convert to numpy array
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR) # stays gray scale
    decode_ms = (time.time() - t0) * 1000 # log decode latency

    # Each stream has its own worker (its own model), so this call runs in
    # parallel with the other streams instead of queueing behind them.
    worker = worker_for(stream_id)

    with inflight_lock:
        inflight += 1
    try:
        results, inference_ms, queue_wait_ms = worker.track(frame, stream_id)
    finally:
        with inflight_lock:
            inflight -= 1

    boxes = results[0].boxes
    dets = []
    # If there are detections:
    # One line per object - ID: Label: Confidence:
    if boxes.id is not None:
        # xywhn is normalised centre form, already computed - the edge uses it to
        # derive density (how small/crowded the boxes are) and per-track
        # displacement (how fast the scene moves) without running a model itself.
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
        "queue_wait_ms": round(queue_wait_ms, 1),  # time queued behind a stream sharing this worker
        "backend": runtime_info["backend"],        # so the edge CSV records which runtime served it
        # which model instance served this frame, and how many cores it had -
        # lets a sweep be reconstructed from the edge CSV alone
        "worker_id": worker.idx,
        "infer_threads": runtime_info["infer_threads"],
        "served_imgsz": runtime_info["imgsz"],     # resolution this frame really ran at
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
    global imgsz, _infer_threads, _pin_cpus

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
    parser.add_argument("--workers", type=int, default=1,
                        help="number of independent model instances. Set this to the "
                             "number of streams you are about to run: each stream gets "
                             "its own model, so they infer in parallel instead of "
                             "queueing. 1 reproduces the old single-model behaviour")
    parser.add_argument("--infer-threads", "--threads", dest="infer_threads",
                        type=int, default=None,
                        help="CPU threads per worker. Default splits the logical cores "
                             "evenly across --workers (ceil), e.g. 8 cores: 1 worker=8, "
                             "2 workers=4 each, 3 workers=3 each, 4 workers=2 each")
    parser.add_argument("--pin", action="store_true",
                        help="confine each request thread to its worker's cores. Off by "
                             "default: OpenVINO's TBB pool is process-wide, so in-process "
                             "workers cannot get truly private cores and masking tends to "
                             "concentrate them instead of spreading them")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--csv", default=None,
                        help=f"cloud metrics CSV path (default: auto-named under {CSV_DIR}/)")
    parser.add_argument("--no-csv", dest="write_csv", action="store_false",
                        help="don't log cloud metrics to disk (dashboard still works)")
    args = parser.parse_args()

    if args.int8 and args.backend != "openvino":
        parser.error("--int8 applies to --backend openvino")

    if args.workers < 1:
        parser.error("--workers must be >= 1")

    if args.infer_threads is not None and args.infer_threads < 1:
        parser.error("--infer-threads must be >= 1")

    # Split the cores across the workers. ceil, not floor, so 3 workers on 8
    # cores get 3 threads each rather than 2 - the partition the sweep asks for.
    cores = psutil.cpu_count(logical=True) or 1
    threads = args.infer_threads or max(1, math.ceil(cores / args.workers))
    args.infer_threads = threads     # so run_tag() labels the file with what was used
    _infer_threads = threads         # read by _patch_thread_caps at load time
    _pin_cpus = args.pin

    if args.backend == "pytorch":
        # torch's thread count is process-global: one setting for every worker.
        import torch
        torch.set_num_threads(threads)
    # Belt and braces for any OpenMP-linked component. Note this alone does NOT
    # cap OpenVINO (TBB build) - that is handled at compile time in load_model.
    os.environ["OMP_NUM_THREADS"] = str(threads)

    imgsz = args.imgsz
    runtime_info.update({"backend": args.backend, "weights": args.weights,
                         "imgsz": args.imgsz, "int8": args.int8,
                         "torch_threads": threads, "workers": args.workers,
                         "infer_threads": threads})

    # Build the pool. Each worker loads its own instance from the same export on
    # disk, and is warmed so no stream pays lazy-init cost on its first frame.
    for i in range(args.workers):
        cores_for_worker = core_slice(i, threads, cores)
        worker = Worker(i, args.weights, args.backend, args.imgsz, args.int8,
                        args.data, cores_for_worker)
        worker.warmup(args.imgsz)
        pool.append(worker)
        print(f"[server] worker {i} ready ({threads} threads on cores {cores_for_worker})")

    # Confirm the artefact really runs at the requested resolution, so a bad
    # cache shows up here instead of invalidating a whole imgsz sweep silently.
    got = effective_imgsz(pool[0].model)
    if got and got[0] != args.imgsz:
        print(f"[server] WARNING: asked for imgsz={args.imgsz}, loaded model runs at "
              f"{got[0]}x{got[1]}. Delete {export_path(args.weights, args.backend, args.imgsz, args.int8)} "
              f"and restart to re-export.")
    elif got:
        print(f"[server] confirmed inference resolution {got[0]}x{got[1]}")

    csv_path = None
    if args.write_csv:
        csv_path = args.csv or os.path.join(CSV_DIR, f"cloud_{run_tag(args)}.csv")
        parent = os.path.dirname(csv_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    threading.Thread(target=sample_cloud_metrics, args=(csv_path,), daemon=True).start()

    print(f"[server] backend={args.backend} weights={args.weights} "
          f"imgsz={args.imgsz} int8={args.int8} "
          f"workers={args.workers} threads/worker={threads} (of {cores} cores) "
          f"pinning={'on' if args.pin else 'off'}")
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
