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

from line_counter import (LineCounter, parse_line, line_for, VIDEO_LINES,
                          DEFAULT_COOLDOWN)

app = Flask(__name__)

MODEL_WEIGHTS = "yolo11n.pt"
HISTORY_LEN = 300          # samples retained per series for the dashboard
STREAM_TIMEOUT_S = 15      # no traffic for this long => stream counted as finished

# Elastic pool dormancy thresholds. A stream that stops calling /detect for
# `IDLE_TO_DORMANT_S` keeps its worker but is marked dormant - it stops counting
# toward the CPU split, so the active streams widen. If it stays dormant for
# `IDLE_TO_RELEASE_S`, the worker is actually released. The middle state exists
# because on wake the worker is already loaded and warm, so the returning stream
# gets an inference immediately instead of paying a fork + warmup cost.
IDLE_TO_DORMANT_S = 5
IDLE_TO_RELEASE_S = 300

# A /detect whose worker is reprovisioned mid-flight is retried this many times
# against the replacement before the frame is failed.
REPROVISION_RETRIES = 2

CSV_DIR = "results"        # cloud metrics CSVs land here, one per run

# --- Line counting ---
DEFAULT_COUNT_MIN_AGE = 2

CLOUD_CSV_PREFIX = ["ts", "elapsed_s", "cpu_percent"]
CLOUD_CSV_SUFFIX = [
    "mem_percent", "mem_used_mb", "net_sent_mb", "net_recv_mb",
    "proc_cpu", "proc_mem_mb", "load_avg",
    "inflight_requests", "active_streams",
    "crossings_total",
    "backend", "weights", "imgsz", "int8", "threads", "workers",
]

BACKENDS = ("pytorch", "openvino", "onnx")
DEFAULT_IMGSZ = 640


@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


edge_metrics_history = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
cloud_metrics_history = deque(maxlen=HISTORY_LEN)
stream_info = {}
lock = threading.Lock()

# --- --- Detector pool --- ---
# Fixed pool used when --elastic is off.
pool = []
stream_worker = {}
assign_lock = threading.Lock()

# Elastic pool used when --elastic is on.
elastic_pool = None

# Pre-compiled model variants keyed by threads-per-worker. Built once at boot.
VARIANTS = {}

# --- --- Line counters --- ---
count_default = None
count_min_age = DEFAULT_COUNT_MIN_AGE
count_enabled = True
stream_counters = {}
no_line_streams = set()
counter_last_frame = {}
counter_last_seen = {}
counter_lock = threading.Lock()

CONF = 0.3
VEHICLE_CLASSES = [2, 3, 5, 7]

TRACKER_CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "botsort_custom.yaml")
if not os.path.exists(TRACKER_CFG):
    print(f"[server] {os.path.basename(TRACKER_CFG)} not found; using stock botsort.yaml")
    TRACKER_CFG = "botsort.yaml"

imgsz = DEFAULT_IMGSZ

runtime_info = {"backend": "pytorch", "weights": MODEL_WEIGHTS,
                "imgsz": DEFAULT_IMGSZ, "effective_imgsz": None,
                "int8": False, "torch_threads": None,
                "workers": 1, "infer_threads": None}

inflight = 0
inflight_lock = threading.Lock()

_infer_threads = None
_pin_cpus = True
_patched = False

CAN_PIN = hasattr(os, "sched_setaffinity")

USE_ELASTIC = False


def core_slice(idx, threads, cores):
    """Even-split core slice used by the fixed pool."""
    start = (idx * threads) % cores
    return sorted({(start + j) % cores for j in range(min(threads, cores))})


def _patch_thread_caps(backend):
    """Make the per-worker thread cap actually reach the inference runtime."""
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
    stem = os.path.splitext(weights)[0]
    if backend == "openvino":
        return f"{stem}_int8_openvino_model" if int8 else f"{stem}_openvino_model"
    return f"{stem}.onnx"


def export_path(weights, backend, size, int8=False):
    base = ultralytics_export_path(weights, backend, int8)
    if backend == "openvino":
        return f"{base}_imgsz{size}"
    return f"{os.path.splitext(base)[0]}_imgsz{size}.onnx"


def load_model(weights, backend, size, int8=False, data=None):
    if backend == "pytorch":
        return YOLO(weights)

    _patch_thread_caps(backend)

    target = export_path(weights, backend, size, int8)
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

    Elastic mode uses three states:
    - active: serving a stream, counted in the CPU split.
    - dormant: assigned to a stream that has gone quiet. Stays loaded and warm
      so a returning stream is served immediately, but not counted in the
      split - so other active workers widen their share.
    - shut down: mid-swap during reprovisioning, or released. Rejects requests.
    """

    def __init__(self, idx, weights, backend, size, int8=False, data=None, cores=None):
        self.idx = idx
        self.lock = threading.Lock()
        self.streams = set()
        self.trackers = {}
        self.pristine = None
        self.cores = cores
        self.threads = _infer_threads
        self.dormant = False
        self.last_inference = time.time()
        self.gen = 0
        self._pinned = threading.local()
        self._shutdown = False

        self.model = load_model(weights, backend, size, int8, data)

    def _apply_affinity(self):
        if self.cores and _pin_cpus and CAN_PIN:
            os.sched_setaffinity(0, self.cores)

    def _release_affinity(self):
        if _pin_cpus and CAN_PIN:
            os.sched_setaffinity(0, range(psutil.cpu_count(logical=True) or 1))

    def _pin_this_thread(self):
        if not getattr(self._pinned, "done", False):
            self._apply_affinity()
            self._pinned.done = True

    def warmup(self, size):
        self.model.track(np.zeros((540, 960, 3), dtype=np.uint8), persist=True,
                         imgsz=size, verbose=False, tracker=TRACKER_CFG)
        predictor = getattr(self.model, "predictor", None)
        if predictor is not None:
            if self.idx == 0 and predictor.trackers:
                a = getattr(predictor.trackers[0], "args", None)
                if a is not None:
                    print(f"[server] tracker {getattr(a, 'tracker_type', '?')}: "
                          f"match_thresh={getattr(a, 'match_thresh', '?')} "
                          f"(IoU floor {1 - getattr(a, 'match_thresh', 0):.2f}) "
                          f"new_track_thresh={getattr(a, 'new_track_thresh', '?')} "
                          f"track_buffer={getattr(a, 'track_buffer', '?')}")
            for t in predictor.trackers:
                t.reset()
            try:
                self.pristine = copy.deepcopy(predictor.trackers)
            except Exception as e:
                print(f"[server] worker {self.idx}: tracker snapshot failed ({e}); "
                      f"isolation degrades if this worker serves >1 stream")
        self.trackers.clear()

    def _swap_in(self, stream_id, predictor):
        saved = self.trackers.get(stream_id)
        if saved is None:
            if self.pristine is not None:
                saved = copy.deepcopy(self.pristine)
            else:
                saved = predictor.trackers
                for t in saved:
                    t.reset()
        predictor.trackers = saved

    def track(self, frame, stream_id):
        self._pin_this_thread()
        t_wait = time.time()
        with self.lock:
            if self._shutdown:
                raise RuntimeError("worker shut down")
            t_infer = time.time()
            shared = len(self.streams) > 1
            predictor = getattr(self.model, "predictor", None)
            if shared and predictor is not None:
                self._swap_in(stream_id, predictor)

            results = self.model.track(frame, persist=True, conf=CONF, verbose=False,
                                       imgsz=imgsz,
                                       classes=VEHICLE_CLASSES,
                                       tracker=TRACKER_CFG)
            inference_ms = (time.time() - t_infer) * 1000

            if shared:
                predictor = predictor or getattr(self.model, "predictor", None)
                if predictor is not None:
                    self.trackers[stream_id] = predictor.trackers

        return results, inference_ms, (t_infer - t_wait) * 1000

    def shutdown(self):
        self._shutdown = True
        self.trackers.clear()
        self.streams.clear()

    def drain_and_shutdown(self):
        with self.lock:
            self.shutdown()

    @classmethod
    def from_variant(cls, idx, template, threads, cores):
        w = cls.__new__(cls)
        w.idx = idx
        w.lock = threading.Lock()
        w.streams = set()
        w.trackers = {}
        w.pristine = copy.deepcopy(getattr(template.predictor, 'trackers', None))
        w.cores = cores
        w.threads = threads
        w.dormant = False
        w.last_inference = time.time()
        w.gen = 0
        w._pinned = threading.local()
        w._shutdown = False
        w.model = template
        return w


class ElasticPool:
    """Grows, shrinks, and idles workers with active inference demand.

    Streams have three states, mirroring their worker:
    - Active: recently called /detect. Counted in the CPU split.
    - Dormant: idle for `IDLE_TO_DORMANT_S`. Worker stays loaded but is
      excluded from the split, so active workers widen their share.
    - Released: dormant for `IDLE_TO_RELEASE_S`, or explicitly disconnected.
      Worker is destroyed.

    Reprovisioning is a pointer swap into VARIANTS + warmup, not a recompile,
    so it's cheap (~tens of ms).
    """

    def __init__(self, cores):
        self.cores = cores
        self.workers = {}
        self.lock = threading.Lock()

    def _active_ids(self):
        """Streams whose workers are counted in the current CPU split."""
        return [sid for sid, w in self.workers.items() if not w.dormant]

    def _rebalance(self):
        """Resize each active worker to its share under the current active count.
        Dormant workers keep their old sizing but don't count against anyone.
        """
        active_ids = self._active_ids()
        n = len(active_ids)
        if n == 0:
            return
        for new_idx, sid in enumerate(active_ids):
            w = self.workers[sid]
            new_t = threads_for_worker(new_idx, n, self.cores)
            if w.threads != new_t:
                self._reprovision(sid, new_idx, n)
            elif w.idx != new_idx:
                # Same thread count, different position in the active list.
                # No model swap needed - just record the new placement.
                w.idx = new_idx
                w.cores = core_slice_uneven(new_idx, n, self.cores)

    def _reprovision(self, sid, idx, n_workers):
        """Swap the worker at `sid` for a fresh one sized for `n_workers`.
        Carries over per-stream tracker state so IDs don't restart mid-run.
        """
        old = self.workers[sid]
        new_threads = threads_for_worker(idx, n_workers, self.cores)
        cores_for_worker = core_slice_uneven(idx, n_workers, self.cores)
        new = Worker.from_variant(idx, VARIANTS[new_threads],
                                  new_threads, cores_for_worker)
        new.warmup(imgsz)
        # Carry over live state so the stream keeps its track IDs.
        new.streams = set(old.streams)
        new.trackers = dict(old.trackers)
        new.dormant = old.dormant
        new.last_inference = old.last_inference
        new.gen = old.gen + 1
        old.drain_and_shutdown()
        self.workers[sid] = new
        print(f"[server] stream {sid}: reprovisioned to {new_threads}t on {cores_for_worker}"
              f"{' (dormant)' if new.dormant else ''}")

    def ensure_capacity(self, stream_id):
        """Return the worker for `stream_id`. Wake it if dormant, create it if
        this is the first sight of the stream.
        """
        with self.lock:
            w = self.workers.get(stream_id)
            if w is not None:
                if w.dormant:
                    # Wake path: flip flag, then rebalance so active peers
                    # shrink to make room for us.
                    w.dormant = False
                    w.last_inference = time.time()
                    print(f"[server] stream {stream_id}: waking dormant worker")
                    self._rebalance()
                return w

            # First sight of this stream. Shrink existing active workers,
            # then add the new one at the end of the active list.
            active_before = self._active_ids()
            n_active_after = len(active_before) + 1

            for new_idx, sid in enumerate(active_before):
                other = self.workers[sid]
                new_t = threads_for_worker(new_idx, n_active_after, self.cores)
                if other.threads != new_t:
                    self._reprovision(sid, new_idx, n_active_after)
                elif other.idx != new_idx:
                    other.idx = new_idx
                    other.cores = core_slice_uneven(new_idx, n_active_after, self.cores)

            new_idx = n_active_after - 1
            new_threads = threads_for_worker(new_idx, n_active_after, self.cores)
            cores_for_worker = core_slice_uneven(new_idx, n_active_after, self.cores)
            new_worker = Worker.from_variant(new_idx, VARIANTS[new_threads],
                                             new_threads, cores_for_worker)
            new_worker.warmup(imgsz)
            new_worker.streams.add(stream_id)
            new_worker.last_inference = time.time()
            self.workers[stream_id] = new_worker
            print(f"[server] stream {stream_id}: new worker {new_idx}, "
                  f"{new_threads}t on {cores_for_worker}")
            return new_worker

    def make_dormant(self, stream_id):
        """Stream went quiet. Keep its worker loaded but stop counting it,
        then widen the active peers."""
        with self.lock:
            w = self.workers.get(stream_id)
            if w is None or w.dormant:
                return
            w.dormant = True
            print(f"[server] stream {stream_id}: worker marked dormant")
            self._rebalance()

    def release(self, stream_id):
        """Stream is gone for good. Drop its worker and re-widen the rest."""
        with self.lock:
            w = self.workers.pop(stream_id, None)
            if w is None:
                return
            w.drain_and_shutdown()
            print(f"[server] stream {stream_id}: worker released")
            self._rebalance()

    def touch(self, stream_id):
        """Update last_inference for the reaper. No lock: torn reads in the
        reaper only cause dormant to fire one cycle late."""
        w = self.workers.get(stream_id)
        if w is not None:
            w.last_inference = time.time()


def worker_for(stream_id):
    """Route a stream to its worker in the fixed pool, assigning on first sight."""
    with assign_lock:
        worker = stream_worker.get(stream_id)
        if worker is None:
            worker = pool[len(stream_worker) % len(pool)]
            stream_worker[stream_id] = worker
            worker.streams.add(stream_id)
            note = "  (sharing - more streams than workers)" if len(worker.streams) > 1 else ""
            print(f"[server] stream {stream_id} -> worker {worker.idx}{note}")
        return worker


def update_crossings(stream_id, video, frame_num, dets):
    if not count_enabled:
        return None

    with counter_lock:
        now = time.time()
        last_frame = counter_last_frame.get(stream_id)
        last_seen = counter_last_seen.get(stream_id)
        restarted = ((last_frame is not None and frame_num < last_frame)
                     or (last_seen is not None and now - last_seen > STREAM_TIMEOUT_S))
        if restarted:
            done = stream_counters.pop(stream_id, None)
            if done is not None:
                print(f"[server] stream {stream_id} restarted - previous run counted "
                      f"{done.total} crossing(s) over {last_frame} frames")
            no_line_streams.discard(stream_id)
        counter_last_frame[stream_id] = frame_num
        counter_last_seen[stream_id] = now

        counter = stream_counters.get(stream_id)
        if counter is None:
            if stream_id in no_line_streams:
                return None
            line = line_for(video) or count_default
            if line is None:
                no_line_streams.add(stream_id)
                print(f"[server] stream {stream_id}: no counting line for "
                      f"{video or 'unknown video'} - crossings not counted. Add one "
                      f"to VIDEO_LINES in line_counter.py, or pass --count-line")
                return None
            counter = LineCounter(tuple(line), count_min_age, None, DEFAULT_COOLDOWN,
                                  once_per_track=True)
            stream_counters[stream_id] = counter
            print(f"[server] stream {stream_id} counting line {tuple(line)} "
                  f"min_age={counter.min_age} cooldown={counter.cooldown} "
                  f"(from {video or 'default'})")

        counter.update(frame_num, dets)
        return {"in": counter.counts["in"], "out": counter.counts["out"],
                "total": counter.total, "unique": len(counter.unique)}


def crossings_snapshot():
    with counter_lock:
        per_stream = {sid: {"in": c.counts["in"], "out": c.counts["out"],
                            "total": c.total, "unique": len(c.unique),
                            "line": list(c.line)}
                      for sid, c in stream_counters.items()}
    return per_stream, sum(v["total"] for v in per_stream.values())


def pool_snapshot():
    """Per-worker view of the pool for the dashboard: which streams sit on
    which worker, whether it is active or dormant, and the cores it holds.

    In elastic mode the pool is keyed by stream, so a worker owns exactly one
    stream. In fixed mode streams round-robin onto a static pool, so a worker
    can own several. Both shapes come back the same way.
    """
    now = time.time()
    cores_total = psutil.cpu_count(logical=True) or 1

    if USE_ELASTIC and elastic_pool is not None:
        with elastic_pool.lock:
            snap = [(sid, w.idx, sorted(w.streams), w.dormant, w.threads,
                     list(w.cores or []), w.last_inference, w.gen)
                    for sid, w in elastic_pool.workers.items()]
        workers = [{
            "owner": sid,
            "slot": idx,
            "gen": gen,
            "streams": streams,
            "state": "dormant" if dormant else "active",
            "threads": threads,
            # A dormant worker keeps the core slice it had when it went quiet.
            # It is doing no inference, so that slice is stale - the active
            # workers have already been widened over it.
            "cores": cores,
            "cores_stale": dormant,
            "idle_s": round(now - last, 1),
        } for sid, idx, streams, dormant, threads, cores, last, gen in snap]
        # Active first, then by slot, so the layout does not jump around.
        workers.sort(key=lambda w: (w["state"] != "active", w["slot"], str(w["owner"])))
    else:
        with assign_lock:
            snap = [(w.idx, sorted(w.streams), w.threads, list(w.cores or []))
                    for w in pool]
        workers = [{"owner": None, "slot": idx, "gen": 0, "streams": streams,
                    "state": "active", "threads": threads, "cores": cores,
                    "cores_stale": False, "idle_s": None}
                   for idx, streams, threads, cores in snap]

    active = [w for w in workers if w["state"] == "active"]
    return {
        "mode": "elastic" if USE_ELASTIC else "fixed",
        "cores_total": cores_total,
        "cores_assigned": sum(w["threads"] or 0 for w in active),
        "active_workers": len(active),
        "dormant_workers": len(workers) - len(active),
        "idle_to_dormant_s": IDLE_TO_DORMANT_S,
        "idle_to_release_s": IDLE_TO_RELEASE_S,
        "pinning": bool(_pin_cpus and CAN_PIN),
        "workers": workers,
    }


def active_streams(now=None):
    now = now or time.time()
    return [sid for sid, info in stream_info.items()
            if now - info["last_seen"] <= STREAM_TIMEOUT_S]


def run_tag(args):
    stem = os.path.splitext(os.path.basename(args.weights))[0]
    parts = [time.strftime("%Y%m%d-%H%M%S"), args.backend, stem, f"imgsz{args.imgsz}"]
    if args.int8:
        parts.append("int8")
    parts.append(f"w{args.workers}")
    if args.infer_threads:
        parts.append(f"t{args.infer_threads}")
    if args.elastic:
        parts.append("elastic")
    return "_".join(parts)


def sample_cloud_metrics(csv_path=None):
    proc = psutil.Process()

    header = writer = csv_file = None
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
        with lock:
            record["active_streams"] = len(active_streams(now))
            cloud_metrics_history.append(record)

        if writer:
            flat = dict(record, elapsed_s=round(now - t_start, 1),
                        weights=runtime_info["weights"],
                        backend=runtime_info["backend"],
                        imgsz=runtime_info["imgsz"],
                        int8=runtime_info["int8"],
                        threads=runtime_info["infer_threads"],
                        workers=runtime_info["workers"])
            flat.update({f"cpu_core{i}": v for i, v in enumerate(record["cpu_per_core"])})
            writer.writerow([flat.get(c) for c in header])
            csv_file.flush()

        time.sleep(1)


def stream_reaper():
    """Two-tier worker lifecycle: quiet -> dormant -> released.

    Dormant keeps the compiled model loaded so a returning stream is served
    immediately. Released reclaims the memory once the stream is judged gone.
    """
    while True:
        time.sleep(2)
        if elastic_pool is None:
            continue
        now = time.time()
        # Snapshot the workers list under the lock so we can iterate without
        # holding it across make_dormant / release (which each take it).
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
    global inflight
    stream_id = request.args.get("stream", "0")
    req_frame = request.args.get("frame", type=int)
    video = request.args.get("video")

    t0 = time.time()
    jpg_bytes = request.data
    arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    decode_ms = (time.time() - t0) * 1000

    # The pool can reprovision our worker between handing it to us and our turn
    # at its lock - a peer stream starting, stopping, or going dormant rebalances
    # the CPU split and swaps this worker out from under us. The replacement
    # carries our tracker state, so the fix is just to ask for the current worker
    # and go again rather than failing the frame.
    for attempt in range(REPROVISION_RETRIES + 1):
        if USE_ELASTIC:
            worker = elastic_pool.ensure_capacity(stream_id)
        else:
            worker = worker_for(stream_id)

        with inflight_lock:
            inflight += 1
        try:
            results, inference_ms, queue_wait_ms = worker.track(frame, stream_id)
            break
        except RuntimeError:
            if not USE_ELASTIC or attempt == REPROVISION_RETRIES:
                raise
            print(f"[server] stream {stream_id}: worker swapped mid-request, retrying")
        finally:
            with inflight_lock:
                inflight -= 1

    # Mark this stream still-inferring for the reaper. AFTER the request so
    # a stream that spent 10s queued doesn't look artificially fresh.
    if USE_ELASTIC:
        elastic_pool.touch(stream_id)

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

    with lock:
        info = stream_info.setdefault(stream_id, {"frames": 0, "video": None, "last_seen": 0})
        info["frames"] += 1
        info["last_seen"] = time.time()
        if video and not info["video"]:
            info["video"] = video
        served = info["frames"]

    crossings = update_crossings(stream_id, video, req_frame if req_frame is not None
                                 else served, dets)

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
        # In elastic mode a worker's thread count changes over the run, so
        # report the CURRENT value rather than the boot-time cap.
        "infer_threads": worker.threads if USE_ELASTIC else runtime_info["infer_threads"],
        "served_imgsz": runtime_info["effective_imgsz"] or runtime_info["imgsz"],
        "served_weights": runtime_info["weights"],
        "served_conf": CONF,
        "served_int8": runtime_info["int8"],
    })


@app.route("/disconnect", methods=["POST"])
def disconnect():
    """Explicit teardown for a stream. Skips the reap timeout entirely."""
    stream_id = request.args.get("stream", "0")
    if USE_ELASTIC and elastic_pool is not None:
        elastic_pool.release(stream_id)
    return jsonify({"status": "released", "stream": stream_id})


@app.route("/metrics", methods=["POST"])
def receive_metrics():
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
    crossings, crossings_total = crossings_snapshot()
    pool_state = pool_snapshot()

    with lock:
        now = time.time()
        live = set(active_streams(now))
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


def build_variants(args, cores):
    """Pre-compile one model per unique thread count the elastic pool might use."""
    thread_options = set()
    for n in range(1, cores + 1):
        base = cores // n
        extra = cores % n
        thread_options.add(base)
        if extra:
            thread_options.add(base + 1)
    thread_options = sorted(thread_options)

    global _infer_threads
    # _infer_threads is what the compile-time patch reads, so it has to be set
    # per variant. It is also what the fixed pool builds its workers with, and
    # the fixed pool is created after this runs - so leaving it on the last
    # variant would silently compile every fixed worker for the widest thread
    # count instead of its own.
    caller_threads = _infer_threads
    try:
        for t in thread_options:
            _infer_threads = t
            m = load_model(args.weights, args.backend, args.imgsz, args.int8, args.data)
            m.track(np.zeros((540, 960, 3), dtype=np.uint8), persist=True,
                    imgsz=args.imgsz, verbose=False, tracker=TRACKER_CFG)
            VARIANTS[t] = m
            print(f"[server] pre-compiled variant threads={t}")
    finally:
        _infer_threads = caller_threads


def threads_for_worker(idx, n_workers, cores):
    """Divide cores as evenly as possible. Extras go to the low-idx workers."""
    base = cores // n_workers
    extra = cores % n_workers
    return base + (1 if idx < extra else 0)


def core_slice_uneven(idx, n_workers, cores):
    """Contiguous cores for worker `idx` under the uneven split above."""
    layout = [threads_for_worker(i, n_workers, cores) for i in range(n_workers)]
    start = sum(layout[:idx])
    return list(range(start, start + layout[idx]))


def main():
    global imgsz, _infer_threads, _pin_cpus
    global count_default, count_min_age, count_enabled
    global elastic_pool, USE_ELASTIC
    global IDLE_TO_DORMANT_S, IDLE_TO_RELEASE_S

    parser = argparse.ArgumentParser(description="Cloud inference service.")
    parser.add_argument("--backend", choices=BACKENDS, default="pytorch")
    parser.add_argument("--weights", default=MODEL_WEIGHTS)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--data", default=None)
    parser.add_argument("--workers", type=int, default=1,
                        help="fixed pool only; ignored under --elastic")
    parser.add_argument("--infer-threads", "--threads", dest="infer_threads",
                        type=int, default=None,
                        help="fixed pool only; ignored under --elastic")
    parser.add_argument("--pin", action="store_true")
    parser.add_argument("--elastic", action="store_true",
                        help="grow, shrink, and idle the worker pool with actual "
                             "inference demand. A stream that stops calling /detect "
                             f"for {IDLE_TO_DORMANT_S}s goes dormant - its worker "
                             "stays loaded and warm but stops counting toward the "
                             "CPU split, so active streams widen. Wake on arrival "
                             "is instant (no recompile). Full release after "
                             f"{IDLE_TO_RELEASE_S}s dormant.")
    parser.add_argument("--idle-dormant", type=float, default=IDLE_TO_DORMANT_S,
                        metavar="SECONDS",
                        help=f"seconds without a /detect before a stream's worker "
                             f"goes dormant (default {IDLE_TO_DORMANT_S}). Lower it "
                             f"to watch scaling react inside a short test run")
    parser.add_argument("--idle-release", type=float, default=IDLE_TO_RELEASE_S,
                        metavar="SECONDS",
                        help=f"seconds dormant before the worker is destroyed "
                             f"(default {IDLE_TO_RELEASE_S}). The default is long "
                             f"enough that a test run never reaches it - drop it to "
                             f"~20 to observe release as well as dormancy")
    parser.add_argument("--count-line", default=None, metavar="X1,Y1,X2,Y2")
    parser.add_argument("--count-min-age", type=int, default=DEFAULT_COUNT_MIN_AGE)
    parser.add_argument("--no-count", dest="count", action="store_false")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--no-csv", dest="write_csv", action="store_false")
    args = parser.parse_args()

    if args.int8 and args.backend != "openvino":
        parser.error("--int8 applies to --backend openvino")

    if args.workers < 1:
        parser.error("--workers must be >= 1")

    if args.idle_dormant <= 0 or args.idle_release <= 0:
        parser.error("--idle-dormant and --idle-release must be > 0")
    if args.idle_release < args.idle_dormant:
        parser.error("--idle-release must be >= --idle-dormant")
    IDLE_TO_DORMANT_S = args.idle_dormant
    IDLE_TO_RELEASE_S = args.idle_release

    count_enabled = args.count
    count_min_age = args.count_min_age
    if count_enabled:
        if args.count_line:
            try:
                count_default = parse_line(args.count_line)
            except ValueError as exc:
                parser.error(str(exc))
        configured = sorted(v for v, line in VIDEO_LINES.items() if line)
        if configured:
            print(f"[server] counting lines set for: {', '.join(configured)}")
        elif count_default is None:
            print("[server] no counting lines set. Fill in VIDEO_LINES in "
                  "line_counter.py, or pass --count-line. Crossings will not be counted.")

    if args.infer_threads is not None and args.infer_threads < 1:
        parser.error("--infer-threads must be >= 1")

    cores = psutil.cpu_count(logical=True) or 1
    threads = args.infer_threads or max(1, math.ceil(cores / args.workers))
    args.infer_threads = threads
    _infer_threads = threads
    _pin_cpus = args.pin
    USE_ELASTIC = args.elastic

    if args.backend == "pytorch":
        import torch
        torch.set_num_threads(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)

    imgsz = args.imgsz
    runtime_info.update({"backend": args.backend, "weights": args.weights,
                         "imgsz": args.imgsz, "int8": args.int8,
                         "torch_threads": threads, "workers": args.workers,
                         "infer_threads": threads})

    build_variants(args, cores)

    if USE_ELASTIC:
        elastic_pool = ElasticPool(cores)
        print(f"[server] elastic mode: dormant after {IDLE_TO_DORMANT_S:g}s idle, "
              f"released after {IDLE_TO_RELEASE_S:g}s dormant")
    else:
        for i in range(args.workers):
            cores_for_worker = core_slice(i, threads, cores)
            worker = Worker(i, args.weights, args.backend, args.imgsz, args.int8,
                            args.data, cores_for_worker)
            worker.warmup(args.imgsz)
            pool.append(worker)
            print(f"[server] worker {i} ready ({threads} threads on cores {cores_for_worker})")

    sample_model = pool[0].model if pool else next(iter(VARIANTS.values()))
    got = effective_imgsz(sample_model)
    runtime_info["effective_imgsz"] = got[0] if got else None
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
    if USE_ELASTIC:
        threading.Thread(target=stream_reaper, daemon=True).start()

    mode = "elastic" if USE_ELASTIC else "fixed"
    print(f"[server] backend={args.backend} weights={args.weights} "
          f"imgsz={args.imgsz} int8={args.int8} mode={mode} "
          f"workers={args.workers} threads/worker={threads} (of {cores} cores) "
          f"pinning={'on' if args.pin else 'off'}")
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()