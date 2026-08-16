import argparse
import csv
import math
import threading
import time
from collections import deque

import cv2
import psutil
import requests

# Cloud URL
CLOUD_HOST = "http://10.0.0.2:8000"

# Content shift detection parameters
DIFF_HISTORY = 90        # frames of frame-diff history to keep for a rolling mean
DIFF_WARMUP = 30         # frames to wait before checking for content shift
DIFF_K = 3.0             # how many standard deviations away from mean single a content shift

# Accumulated-change budget parameters ('budget' gate).
# The adaptive gate above triggers on frame_diff being statistically unusual, so
# in a scene with constant motion nothing is ever unusual and new objects are
# missed. The budget gate instead sums frame_diff since the last inference and
# fires when that total crosses BUDGET - it responds to how much the scene has
# changed, not to how surprising that change is. No baseline, no rolling stats,
# nothing to recalibrate when the scene changes.
DEFAULT_DIFF_BUDGET = 30.0   # summed frame_diff that must accrue before inferring
                             # (set per-video from offline profiling of frame_diff)
DEFAULT_MAX_SKIP = 30        # safety floor: never skip more than this many
                             # consecutive frames, whatever the budget says

PUSH_EVERY = 15          # metric push cadence

# Real-time pacing: a file decodes as fast as the CPU allows, so set a sleep to pace to the videos fps.
FALLBACK_FPS = 30.0      # used when the container reports no usable fps

# For comparison of methods - `none` infer on every frame, `fixed` infer every N
# frames, `adaptive` on content-shift changes, `budget` once accumulated
# change since the last inference exceeds a budget, and `motion` once tracked
# objects are predicted to have drifted too far
GATE_MODES = ("none", "fixed", "adaptive", "budget", "motion")
DEFAULT_FRAME_GAP = 5

# --- 'motion' gate ---
# Same shape as 'budget', but what accumulates is predicted object displacement
# instead of pixel change. frame_diff answers "did the image change", which
# lighting, compression noise and wind all trip; track displacement answers "did
# the things we care about move", and its threshold is a staleness bound that
# can be stated: infer before any box has drifted more than this fraction of the
# frame width. Requires box centres, so it only works against a cloud that
# returns them.
DEFAULT_MOTION_BUDGET = 0.04   # normalised frame widths of drift to allow
MOTION_ALPHA = 0.2             # EWMA weight on the measured displacement rate

# --- density (spatial) signals ---
# Derived on the edge from the boxes the cloud returns. Nothing consumes these
# yet - they are logged so the trigger and thresholds can be chosen offline.
CLOUD_CONF_FLOOR = 0.3    # must match CONF in server_side.py
LOW_CONF_MARGIN = 0.1     # a detection within this of the floor counts as unsure
SMALL_BOX_AREA = 0.002    # normalised area below which a box is "small" - roughly
                          # where 640px inference starts losing distant vehicles
OVERLAP_IOU = 0.1         # IoU above which two boxes count as a crowded pair

DENSITY_ALPHA = 0.1              # EWMA weight, updated per inference not per frame
DENSITY_HYSTERESIS_FRAC = 0.15   # boundary widening, as a fraction of the lo-hi band
DENSITY_DWELL = 30               # frames a state must hold before it may change

# Candidate triggers for "this scene needs more pixels". All are oriented so
# higher = harder; which one actually predicts it is what the sweep answers.
DENSITY_METRICS = {
    "small_boxes": "small_boxes",
    "box_count": "box_count",
    "low_conf": "low_conf_frac",
    "overlap": "overlap_pairs",
}
DEFAULT_DENSITY_METRIC = "small_boxes"
DEFAULT_DENSITY_LO = 2.0
DEFAULT_DENSITY_HI = 6.0

# Wire format for frames sent to the cloud. `png` is lossless - the golden run's
# upper bound on both accuracy and payload. `jpeg` is the lossy baseline, swept
# across --width to trade bandwidth against detection quality.
ENCODINGS = ("png", "jpeg")
DEFAULT_JPEG_QUALITY = 80

# Save results for post-run analysis
CSV_HEADER = [
    "stream_id", "frame", "ts", "storage_io_ms", "preprocess_ms", "round_trip_ms",
    "decode_ms", "inference_ms", "queue_wait_ms", "network_ms", "end_to_end_ms",
    "throughput_fps", "objects_in_frame", "unique_total",
    "edge_cpu", "edge_mem", "payload_kb", "bandwidth_mbps",
    "frame_diff", "content_shift_detected",
    # summed frame_diff since the last inference, as it stood when this frame was
    # gated (so on an inferred row it is the change that paid for the inference).
    # Profile this column offline to pick --budget.
    "change_since_infer",
    "ttff_ms",
    # how far behind the frame's scheduled arrival time we finished it - the
    # real "can the edge keep up?" signal once the loop is paced
    "pacing_lag_ms",
    # --- frame gating ---
    "gate_mode", "inference_ran", "filter_rate",
    # cloud runtime that served this run, so backend comparisons are self-labelling
    "backend",
    # --- cloud concurrency config, reported per response ---
    # which model instance served this stream, and how many CPU threads it had.
    # Lets a stream-count sweep be reconstructed from the edge CSV alone, and
    # confirms each stream really did land on its own worker.
    "worker_id", "infer_threads",
    # resolution the cloud actually served this frame at
    "served_imgsz",
    # --- density (spatial) signals: how hard this frame is to resolve ---
    # Recomputed on inference and carried forward on gated frames, so every row
    # describes the detections it actually reports.
    "box_count", "small_boxes", "mean_box_area", "min_box_area",
    "mean_conf", "low_conf_frac", "overlap_pairs",
    "density_metric", "density_ewma", "density_state",
    # --- motion (temporal) signals: how fast the scene is moving ---
    # disp_* are normalised frame widths per frame. disp_accum is the predicted
    # drift since the last inference, as it stood when this frame was gated.
    "disp_rate", "disp_rate_max", "disp_rate_ewma", "tracks_matched", "disp_accum",
]

# Instead of sampling CPU metrics per instance, sampling is done periodically
# psutil.cpu_percent() measures difference since last call, if called too frequently will return 0
host_usage = {"cpu": 0.0, "mem": 0.0}

def sample_host_usage():
    while True:
        host_usage["cpu"] = psutil.cpu_percent()
        host_usage["mem"] = psutil.virtual_memory().percent
        time.sleep(0.5)

# lock terminal print
print_lock = threading.Lock()


def rnd(value, digits):
    """round(), but None passes through - gated frames report null latencies."""
    return None if value is None else round(value, digits)


def box_iou(a, b):
    """IoU of two normalised centre-form boxes."""
    iw = min(a["cx"] + a["w"] / 2, b["cx"] + b["w"] / 2) - max(a["cx"] - a["w"] / 2, b["cx"] - b["w"] / 2)
    ih = min(a["cy"] + a["h"] / 2, b["cy"] + b["h"] / 2) - max(a["cy"] - a["h"] / 2, b["cy"] - b["h"] / 2)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def density_signals(dets):
    """Spatial-difficulty components for one set of detections.

    Areas are normalised, so they compare across resolutions and videos. The
    overlap count is O(n^2) but n is vehicles in one frame.
    """
    boxed = [d for d in dets if "w" in d]
    if not boxed:
        return {"box_count": len(dets), "small_boxes": 0, "mean_box_area": None,
                "min_box_area": None, "mean_conf": None, "low_conf_frac": None,
                "overlap_pairs": 0}

    areas = [d["w"] * d["h"] for d in boxed]
    confs = [d["conf"] for d in boxed]
    overlaps = sum(1 for i in range(len(boxed)) for j in range(i + 1, len(boxed))
                   if box_iou(boxed[i], boxed[j]) > OVERLAP_IOU)
    return {
        "box_count": len(dets),
        "small_boxes": sum(1 for a in areas if a < SMALL_BOX_AREA),
        "mean_box_area": sum(areas) / len(areas),
        "min_box_area": min(areas),
        "mean_conf": sum(confs) / len(confs),
        # detections sitting on the confidence floor: evidence the model is
        # unsure, which box count alone never shows
        "low_conf_frac": sum(1 for c in confs if c <= CLOUD_CONF_FLOOR + LOW_CONF_MARGIN) / len(confs),
        "overlap_pairs": overlaps,
    }


def track_displacement(prev_dets, dets, frames_elapsed):
    """Per-frame centre displacement of tracks present in both frames.

    Matched by track ID, so an object that entered or left contributes nothing -
    there is no displacement to measure for it. Returns
    (mean, max, matched_count) in normalised frame widths per frame, or
    (None, None, 0) if nothing could be matched.
    """
    if not prev_dets or not dets or frames_elapsed <= 0:
        return None, None, 0

    prev_by_id = {d["id"]: d for d in prev_dets if "cx" in d}
    moves = []
    for d in dets:
        prev = prev_by_id.get(d["id"]) if "cx" in d else None
        if prev is not None:
            moves.append(math.hypot(d["cx"] - prev["cx"], d["cy"] - prev["cy"]))
    if not moves:
        return None, None, 0
    return (sum(moves) / len(moves) / frames_elapsed,
            max(moves) / frames_elapsed,
            len(moves))


def classify_density(value, state, lo, hi, margin):
    """Band `value` into low/med/high, biased towards staying put.

    Boundaries widen by `margin` in whichever direction would leave the current
    state, so a value hovering on one doesn't oscillate.
    """
    if state == "low":
        return "med" if value > lo + margin else "low"
    if state == "high":
        return "med" if value < hi - margin else "high"
    if value < lo - margin:
        return "low"
    if value > hi + margin:
        return "high"
    return "med"


def run_stream(stream_id, video_path, host, gate_mode="none", frame_gap=DEFAULT_FRAME_GAP,
               realtime=True, encoding="png", width=None, jpeg_quality=DEFAULT_JPEG_QUALITY,
               diff_budget=DEFAULT_DIFF_BUDGET, max_skip=DEFAULT_MAX_SKIP,
               motion_budget=DEFAULT_MOTION_BUDGET, density_metric=DEFAULT_DENSITY_METRIC,
               density_lo=DEFAULT_DENSITY_LO, density_hi=DEFAULT_DENSITY_HI):
    """Process a single video stream, sending frames to the cloud for inference, and logging metrics."""

    detect_url = f"{host}/detect"
    metrics_url = f"{host}/metrics"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[stream {stream_id}] ERROR: could not open {video_path}")
        return

    # Pace to the video's own fps, so the loop behaves like a camera feed
    # instead of racing through the file at decode speed.
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not (src_fps and src_fps > 0):   # also catches the NaN some containers report
        src_fps = FALLBACK_FPS
        if realtime:
            with print_lock:
                print(f"[stream {stream_id}] no fps in container, pacing at {FALLBACK_FPS:g} fps")
    # 0 disables pacing - the loop then runs at whatever speed decode allows
    frame_interval = (1.0 / src_fps) if realtime else 0.0

    # Resolve --width against the source once, so the resize target is fixed for
    # the run. Height follows the source aspect ratio, and a --width at or above
    # the native width is treated as "send native" - upscaling on the edge would
    # cost bandwidth without adding detail.
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resize_to = None
    if width and src_w and src_h and width < src_w:
        resize_to = (width, round(src_h * width / src_w))
    encode_desc = f"{encoding}" + (f"q{jpeg_quality}" if encoding == "jpeg" else "")
    res_desc = f"{resize_to[0]}x{resize_to[1]}" if resize_to else f"native {src_w}x{src_h}"
    with print_lock:
        print(f"[stream {stream_id}] encoding {encode_desc} at {res_desc}")

    # one connection pool per stream, so streams don't queue behind each other
    session = requests.Session()

    seen_ids = set()
    frame_num = 0
    wall_start = time.time()   # for throughput: frames per real second, and TTFF
    ttff_ms = None             # set once, when the first frame's result comes back

    prev_gray = None
    diff_history = deque(maxlen=DIFF_HISTORY)

    # 'budget' gate state: change accrued since the last inference, and how many
    # frames in a row have been skipped (for the MAX_SKIP safety floor).
    diff_accum = 0.0
    skipped_since_infer = 0
    budget_floor_hits = 0      # inferences forced by MAX_SKIP rather than the budget

    # Cache last detections so they're reused for frames not sent to the cloud
    last_dets = None
    frames_inferred = 0
    frames_skipped = 0
    frames_late = 0            # frames that finished after their scheduled slot
    max_lag_ms = 0.0
    backend = None            # reported by the cloud on each /detect response
    worker_id = None          # cloud model instance serving this stream
    infer_threads = None      # CPU threads that instance was given
    served_imgsz = None       # resolution the cloud ran at

    # Density state. Updated per inference, not per frame - a gated frame brings
    # no new detections, so folding it in would just weight the EWMA by how
    # heavily the stream happens to be gated.
    density = density_signals([])
    density_ewma = None
    density_state = "low"
    density_state_since = 0
    density_margin = DENSITY_HYSTERESIS_FRAC * (density_hi - density_lo)

    # Motion state for the 'motion' gate.
    last_infer_frame = None
    disp_rate = None
    disp_rate_max = None
    disp_rate_ewma = None     # None until two inferred frames share a track ID
    tracks_matched = 0
    disp_accum = 0.0

    csv_file = open(f"edge_metrics_stream{stream_id}.csv", "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(CSV_HEADER)
    csv_file.flush()

    while cap.isOpened():
        # read frame from disk and time it
        io0 = time.time()  
        success, frame = cap.read()
        storage_io_ms = (time.time() - io0) * 1000  
        if not success:
            with print_lock:
                print(f"[stream {stream_id}] Playback complete.")
            break
        frame_num += 1

        # Content shift detection
        shift0 = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_diff = None
        content_shift_detected = False
        if prev_gray is not None:
            frame_diff = float(cv2.absdiff(gray, prev_gray).mean())
            if len(diff_history) >= DIFF_WARMUP:
                baseline_mean = sum(diff_history) / len(diff_history)
                variance = sum((d - baseline_mean) ** 2 for d in diff_history) / len(diff_history)
                baseline_std = variance ** 0.5
                if abs(frame_diff - baseline_mean) > DIFF_K * baseline_std:
                    content_shift_detected = True
            diff_history.append(frame_diff)
        prev_gray = gray
        shift_ms = (time.time() - shift0) * 1000  

        # Accumulated change since the last inference. The first frame has no
        # predecessor, so frame_diff is None and contributes nothing.
        diff_accum += frame_diff or 0.0
        # Snapshot before the gate resets it, so the CSV row for an inferred frame
        # carries the change that actually triggered it.
        change_since_infer = diff_accum

        # Predicted drift since the last inference. Unlike diff_accum this is a
        # forecast, not a measurement - a gated frame has no fresh boxes to
        # measure from, so the last known rate is extrapolated forward.
        disp_accum += disp_rate_ewma or 0.0
        motion_since_infer = disp_accum

        # Decide whether to run inference on this frame based on gating mode defined
        forced_by_floor = False
        if last_dets is None or gate_mode == "none":
            # last_dets is None covers the first frame of the stream (and any frame
            # after a failed request): a stream always starts with an inference.
            run_inference = True
        elif gate_mode == "fixed":
            run_inference = (frame_num % frame_gap == 0)
        elif gate_mode == "budget":
            # Fire when enough change has accrued, or when the safety floor says
            # we've coasted on stale detections for long enough. The floor bounds
            # the worst-case miss independently of how good the diff signal is.
            forced_by_floor = skipped_since_infer >= max_skip
            run_inference = diff_accum >= diff_budget or forced_by_floor
        elif gate_mode == "motion":
            forced_by_floor = skipped_since_infer >= max_skip
            if disp_rate_ewma is None:
                # No rate yet: it takes two inferred frames sharing a track for
                # one to exist, so infer back to back until they do - but only
                # while there is a track to wait for. With nothing detected there
                # is nothing to bootstrap from, so coast on the floor instead of
                # inferring every frame at an empty camera. Costs one floor
                # interval at startup, while the tracker confirms its first
                # tracks and reports nothing.
                run_inference = bool(last_dets) or forced_by_floor
            else:
                run_inference = disp_accum >= motion_budget or forced_by_floor
        else:  # adaptive
            run_inference = content_shift_detected

        if run_inference:
            # Edge preprocessing - resize and encode, image already in gray scale
            prep0 = time.time()
            # Downscale first when asked, so the encoder only works on the pixels
            # that go on the wire.
            to_encode = cv2.resize(frame, resize_to) if resize_to else frame
            if encoding == "jpeg":
                ok, buf = cv2.imencode(".jpg", to_encode,
                                       [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            else:
                ok, buf = cv2.imencode(".png", to_encode)   # lossless, no quality knob
            preprocess_ms = (time.time() - prep0) * 1000  # preprocessing duration
            payload_kb = len(buf) / 1024  # size of compressed image

            # Send to cloud for inference and measure round trip time
            try:
                rt0 = time.time()
                # Post - stream id keeps the cloud's tracker state separate per video
                resp = session.post(detect_url, data=buf.tobytes(),
                                    params={"stream": stream_id},
                                    headers={"Content-Type": "application/octet-stream"},
                                    timeout=15)
                round_trip_ms = (time.time() - rt0) * 1000
                data = resp.json()
                # Get inference times for metrics Post
                dets = data["detections"]
                decode_ms = data.get("decode_ms", 0)
                inference_ms = data.get("inference_ms", 0)
                queue_wait_ms = data.get("queue_wait_ms", 0)
                backend = data.get("backend")
                worker_id = data.get("worker_id")
                infer_threads = data.get("infer_threads")
                served_imgsz = data.get("served_imgsz")
            except requests.RequestException as e:
                with print_lock:
                    print(f"[stream {stream_id}] Request failed: {e}")
                continue

            # Displacement needs the previous inferred frame, so it is measured
            # before last_dets is replaced.
            if last_infer_frame is not None:
                disp_rate, disp_rate_max, tracks_matched = track_displacement(
                    last_dets, dets, frame_num - last_infer_frame)
                if disp_rate is None and not dets and disp_rate_ewma is not None:
                    # Empty scene: nothing on screen can go stale, so the rate is
                    # genuinely zero rather than unmeasurable.
                    # Only once a real rate exists, though - the tracker returns
                    # no detections for its first few frames while it confirms
                    # tracks, and folding those zeros in would pin the EWMA at 0
                    # before it had ever measured anything.
                    disp_rate = disp_rate_max = 0.0
                if disp_rate is not None:
                    disp_rate_ewma = (disp_rate if disp_rate_ewma is None else
                                      MOTION_ALPHA * disp_rate + (1 - MOTION_ALPHA) * disp_rate_ewma)

            density = density_signals(dets)
            metric_value = density[DENSITY_METRICS[density_metric]] or 0.0
            density_ewma = (metric_value if density_ewma is None else
                            DENSITY_ALPHA * metric_value + (1 - DENSITY_ALPHA) * density_ewma)
            # Hysteresis picks the candidate state, dwell decides whether it may
            # take effect yet. Both are needed: hysteresis stops a value hovering
            # on a boundary from oscillating, dwell stops a brief excursion well
            # past one from doing the same.
            candidate = classify_density(density_ewma, density_state,
                                         density_lo, density_hi, density_margin)
            if candidate != density_state and frame_num - density_state_since >= DENSITY_DWELL:
                density_state = candidate
                density_state_since = frame_num

            last_dets = dets
            last_infer_frame = frame_num
            frames_inferred += 1
            # Spend the accumulated change only once the frame has actually been
            # served - a failed request 'continue's above with the total intact.
            diff_accum = 0.0
            disp_accum = 0.0
            skipped_since_infer = 0
            if forced_by_floor:
                budget_floor_hits += 1

            # network time = round trip minus cloud-side decode + inference + lock wait
            network_ms = round_trip_ms - decode_ms - inference_ms - queue_wait_ms

            # end-to-end latency: disk read, encode, cloud response
            end_to_end_ms = storage_io_ms + preprocess_ms + round_trip_ms

            # Mbps - bits / time (s) / 1e6 for megabits
            bandwidth_mbps = (len(buf) * 8 / (network_ms / 1000) / 1e6) if network_ms > 0 else 0

            # time to first frame: only set once, on the first successful result
            if ttff_ms is None:
                ttff_ms = (time.time() - wall_start) * 1000
                with print_lock:
                    print(f">>> [stream {stream_id}] Time to first frame: {ttff_ms:.0f}ms")
        else:
            # No inference, reuse last detections
            dets = last_dets
            frames_skipped += 1
            skipped_since_infer += 1
            preprocess_ms = None
            payload_kb = None
            round_trip_ms = None
            decode_ms = None
            inference_ms = None
            queue_wait_ms = None
            network_ms = None
            end_to_end_ms = shift_ms + storage_io_ms  # only disk read + content shift detection
            bandwidth_mbps = None

        # frames completed per second so far
        throughput_fps = frame_num / (time.time() - wall_start)

        # How late this frame is against its slot in the stream's schedule.
        # Frame N is due at wall_start + (N-1) * frame_interval; a lag that
        # climbs means the edge can't service the feed in real time.
        if frame_interval:
            pacing_lag_ms = (time.time() - wall_start - (frame_num - 1) * frame_interval) * 1000
            if pacing_lag_ms > frame_interval * 1000:
                frames_late += 1
            max_lag_ms = max(max_lag_ms, pacing_lag_ms)
        else:
            pacing_lag_ms = None   # unpaced: there's no schedule to be late against

        # Count labels
        labels = [d["label"] for d in dets]
        object_counts = {l: labels.count(l) for l in set(labels)}
        frame_ids = sorted(d["id"] for d in dets)
        seen_ids.update(frame_ids)

        # edge resource metrics 
        edge_cpu = host_usage["cpu"]
        edge_mem = host_usage["mem"]

        # --- assemble the metrics record ---
        # Assembled for EVERY decoded frame, gated or not, so the push cadence
        # below is unaffected by the gate mode. rnd() keeps the null-on-reuse
        record = {
            "stream_id": stream_id,
            "video": video_path,
            "frame": frame_num,
            "ts": time.time(),
            "storage_io_ms": round(storage_io_ms, 1),
            "preprocess_ms": rnd(preprocess_ms, 1),
            "round_trip_ms": rnd(round_trip_ms, 1),
            "decode_ms": rnd(decode_ms, 1),
            "inference_ms": rnd(inference_ms, 1),
            "queue_wait_ms": rnd(queue_wait_ms, 1),
            "network_ms": rnd(network_ms, 1),
            "end_to_end_ms": rnd(end_to_end_ms, 1),
            "throughput_fps": round(throughput_fps, 1),
            "objects_in_frame": len(dets),
            "counts": object_counts,
            "unique_total": len(seen_ids),
            "edge_cpu": edge_cpu,
            "edge_mem": edge_mem,
            "payload_kb": rnd(payload_kb, 1),
            "bandwidth_mbps": rnd(bandwidth_mbps, 2),
            "frame_diff": rnd(frame_diff, 2),
            "content_shift_detected": content_shift_detected,
            # change accrued since the previous inference (pre-reset), so the
            # budget can be profiled offline from any run, whatever the gate mode
            "change_since_infer": round(change_since_infer, 2),
            "ttff_ms": round(ttff_ms, 1) if frame_num == 1 else None,
            "pacing_lag_ms": rnd(pacing_lag_ms, 1),
            # fps the loop is pacing to, so the dashboard can show achieved
            # rate against the target instead of a bare throughput number
            "target_fps": round(src_fps, 2) if frame_interval else None,
            # frame gating
            "gate_mode": gate_mode,
            "inference_ran": run_inference,
            # fraction of decoded frames that never reached the model so far
            "filter_rate": round(frames_skipped / frame_num, 3),
            "backend": backend,
            "worker_id": worker_id,
            "infer_threads": infer_threads,
            "served_imgsz": served_imgsz,
            # density signals - logged, not yet consumed
            "box_count": density["box_count"],
            "small_boxes": density["small_boxes"],
            "mean_box_area": rnd(density["mean_box_area"], 5),
            "min_box_area": rnd(density["min_box_area"], 5),
            "mean_conf": rnd(density["mean_conf"], 3),
            "low_conf_frac": rnd(density["low_conf_frac"], 3),
            "overlap_pairs": density["overlap_pairs"],
            "density_metric": density_metric,
            "density_ewma": rnd(density_ewma, 3),
            "density_state": density_state,
            # motion signals - disp_rate* are null until two inferred frames share a track
            "disp_rate": rnd(disp_rate, 5),
            "disp_rate_max": rnd(disp_rate_max, 5),
            "disp_rate_ewma": rnd(disp_rate_ewma, 5),
            "tracks_matched": tracks_matched,
            "disp_accum": round(motion_since_infer, 5),
        }

        # Print to terminal. IDs in the frame are listed, not just counted, so a
        # gate that drops an object is visible as an ID that never appears.
        ids_desc = ",".join(str(i) for i in frame_ids) if frame_ids else "-"
        # The 'motion' gate spends drift, not pixel change, so show what it acted on
        accrued = (f"drift since infer: {motion_since_infer:.3f}" if gate_mode == "motion"
                   else f"change since infer: {change_since_infer:.1f}")
        with print_lock:
            state = "**Inferred**" + (" [floor]" if forced_by_floor else "") if run_inference else "Not inferred"
            print(f"Stream: {stream_id} Frame:{frame_num} | {state} | End-to-End: {end_to_end_ms:.0f}ms "
                  f"| {throughput_fps:.1f} FPS | {accrued} | density: {density_state} "
                  f"| IDs: [{ids_desc}] | Unique objects: {len(seen_ids)}")

        #  CSV log
        writer.writerow([record[k] for k in CSV_HEADER])
        # Flushed every row: these threads are daemons, so Ctrl-C kills the
        # process without closing the file and anything still buffered is lost.
        # Same reason the cloud metrics CSV flushes per row.
        csv_file.flush()

        # push metrics to cloud dashboard every N frames - regardless of gate mode
        if frame_num % PUSH_EVERY == 0:
            try:
                session.post(metrics_url, json=record, timeout=2)
            except requests.RequestException:
                pass

        # If frame processing finishes early, wait for the next frames slot rather than racing ahead infront of real time.
        # Sleeping against an absolute deadline (rather than a fixed sleep per
        # frame) means processing time is absorbed by the wait instead of
        # accumulating as drift.
        if frame_interval:
            next_due = wall_start + frame_num * frame_interval
            wait = next_due - time.time()
            if wait > 0:
                time.sleep(wait)

    cap.release()
    csv_file.close()

    with print_lock:
        pacing_desc = (f"paced={src_fps:.3g}fps late={frames_late} max_lag={max_lag_ms:.0f}ms "
                       if frame_interval else "unpaced ")
        if gate_mode == "budget":
            gate_detail = f"budget={diff_budget:g} max_skip={max_skip} floor_hits={budget_floor_hits} "
        elif gate_mode == "motion":
            gate_detail = (f"motion_budget={motion_budget:g} max_skip={max_skip} "
                           f"floor_hits={budget_floor_hits} final_rate={disp_rate_ewma or 0:.4f} ")
        else:
            gate_detail = ""
        print(f"stream {stream_id} COMPLETE: gate={gate_mode} {gate_detail}{pacing_desc}"
              f"frames={frame_num} inferred={frames_inferred} skipped={frames_skipped} "
              f"Unique Objects={len(seen_ids)}, Total time={time.time() - wall_start:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Edge client - processes N video streams concurrently.")
    parser.add_argument("--videos", nargs="+", default=["traffic.mp4"],
                        help="video files to process, one stream each")
    parser.add_argument("--streams", type=int, default=None,
                        help="run this many streams, cycling through --videos "
                             "(lets you load-test with copies of one file)")
    parser.add_argument("--host", default=CLOUD_HOST, help="cloud base URL")
    parser.add_argument("--gate", choices=GATE_MODES, default="none",
                        help="frame gating mode: none = infer on every frame "
                             "(baseline), fixed = every --frame-gap'th frame, "
                             "adaptive = only on content-shift frames (frame_diff "
                             "unusual vs a rolling baseline), budget = whenever the "
                             "change accumulated since the last inference exceeds "
                             "--budget, with a --max-skip safety floor, motion = "
                             "whenever tracked boxes are predicted to have drifted "
                             "past --motion-budget, same floor")
    parser.add_argument("--no-realtime", dest="realtime", action="store_false",
                        help="decode as fast as the machine allows instead of pacing to "
                             "the video's fps - measures raw pipeline capacity rather "
                             "than live-stream latency")
    parser.add_argument("--frame-gap", type=int, default=DEFAULT_FRAME_GAP,
                        help=f"'fixed' gate only: run inference every Nth frame "
                             f"(default {DEFAULT_FRAME_GAP})")
    parser.add_argument("--budget", type=float, default=DEFAULT_DIFF_BUDGET,
                        help=f"'budget' gate only: infer once summed frame_diff since "
                             f"the last inference reaches this value (default "
                             f"{DEFAULT_DIFF_BUDGET:g}). Pick it offline from the "
                             f"frame_diff column of an ungated run - it is the change "
                             f"you are willing to let pass unseen, so e.g. the sum of "
                             f"frame_diff over the longest gap you can tolerate")
    parser.add_argument("--max-skip", type=int, default=DEFAULT_MAX_SKIP,
                        help=f"'budget' and 'motion' gates: never skip more than this "
                             f"many consecutive frames, whatever the budget says - bounds "
                             f"the worst-case miss (default {DEFAULT_MAX_SKIP})")
    parser.add_argument("--motion-budget", type=float, default=DEFAULT_MOTION_BUDGET,
                        help=f"'motion' gate only: infer once a tracked box is predicted "
                             f"to have drifted this far, as a fraction of frame width "
                             f"(default {DEFAULT_MOTION_BUDGET:g}). Unlike --budget this "
                             f"is directly interpretable - 0.04 means 'never let a box go "
                             f"more than 4%% of the frame stale'. Profile the disp_rate_ewma "
                             f"column of an ungated run to see what a scene actually costs")
    parser.add_argument("--density-metric", choices=sorted(DENSITY_METRICS),
                        default=DEFAULT_DENSITY_METRIC,
                        help=f"which spatial signal drives the logged density state "
                             f"(default {DEFAULT_DENSITY_METRIC}). Nothing consumes the "
                             f"state yet - this chooses what gets EWMA'd and classified "
                             f"so the trigger can be picked from the CSV")
    parser.add_argument("--density-lo", type=float, default=DEFAULT_DENSITY_LO,
                        help=f"below this the density state is 'low' (default "
                             f"{DEFAULT_DENSITY_LO:g}). Units follow --density-metric, so "
                             f"pick both together from a profiling run")
    parser.add_argument("--density-hi", type=float, default=DEFAULT_DENSITY_HI,
                        help=f"above this the density state is 'high' (default "
                             f"{DEFAULT_DENSITY_HI:g})")
    parser.add_argument("--encode", choices=ENCODINGS, default="png",
                        help="wire format: png = lossless golden run, "
                             "jpeg = lossy baseline")
    parser.add_argument("--width", type=int, default=None,
                        help="downscale to this width before encoding, keeping the "
                             "source aspect ratio (e.g. 1920 / 1280 / 640). Omit, or "
                             "pass the native width, to send at full resolution. Match "
                             "this to the cloud's --imgsz for a like-for-like sweep")
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY,
                        help=f"jpeg only: encoder quality 1-100 (default {DEFAULT_JPEG_QUALITY})")
    args = parser.parse_args()

    if args.frame_gap < 1:
        parser.error("--frame-gap must be >= 1")

    if args.budget <= 0:
        parser.error("--budget must be > 0")

    if args.max_skip < 1:
        parser.error("--max-skip must be >= 1")

    if args.motion_budget <= 0:
        parser.error("--motion-budget must be > 0")

    if args.density_lo >= args.density_hi:
        parser.error("--density-lo must be < --density-hi")

    if args.width is not None and args.width < 1:
        parser.error("--width must be >= 1")

    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 1 and 100")

    count = args.streams or len(args.videos)
    sources = [args.videos[i % len(args.videos)] for i in range(count)]

    threading.Thread(target=sample_host_usage, daemon=True).start()

    gate_desc = args.gate
    if args.gate == "fixed":
        gate_desc += f" (every {args.frame_gap} frames)"
    elif args.gate == "budget":
        gate_desc += f" (budget {args.budget:g}, max skip {args.max_skip})"
    elif args.gate == "motion":
        gate_desc += f" (drift {args.motion_budget:g}, max skip {args.max_skip})"
    pace_desc = "real-time (source fps)" if args.realtime else "unpaced (max decode speed)"
    print(f"Starting {count} concurrent stream(s) -> {args.host} | gate: {gate_desc} | pacing: {pace_desc} "
          f"| density signal: {args.density_metric} ({args.density_lo:g}/{args.density_hi:g})")
    threads = []
    for stream_id, video_path in enumerate(sources):
        t = threading.Thread(target=run_stream,
                             kwargs=dict(stream_id=stream_id, video_path=video_path,
                                         host=args.host, gate_mode=args.gate,
                                         frame_gap=args.frame_gap, realtime=args.realtime,
                                         encoding=args.encode, width=args.width,
                                         jpeg_quality=args.jpeg_quality,
                                         diff_budget=args.budget, max_skip=args.max_skip,
                                         motion_budget=args.motion_budget,
                                         density_metric=args.density_metric,
                                         density_lo=args.density_lo, density_hi=args.density_hi),
                             name=f"stream-{stream_id}", daemon=True)
        t.start()
        threads.append(t)

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\nInterrupted - stopping.")

    print("All streams finished.")


if __name__ == "__main__":
    main()
