import argparse
import csv
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

PUSH_EVERY = 15          # metric push cadence

# Real-time pacing: a file decodes as fast as the CPU allows, so set a sleep to pace to the videos fps.
FALLBACK_FPS = 30.0      # used when the container reports no usable fps

# For comparison of methods - `none` infer on every frame, `fixed` infer every N frames, and `adaptive` on content-shift changes
GATE_MODES = ("none", "fixed", "adaptive")
DEFAULT_FRAME_GAP = 5

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
    "frame_diff", "content_shift_detected", "ttff_ms",
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


def run_stream(stream_id, video_path, host, gate_mode="none", frame_gap=DEFAULT_FRAME_GAP,
               realtime=True, encoding="png", width=None, jpeg_quality=DEFAULT_JPEG_QUALITY):
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

    # Cache last detections so they're reused for frames not sent to the cloud
    last_dets = None
    frames_inferred = 0
    frames_skipped = 0
    frames_late = 0            # frames that finished after their scheduled slot
    max_lag_ms = 0.0
    backend = None            # reported by the cloud on each /detect response
    worker_id = None          # cloud model instance serving this stream
    infer_threads = None      # CPU threads that instance was given

    csv_file = open(f"edge_metrics_stream{stream_id}.csv", "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(CSV_HEADER)

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

        # Decide whether to run inference on this frame based on gating mode defined
        if last_dets is None or gate_mode == "none":
            run_inference = True
        elif gate_mode == "fixed":
            run_inference = (frame_num % frame_gap == 0)
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
            except requests.RequestException as e:
                with print_lock:
                    print(f"[stream {stream_id}] Request failed: {e}")
                continue

            last_dets = dets
            frames_inferred += 1

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
        seen_ids.update(d["id"] for d in dets)

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
        }

        # Print to terminal
        with print_lock:
            if run_inference:
                print(f"Stream: {stream_id} Frame:{frame_num} | **Inferred** |End-to-End: {end_to_end_ms:.0f}ms | {throughput_fps:.1f} FPS | Unique objects: {len(seen_ids)}")
            else:
                print(f"Stream: {stream_id}, Frame:{frame_num} | Not inferred | End-to-End: {end_to_end_ms:.0f}ms | {throughput_fps:.1f} FPS | Unique objects: {len(seen_ids)}")

        #  CSV log
        writer.writerow([record[k] for k in CSV_HEADER])

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
        print(f"stream {stream_id} COMPLETE: gate={gate_mode} {pacing_desc}"
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
                             "(baseline), fixed = every --frame-gap'th frame, " #####edit this desciptio
                             "adaptive = only on content-shift frames")
    parser.add_argument("--no-realtime", dest="realtime", action="store_false",
                        help="decode as fast as the machine allows instead of pacing to "
                             "the video's fps - measures raw pipeline capacity rather "
                             "than live-stream latency")
    parser.add_argument("--frame-gap", type=int, default=DEFAULT_FRAME_GAP,
                        help=f"'fixed' gate only: run inference every Nth frame "
                             f"(default {DEFAULT_FRAME_GAP})")
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

    if args.width is not None and args.width < 1:
        parser.error("--width must be >= 1")

    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 1 and 100")

    count = args.streams or len(args.videos)
    sources = [args.videos[i % len(args.videos)] for i in range(count)]

    threading.Thread(target=sample_host_usage, daemon=True).start()

    gate_desc = args.gate + (f" (every {args.frame_gap} frames)" if args.gate == "fixed" else "")
    pace_desc = "real-time (source fps)" if args.realtime else "unpaced (max decode speed)"
    print(f"Starting {count} concurrent stream(s) -> {args.host} | gate: {gate_desc} | pacing: {pace_desc}")
    threads = []
    for stream_id, video_path in enumerate(sources):
        t = threading.Thread(target=run_stream,
                             args=(stream_id, video_path, args.host, args.gate, args.frame_gap,
                                   args.realtime, args.encode, args.width, args.jpeg_quality),
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
