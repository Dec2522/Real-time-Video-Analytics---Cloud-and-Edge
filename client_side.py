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

# For comparison of methods - `none` infer on every frame, `fixed` infer every N frames, and `adaptive` on content-shift changes
GATE_MODES = ("none", "fixed", "adaptive")
DEFAULT_FRAME_GAP = 5    

# Save results for post-run analysis
CSV_HEADER = [
    "stream_id", "frame", "ts", "storage_io_ms", "preprocess_ms", "round_trip_ms",
    "decode_ms", "inference_ms", "network_ms", "end_to_end_ms",
    "throughput_fps", "objects_in_frame", "unique_total",
    "edge_cpu", "edge_mem", "payload_kb", "bandwidth_mbps",
    "frame_diff", "content_shift_detected", "ttff_ms",
    # --- frame gating ---
    "gate_mode", "inference_ran", "filter_rate",
    # cloud runtime that served this run, so backend comparisons are self-labelling
    "backend",
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


def run_stream(stream_id, video_path, host, gate_mode="none", frame_gap=DEFAULT_FRAME_GAP):
    """Process a single video stream, sending frames to the cloud for inference, and logging metrics."""
  
    detect_url = f"{host}/detect"
    metrics_url = f"{host}/metrics"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[stream {stream_id}] ERROR: could not open {video_path}")
        return

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
    backend = None            # reported by the cloud on each /detect response

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
            small = cv2.resize(frame, (960, 540))
            ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])  # set image quality
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
                backend = data.get("backend")
            except requests.RequestException as e:
                with print_lock:
                    print(f"[stream {stream_id}] Request failed: {e}")
                continue

            last_dets = dets
            frames_inferred += 1

            # network time = round trip minus cloud-side decode + inference
            network_ms = round_trip_ms - decode_ms - inference_ms

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
            network_ms = None
            end_to_end_ms = shift_ms + storage_io_ms  # only disk read + content shift detection
            bandwidth_mbps = None

        # frames completed per second so far
        throughput_fps = frame_num / (time.time() - wall_start)

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
        # latency fields as real nulls instead of crashing on round(None).
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
            # frame gating
            "gate_mode": gate_mode,
            "inference_ran": run_inference,
            # fraction of decoded frames that never reached the model so far
            "filter_rate": round(frames_skipped / frame_num, 3),
            "backend": backend,
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

    cap.release()
    csv_file.close()

    with print_lock:
        rate = frames_skipped / frame_num if frame_num else 0
        print(f"stream {stream_id} COMPLETE: gate={gate_mode} "
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
                             "adaptive = only on content-shift frames")
    parser.add_argument("--frame-gap", type=int, default=DEFAULT_FRAME_GAP,
                        help=f"'fixed' gate only: run inference every Nth frame "
                             f"(default {DEFAULT_FRAME_GAP})")
    args = parser.parse_args()

    if args.frame_gap < 1:
        parser.error("--frame-gap must be >= 1")

    count = args.streams or len(args.videos)
    sources = [args.videos[i % len(args.videos)] for i in range(count)]

    threading.Thread(target=sample_host_usage, daemon=True).start()

    gate_desc = args.gate + (f" (every {args.frame_gap} frames)" if args.gate == "fixed" else "")
    print(f"Starting {count} concurrent stream(s) -> {args.host} | gate: {gate_desc}")
    threads = []
    for stream_id, video_path in enumerate(sources):
        t = threading.Thread(target=run_stream,
                             args=(stream_id, video_path, args.host, args.gate, args.frame_gap),
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
