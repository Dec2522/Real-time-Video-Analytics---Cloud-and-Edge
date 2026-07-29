import cv2
import time
import requests
import psutil
import csv
from collections import deque

CLOUD_DETECT = "http://10.0.0.2:8000/detect"
CLOUD_METRICS = "http://10.0.0.2:8000/metrics"

cap = cv2.VideoCapture("traffic.mp4")
if not cap.isOpened():
    print("ERROR: could not open video file")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
frame_delay = 1.0 / fps

seen_ids = set()
frame_num = 0
wall_start = time.time()          # for throughput: frames per real second, and TTFF
ttff_ms = None                    # set once, when the first frame's result comes back

# --- content-shift signal (Reducto-style): rolling baseline of frame-to-frame pixel diff ---
prev_gray = None
diff_history = deque(maxlen=90)   # ~3s of history at 30fps to build the rolling baseline
DIFF_WARMUP = 30                  # frames before we trust the rolling mean/std
DIFF_K = 3.0                      # outlier threshold: mean + k*std

# CSV logging — for the evaluation graphs later
csv_file = open("edge_metrics.csv", "w", newline="")
writer = csv.writer(csv_file)
writer.writerow([
    "frame", "ts", "storage_io_ms", "preprocess_ms", "round_trip_ms",
    "decode_ms", "inference_ms", "network_ms", "end_to_end_ms",
    "throughput_fps", "objects_in_frame", "unique_total",
    "edge_cpu", "edge_mem", "payload_kb", "bandwidth_mbps",
    "frame_diff", "content_shift_detected", "ttff_ms"
])

while cap.isOpened():
    start = time.time()

    # --- storage I/O: time to pull the next frame from disk ---
    io0 = time.time()
    success, frame = cap.read()
    storage_io_ms = (time.time() - io0) * 1000
    if not success:
        print("Playback complete.")
        break
    frame_num += 1

    # --- content-shift signal: cheap grayscale frame-to-frame diff (Reducto-style) ---
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

    # --- edge preprocessing (timed) ---
    prep0 = time.time()
    small = cv2.resize(frame, (960, 540))
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
    preprocess_ms = (time.time() - prep0) * 1000
    payload_kb = len(buf) / 1024

    # --- send frame, time the round trip ---
    try:
        rt0 = time.time()
        resp = requests.post(CLOUD_DETECT, data=buf.tobytes(),
                             headers={"Content-Type": "application/octet-stream"},
                             timeout=5)
        round_trip_ms = (time.time() - rt0) * 1000
        data = resp.json()
        dets = data["detections"]
        decode_ms = data.get("decode_ms", 0)
        inference_ms = data.get("inference_ms", 0)
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        continue

    # network time = round trip minus cloud-side decode + inference
    network_ms = round_trip_ms - decode_ms - inference_ms

    # end-to-end latency: disk read -> encode -> network -> cloud decode+infer -> network back
    end_to_end_ms = storage_io_ms + preprocess_ms + round_trip_ms

    # bandwidth actually achieved on this frame's upload
    bandwidth_mbps = (len(buf) * 8 / (network_ms / 1000) / 1e6) if network_ms > 0 else 0

    # time to first frame: only set once, on the first successful result
    if ttff_ms is None:
        ttff_ms = (time.time() - wall_start) * 1000
        print(f">>> Time to first frame: {ttff_ms:.0f}ms")

    # --- throughput: frames completed per real second so far ---
    throughput_fps = frame_num / (time.time() - wall_start)

    # --- application metrics ---
    labels = [d["label"] for d in dets]
    counts = {l: labels.count(l) for l in set(labels)}
    seen_ids.update(d["id"] for d in dets)

    # --- edge resource metrics ---
    edge_cpu = psutil.cpu_percent()
    edge_mem = psutil.virtual_memory().percent

    # --- assemble the metrics record ---
    record = {
        "frame": frame_num,
        "ts": time.time(),
        "storage_io_ms": round(storage_io_ms, 1),
        "preprocess_ms": round(preprocess_ms, 1),
        "round_trip_ms": round(round_trip_ms, 1),
        "decode_ms": round(decode_ms, 1),
        "inference_ms": round(inference_ms, 1),
        "network_ms": round(network_ms, 1),
        "end_to_end_ms": round(end_to_end_ms, 1),
        "throughput_fps": round(throughput_fps, 1),
        "objects_in_frame": len(dets),
        "counts": counts,
        "unique_total": len(seen_ids),
        "edge_cpu": edge_cpu,
        "edge_mem": edge_mem,
        "payload_kb": round(payload_kb, 1),
        "bandwidth_mbps": round(bandwidth_mbps, 2),
        "frame_diff": round(frame_diff, 2) if frame_diff is not None else None,
        "content_shift_detected": content_shift_detected,
        "ttff_ms": round(ttff_ms, 1) if frame_num == 1 else None,
    }

    # --- print ---
    shift_flag = " *** CONTENT SHIFT ***" if content_shift_detected else ""
    print(f"F{frame_num} | io {storage_io_ms:.0f}ms + prep {preprocess_ms:.0f}ms + RT {round_trip_ms:.0f}ms "
          f"(decode {decode_ms:.0f} + infer {inference_ms:.0f} + net {network_ms:.0f}) "
          f"| e2e {end_to_end_ms:.0f}ms | {throughput_fps:.1f} FPS | "
          f"in-frame {len(dets)} {counts} | unique {len(seen_ids)} | "
          f"edge cpu {edge_cpu}% mem {edge_mem}% | {payload_kb:.0f}KB @ {bandwidth_mbps:.1f}Mbps"
          f"{shift_flag}")

    # --- CSV log ---
    writer.writerow([
        frame_num, record["ts"], record["storage_io_ms"], record["preprocess_ms"],
        record["round_trip_ms"], record["decode_ms"], record["inference_ms"],
        record["network_ms"], record["end_to_end_ms"], record["throughput_fps"],
        record["objects_in_frame"], record["unique_total"],
        record["edge_cpu"], record["edge_mem"], record["payload_kb"],
        record["bandwidth_mbps"], record["frame_diff"],
        record["content_shift_detected"], record["ttff_ms"]
    ])

    # --- push to cloud dashboard (every 15 frames to keep it light) ---
    if frame_num % 15 == 0:
        try:
            requests.post(CLOUD_METRICS, json=record, timeout=2)
        except requests.RequestException:
            pass

cap.release()
csv_file.close()