import cv2
import time
import requests
import psutil
import csv

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

# CSV logging — for the evaluation graphs later
csv_file = open("edge_metrics.csv", "w", newline="")
writer = csv.writer(csv_file)
writer.writerow([
    "frame", "ts", "round_trip_ms", "inference_ms", "network_ms",
    "objects_in_frame", "unique_total",
    "edge_cpu", "edge_mem", "payload_kb"
])

while cap.isOpened():
    start = time.time()
    success, frame = cap.read()
    if not success:
        print("Playback complete.")
        break
    frame_num += 1

    # --- edge preprocessing ---
    small = cv2.resize(frame, (960, 540))
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
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
        inference_ms = data.get("inference_ms", 0)
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        continue

    # network time = round trip minus cloud compute (rough)
    network_ms = round_trip_ms - inference_ms

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
        "round_trip_ms": round(round_trip_ms, 1),
        "inference_ms": round(inference_ms, 1),
        "network_ms": round(network_ms, 1),
        "objects_in_frame": len(dets),
        "counts": counts,
        "unique_total": len(seen_ids),
        "edge_cpu": edge_cpu,
        "edge_mem": edge_mem,
        "payload_kb": round(payload_kb, 1),
    }

    # --- print ---
    print(f"F{frame_num} | RT {round_trip_ms:.0f}ms "
          f"(infer {inference_ms:.0f} + net {network_ms:.0f}) | "
          f"in-frame {len(dets)} {counts} | unique {len(seen_ids)} | "
          f"edge cpu {edge_cpu}% mem {edge_mem}% | {payload_kb:.0f}KB")

    # --- CSV log ---
    writer.writerow([
        frame_num, record["ts"], record["round_trip_ms"], record["inference_ms"],
        record["network_ms"], record["objects_in_frame"], record["unique_total"],
        record["edge_cpu"], record["edge_mem"], record["payload_kb"]
    ])

    # --- push to cloud dashboard (every 15 frames to keep it light) ---
    if frame_num % 15 == 0:
        try:
            requests.post(CLOUD_METRICS, json=record, timeout=2)
        except requests.RequestException:
            pass   # don't let a metrics push failure break the pipeline

cap.release()
csv_file.close()