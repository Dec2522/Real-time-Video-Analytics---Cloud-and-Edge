import cv2
import time
import requests

CLOUD_URL = "http://10.0.0.2:8000/detect"   # local test: "http://127.0.0.1:8000/detect"
SHOW_VIDEO = False                            # False on headless VMs

cap = cv2.VideoCapture("traffic.mp4")
if not cap.isOpened():
    print("ERROR: could not open video file")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
frame_delay = 1.0 / fps

seen_ids = set()                              # every unique track ID ever seen

while cap.isOpened():
    start = time.time()
    success, frame = cap.read()
    if not success:
        print("Playback complete.")
        break

    # --- edge preprocessing: resize + JPEG encode ---
    small = cv2.resize(frame, (960, 540))
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])

    # --- send to cloud, get detections + metrics ---
    try:
        resp = requests.post(CLOUD_URL, data=buf.tobytes(),
                             headers={"Content-Type": "application/octet-stream"},
                             timeout=5)
        data = resp.json()
        dets = data["detections"]
        metrics = data.get("metrics", {})
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        continue

    # --- per-frame counts ---
    labels = [d["label"] for d in dets]
    counts = {l: labels.count(l) for l in set(labels)}

    # --- unique vehicle total across the whole run ---
    seen_ids.update(d["id"] for d in dets)

    elapsed = time.time() - start

    # --- print everything together ---
    print(
        f"Round-trip: {elapsed*1000:.0f}ms | "
        f"In frame: {len(dets)} {counts} | "
        f"Unique vehicles: {len(seen_ids)} | "
        f"Cloud: infer {metrics.get('inference_ms', '?')}ms "
        f"CPU {metrics.get('cpu_percent', '?')}% "
        f"Mem {metrics.get('mem_percent', '?')}%"
    )

    if SHOW_VIDEO:
        cv2.imshow("Edge View", small)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
if SHOW_VIDEO:
    cv2.destroyAllWindows()