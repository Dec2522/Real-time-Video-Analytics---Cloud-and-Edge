import cv2
import time
import requests

CLOUD_URL = "http://10.0.0.2:8000/detect" # change to http://10.0.0.2:8000/detect on VMs "http://127.0.0.1:8000/detect" 
SHOW_VIDEO = Falsec                              # False on headless VMs

cap = cv2.VideoCapture("traffic.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
frame_delay = 1.0 / fps

while cap.isOpened():
    start = time.time()
    success, frame = cap.read()
    if not success:
        print("Playback complete.")
        break

    # --- edge preprocessing: resize + JPEG encode ---
    small = cv2.resize(frame, (960, 540))
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])

    # --- send to cloud, get detections ---
    try:
        resp = requests.post(CLOUD_URL, data=buf.tobytes(),
                             headers={"Content-Type": "application/octet-stream"},
                             timeout=5)
        dets = resp.json()["detections"]
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        continue

    # --- draw results on the (resized) frame ---
    for d in dets:
        x1, y1, x2, y2 = map(int, d["box"])
        cv2.rectangle(small, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(small, f'{d["label"]} {d["id"]}', (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    elapsed = time.time() - start
    print(f"Round-trip: {elapsed*1000:.0f}ms | {len(dets)} objects")

    if SHOW_VIDEO:
        cv2.imshow("Edge View", small)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
if SHOW_VIDEO:
    cv2.destroyAllWindows()