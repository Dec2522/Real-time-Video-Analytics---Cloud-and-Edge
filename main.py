import cv2
import time
from ultralytics import YOLO


# yolo11n.pt - nano
# yolo11s.pt - small
# yolo11m.pt - medium

model = YOLO('yolo11s.pt')

video_path = "traffic.mp4"
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
frame_delay = 1.0 / fps

frame_num = 0
while cap.isOpened():
    start = time.time()
    success, frame = cap.read()
    if not success:
        print("Playback complete.")
        break

    frame_num += 1
    if frame_num % 2 != 0:
        print(f"Skipped inference: frame {frame_num}")
        continue

    results = model.track(frame, persist=True, conf=0.3, verbose = True, imgsz = 620)

    # --- print what was seen this frame ---
    boxes = results[0].boxes
    if boxes.id is not None:                      # IDs exist only when tracking locks on
        ids = boxes.id.int().tolist()             # track IDs
        classes = boxes.cls.int().tolist()        # class indices
        confs = boxes.conf.tolist()               # confidence scores
        names = results[0].names                  # index → label lookup (e.g. 2 → 'car')

        print(f"Frame Length: {frame_delay * 1000} ms")
        print(f"Frame {frame_num}: {len(ids)} objects")
        for tid, cls, conf in zip(ids, classes, confs):
            print(f"   ID {tid}: {names[cls]} ({conf:.2f})")
    else:
        print(f"Frame {frame_num}: no tracked objects")
    # ---------------------------------------

    annotated = results[0].plot()
    cv2.imshow("Traffic Analytics", annotated)

    elapsed = time.time() - start
    sleep_ms = max(1, int((frame_delay - elapsed) * 1000))
    if cv2.waitKey(sleep_ms) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()