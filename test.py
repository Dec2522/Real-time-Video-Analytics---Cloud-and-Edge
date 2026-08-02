from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# .predictor doesn't exist until the first inference — so run one
import numpy as np
dummy = np.zeros((540, 960, 3), dtype=np.uint8)   # a blank frame
model.track(dummy, persist=True, verbose=False)

# now check
print("has predictor:", hasattr(model, "predictor"))
print("has trackers: ", hasattr(model.predictor, "trackers"))
print("trackers:     ", model.predictor.trackers)