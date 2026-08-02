import kagglehub
import os
import shutil

path = kagglehub.dataset_download("chicicecream/720p-road-and-traffic-video-for-object-detection")
print("Dataset folder:", path)

for root, dirs, files in os.walk(path):
    for f in files:
        if f.lower().endswith(".mp4"):
            src = os.path.join(root, f)
            dst = "traffic.mp4"   
            shutil.copy(src, dst)
            print(f"Copied to: {os.path.abspath(dst)}")