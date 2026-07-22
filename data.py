import kagglehub
import os
import shutil

# 1. Download the dataset
path = kagglehub.dataset_download("chicicecream/720p-road-and-traffic-video-for-object-detection")
print("Dataset folder:", path)

# 2. Find the .mp4 and copy it into the current project folder as traffic.mp4
for root, dirs, files in os.walk(path):
    for f in files:
        if f.lower().endswith(".mp4"):
            src = os.path.join(root, f)
            dst = "traffic.mp4"   # lands in your project folder
            shutil.copy(src, dst)
            print(f"Copied to: {os.path.abspath(dst)}")