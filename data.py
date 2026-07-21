import kagglehub
import os

# 1. Download the dataset
path = kagglehub.dataset_download("chicicecream/720p-road-and-traffic-video-for-object-detection")
print("Dataset folder:", path)

# 2. List what's actually in there so you know the video filename
for root, dirs, files in os.walk(path):
    for f in files:
        print(os.path.join(root, f))