import os

import cv2

path = "data/augmentedSpectrogramImages-640"

for d in os.listdir(path):
    for f in os.listdir(os.path.join(path, d)):
        img = cv2.imread(os.path.join(path, d, f))
        img = cv2.resize(img, (640, 640))
        cv2.imwrite(os.path.join(path, d, f), img)
        print(f"Resized {f} in {d}")

print("Done")
