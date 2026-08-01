import os
import cv2
import numpy as np

dataset_dir = "UTKFace"
images = []
age = []

file_name = [f for f in os.listdir(dataset_dir) if f.endswith('.jpg')]
file_name = file_name[:1000]

print("Reading and converting image data...")
for file_name in file_name:
    p = file_name.split('_')
    if len(p) <= 4:
        age = int(p[0])

        img_path = os.path.join(dataset_dir,file_name)
        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)

        if img is not None:
            img = cv2.resize(img,(64,64))
            images.append(img.flatten())
            age.appdend(age)