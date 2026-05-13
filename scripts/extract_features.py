import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from scipy.spatial import distance

# EAR funkcija
def calculate_ear(eye_points):
    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    C = distance.euclidean(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

# MAR funkcija
def calculate_mar(mouth_points):
    A = distance.euclidean(mouth_points[1], mouth_points[7])
    B = distance.euclidean(mouth_points[2], mouth_points[6])
    C = distance.euclidean(mouth_points[3], mouth_points[5])
    D = distance.euclidean(mouth_points[0], mouth_points[4])
    mar = (A + B + C) / (2.0 * D)
    return mar

# Novi MediaPipe API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='/Users/adelisa/FIT/ML/face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# Landmarks indeksi
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [61, 39, 37, 0, 267, 269, 291, 405]

DATASET_PATH = "/Users/adelisa/FIT/ML/dataset_organized"
OUTPUT_PATH = "/Users/adelisa/FIT/ML/outputs/features.csv"

data = []
total = 0
skipped = 0

print("Skripta pokrenuta!")

for label_folder in ["alert", "microsleep", "yawning"]:
    folder_path = os.path.join(DATASET_PATH, label_folder)
    images = os.listdir(folder_path)
    images = [img for img in images if img.endswith((".jpg", ".jpeg", ".png"))][:2000]
    
    print(f"Obrada: {label_folder} ({len(images)} slika)...")
    
    for img_file in images:
        img_path = os.path.join(folder_path, img_file)
        image = cv2.imread(img_path)
        if image is None:
            skipped += 1
            continue
        
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)
        
        if not result.face_landmarks:
            skipped += 1
            continue
        
        landmarks = result.face_landmarks[0]
        h, w = image.shape[:2]
        
        def get_points(indices):
            return [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]
        
        left_eye = get_points(LEFT_EYE)
        right_eye = get_points(RIGHT_EYE)
        mouth = get_points(MOUTH)
        
        ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0
        mar = calculate_mar(mouth)
        
        if label_folder == "alert":
            label = 0
        else:
            label = 1
        
        data.append({
            "ear": round(ear, 4),
            "mar": round(mar, 4),
            "driver_state": label_folder,
            "label": label
        })
        total += 1

df = pd.DataFrame(data)
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nGotovo!")
print(f"Ukupno obrađeno: {total}")
print(f"Preskočeno: {skipped}")
print(df["driver_state"].value_counts())