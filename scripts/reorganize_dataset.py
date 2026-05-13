import os
import json
import shutil

DATASET_PATH = "/Users/adelisa/FIT/ML/dataset"
OUTPUT_PATH = "/Users/adelisa/FIT/ML/dataset_organized"

# Napravi output foldere
for label in ["alert", "microsleep", "yawning"]:
    os.makedirs(os.path.join(OUTPUT_PATH, label), exist_ok=True)

count = {"alert": 0, "microsleep": 0, "yawning": 0}

# Prodi kroz sve foldere osoba
for person_folder in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person_folder)
    
    if not os.path.isdir(person_path):
        continue
    
    json_path = os.path.join(person_path, "annotations_final.json")
    
    if not os.path.exists(json_path):
        continue
    
    with open(json_path) as f:
        annotations = json.load(f)
    
    for frame_name, info in annotations.items():
        driver_state = info["driver_state"]
        
        if driver_state not in count:
            continue
        
        src = os.path.join(person_path, frame_name)
        dst = os.path.join(OUTPUT_PATH, driver_state, 
                          f"{person_folder}_{frame_name}")
        
        if os.path.exists(src):
            shutil.copy2(src, dst)
            count[driver_state] += 1

print("Gotovo!")
print(f"Alert: {count['alert']}")
print(f"Microsleep: {count['microsleep']}")
print(f"Yawning: {count['yawning']}")