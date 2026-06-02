# Driver Monitoring System

A machine learning system for real-time driver drowsiness and fatigue detection, built using facial landmark analysis and three different ML models.

## Motivation

Road traffic accidents remain one of the leading causes of death worldwide. According to the World Health Organization, **1.19 million people die on roads every year** — approximately 3,200 deaths per day. Drowsy driving accounts for up to **20% of all traffic accidents globally**, and research shows that 18 hours without sleep impairs driving ability equivalent to a blood alcohol level of 0.05%, while 24 hours without sleep equals 0.10% — above the legal limit in most countries.

Since 2024, the European Union requires all new vehicles to include fatigue and attention detection systems. This project demonstrates how such systems work under the hood.

---

## How It Works

The system uses **MediaPipe** to extract 3D facial landmark coordinates from images and computes two key metrics:

- **EAR (Eye Aspect Ratio)** — calculated from 6 landmark points around each eye. High EAR (~0.3) means open eyes; as eyes close, EAR drops toward 0.
- **MAR (Mouth Aspect Ratio)** — calculated from 8 landmark points around the mouth. Low MAR (~1.0) means closed mouth; MAR increases when the mouth opens (yawning).

These values are used to classify the driver's state.

---

## Dataset

- ~50,000 labeled images (approx. 700MB total, small resolution)
- Each image pre-labeled as one of three classes: **Alert**, **Microsleep**, or **Yawning**
- Images converted to a CSV dataset containing EAR, MAR, state, and fatigue label (0 or 1)
- Class balance: **33% per class** — near-perfect distribution

### Data Cleaning

Outliers were removed using the following thresholds:
- EAR > 0.6 → removed
- MAR > 4.0 → removed

Only 2 outliers were found and removed, leaving class balance unchanged.

---

## Models

Three models were trained and compared on the same dataset (80% train / 20% validation split):

### 1. Logistic Regression
- Baseline model — simplest approach
- Binary classification: Alert vs. Fatigued
- Finds a linear decision boundary based on EAR and MAR values
- **Accuracy: 85.94%**

### 2. Random Forest (100 trees)
- Binary classification: Alert vs. Fatigued
- Each tree votes independently; majority wins
- Resistant to outliers; no normalization required
- Feature importance: EAR = **69.8%**, MAR = **30.2%**
- **Accuracy: 87.21%**

### 3. CNN (Convolutional Neural Network)
- 3-class classification: Alert, Microsleep, Yawning
- Trained directly on raw 64×64 pixel images (6,000 images, 2,000 per class)
- 10 training epochs
- Learns spatial features: eye shape, mouth shape, head position
- **Accuracy: 97.08%**

---

## Results

| Model               | Classes              | Accuracy |
|---------------------|----------------------|----------|
| Logistic Regression | Alert / Fatigued     | 85.94%   |
| Random Forest       | Alert / Fatigued     | 87.21%   |
| CNN                 | Alert / Microsleep / Yawning | **97.08%** |

The CNN significantly outperforms both classical models. Training and validation accuracy curves converge cleanly with no signs of overfitting. The model's most critical error — predicting Alert when the driver is in Microsleep — occurred only **15 times**, compared to 79 times for Logistic Regression.

---

## Project Structure

```
├── scripts/
│   ├── app.py                  # Main application
│   ├── eda.py                  # Exploratory data analysis
│   ├── extract_features.py     # MediaPipe landmark extraction → CSV
│   ├── model_cnn.py            # CNN model training
│   ├── model_lr.py             # Logistic Regression training
│   ├── model_rf.py             # Random Forest training
│   ├── reorganize_dataset.py   # Dataset preparation
│   └── test.py                 # Model evaluation
├── face_landmarker.task        # MediaPipe face landmark model
├── requirements.txt
└── .gitignore
```

---

## Setup

```bash
# Clone the repository
git clone https://github.com/adelisamujaric/Driver-Monitoring-System.git
cd Driver-Monitoring-System

# Install dependencies
pip install -r requirements.txt

#Then start the application:
python3 scripts/app.py
```

After startup, copy the link `http://127.0.0.1:7860` and open it in Chrome or Firefox (not in the VS Code built-in browser). Allow camera access when the browser prompts you.

---
## 🤖 Agent-like Behavior

The Live Camera tab implements a reactive perception-action loop:

- **Perceives:** Each webcam frame (live image of the driver's face)
- **Decides:** The Random Forest model computes EAR and MAR values from facial landmarks and classifies the driver's state as Alert or Fatigued. If the driver appears fatigued for more than 3 consecutive seconds, an alarm is triggered.
- **Acts:** Displays a warning overlay on the video feed and plays an audio alarm
- **Learning:** The model does not learn online — it was pre-trained on images. Adapting to new data requires retraining.

---
## Technologies

- **Python**
- **MediaPipe** — facial landmark detection
- **scikit-learn** — Logistic Regression, Random Forest
- **TensorFlow / Keras** — CNN
- **pandas / NumPy** — data processing
- **matplotlib** — visualization
