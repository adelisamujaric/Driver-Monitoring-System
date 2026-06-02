import gradio as gr
import numpy as np
import pickle
import cv2
import mediapipe as mp
import tensorflow as tf
from keras.models import load_model
from keras.utils import load_img, img_to_array
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import time
import os


# Globalne varijable za praćenje vremena
umoran_od = None
PRAG_SEKUNDI = 3

# ============================================================
# UČITAVANJE MODELA
# ============================================================

#Logisticka regresija
with open("outputs/driver_model_lr.pkl", "rb") as f:
    lr_model, lr_scaler = pickle.load(f)

# Random Forest + Scaler
with open("outputs/driver_model_rf.pkl", "rb") as f:
    rf_model, scaler = pickle.load(f)

# CNN model
cnn_model = load_model("outputs/driver_model_cnn.h5")


# MediaPipe

base_options = mp_python.BaseOptions(
    model_asset_path='face_landmarker.task'
)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1
)
face_mesh = vision.FaceLandmarker.create_from_options(options)

# Indeksi za EAR i MAR
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [61, 39, 37, 0, 267, 269, 291, 405]

KLASE_CNN = ['alert', 'microsleep', 'yawning']
IMG_SIZE = 64

# ============================================================
# HELPER FUNKCIJE
# ============================================================

def euclidean(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def compute_ear(landmarks, indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    A = euclidean(pts[1], pts[5])
    B = euclidean(pts[2], pts[4])
    C = euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C)

def compute_mar(landmarks, indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    A = euclidean(pts[1], pts[7])
    B = euclidean(pts[2], pts[6])
    C = euclidean(pts[3], pts[5])
    D = euclidean(pts[0], pts[4])
    return (A + B + C) / (2.0 * D)

# ============================================================
# TAB 1 - Logisticka regresija
# ============================================================

def predict_lr(image):
    if image is None:
        return "❌ Nisi uploadala sliku!", "", ""

    img_rgb = np.array(image)
    h, w = img_rgb.shape[:2]

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    results = face_mesh.detect(mp_image)

    if not results.face_landmarks:
        return "❌ Lice nije detektovano!", "", ""

    landmarks = results.face_landmarks[0]

    ear_left = compute_ear(landmarks, LEFT_EYE, w, h)
    ear_right = compute_ear(landmarks, RIGHT_EYE, w, h)
    ear = (ear_left + ear_right) / 2.0
    mar = compute_mar(landmarks, MOUTH, w, h)

    features = lr_scaler.transform([[ear, mar]])
    prediction = lr_model.predict(features)[0]
    proba = lr_model.predict_proba(features)[0]

    if prediction == 0:
        rezultat = "✅ ALERT - Vozač je budan"
    else:
        rezultat = "🚨 UMORAN - Vozač je umoran!"

    ear_mar_info = f"EAR: {ear:.4f} | MAR: {mar:.4f}"
    proba_info = f"Alert={proba[0]*100:.1f}% | Umoran={proba[1]*100:.1f}%"

    return rezultat, ear_mar_info, proba_info

# ============================================================
# TAB 2 - RANDOM FOREST
# ============================================================

def predict_rf(image):
    if image is None:
        return "❌ Nisi uploadala sliku!", "", ""

    # Konvertuj u BGR za MediaPipe
    img_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    results = face_mesh.detect(mp_image)

    if not results.face_landmarks:
        return "❌ Lice nije detektovano!", "", ""

    landmarks = results.face_landmarks[0]

    ear_left = compute_ear(landmarks, LEFT_EYE, w, h)
    ear_right = compute_ear(landmarks, RIGHT_EYE, w, h)
    ear = (ear_left + ear_right) / 2.0
    mar = compute_mar(landmarks, MOUTH, w, h)

    # Predikcija
    features = scaler.transform([[ear, mar]])
    prediction = rf_model.predict(features)[0]
    proba = rf_model.predict_proba(features)[0]

    if prediction == 0:
        rezultat = "✅ ALERT - Vozač je budan"
    else:
        rezultat = "🚨 UMORAN - Vozač je umoran!"

    ear_mar_info = f"EAR: {ear:.4f} | MAR: {mar:.4f}"
    proba_info = f"Vjerovatnoća: Alert={proba[0]*100:.1f}% | Umoran={proba[1]*100:.1f}%"

    return rezultat, ear_mar_info, proba_info

# ============================================================
# TAB 3 - CNN
# ============================================================

def predict_cnn(image):
    if image is None:
        return "❌ Nisi uploadala sliku!", ""

    # Pripremi sliku za CNN
    img = np.array(image)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predikcija
    proba = cnn_model.predict(img, verbose=0)[0]
    klasa_idx = np.argmax(proba)
    klasa = KLASE_CNN[klasa_idx]
    confidence = proba[klasa_idx] * 100

    if klasa == 'alert':
        rezultat = f"✅ ALERT - Vozač je budan ({confidence:.1f}%)"
    elif klasa == 'microsleep':
        rezultat = f"🚨 MICROSLEEP - Vozač spava! ({confidence:.1f}%)"
    else:
        rezultat = f"😴 YAWNING - Vozač zijeva ({confidence:.1f}%)"

    proba_info = f"Alert: {proba[0]*100:.1f}% | Microsleep: {proba[1]*100:.1f}% | Yawning: {proba[2]*100:.1f}%"

    return rezultat, proba_info

# ============================================================
# TAB 4 - KAMERA UŽIVO
# ============================================================

def predict_kamera(image):
    global umoran_od
    
    if image is None:
        return None, "⏳ Čekam kameru..."

    img_rgb = np.array(image)
    processed = img_rgb.copy()
    h, w = img_rgb.shape[:2]

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    results = face_mesh.detect(mp_image)

    if not results.face_landmarks:
        return processed, "❌ Lice nije detektovano!"

    landmarks = results.face_landmarks[0]
    ear_left = compute_ear(landmarks, LEFT_EYE, w, h)
    ear_right = compute_ear(landmarks, RIGHT_EYE, w, h)
    ear = (ear_left + ear_right) / 2.0
    mar = compute_mar(landmarks, MOUTH, w, h)

    features = scaler.transform([[ear, mar]])
    prediction = rf_model.predict(features)[0]

    # Logika za 3 sekunde
    if prediction == 1:
        if umoran_od is None:
            umoran_od = time.time()
        trajanje = time.time() - umoran_od

        if trajanje >= PRAG_SEKUNDI:
            boja = (255, 0, 0)
            tekst = "UMORAN!"
            status = f"🚨 UMORAN {trajanje:.0f}s! | EAR: {ear:.3f} | MAR: {mar:.3f}"
            os.system("afplay /System/Library/Sounds/Sosumi.aiff &")
        else:
            boja = (0, 165, 255)
            tekst = f"Provjera... {trajanje:.1f}s"
            status = f"⚠️ Moguć umor ({trajanje:.1f}s) | EAR: {ear:.3f}"
    else:
        umoran_od = None
        boja = (0, 255, 0)
        tekst = "ALERT"
        status = f"✅ ALERT | EAR: {ear:.3f} | MAR: {mar:.3f}"

    # Box oko lica
    xs = [int(lm.x * w) for lm in landmarks]
    ys = [int(lm.y * h) for lm in landmarks]
    fx1, fy1 = max(0, min(xs)-20), max(0, min(ys)-20)
    fx2, fy2 = min(w, max(xs)+20), min(h, max(ys)+20)
    cv2.rectangle(processed, (fx1, fy1), (fx2, fy2), boja, 2)
    cv2.putText(processed, tekst, (fx1, fy1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, boja, 2)

    # Box oko lijevog oka
    lx1 = min(int(landmarks[i].x * w) for i in LEFT_EYE) - 8
    ly1 = min(int(landmarks[i].y * h) for i in LEFT_EYE) - 8
    lx2 = max(int(landmarks[i].x * w) for i in LEFT_EYE) + 8
    ly2 = max(int(landmarks[i].y * h) for i in LEFT_EYE) + 8
    cv2.rectangle(processed, (lx1, ly1), (lx2, ly2), boja, 2)
    cv2.putText(processed, f"L:{ear_left:.2f}", (lx1, ly1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, boja, 1)

    # Box oko desnog oka
    rx1 = min(int(landmarks[i].x * w) for i in RIGHT_EYE) - 8
    ry1 = min(int(landmarks[i].y * h) for i in RIGHT_EYE) - 8
    rx2 = max(int(landmarks[i].x * w) for i in RIGHT_EYE) + 8
    ry2 = max(int(landmarks[i].y * h) for i in RIGHT_EYE) + 8
    cv2.rectangle(processed, (rx1, ry1), (rx2, ry2), boja, 2)
    cv2.putText(processed, f"R:{ear_right:.2f}", (rx1, ry1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, boja, 1)

    # Box oko usta
    mx1 = min(int(landmarks[i].x * w) for i in MOUTH) - 8
    my1 = min(int(landmarks[i].y * h) for i in MOUTH) - 8
    mx2 = max(int(landmarks[i].x * w) for i in MOUTH) + 8
    my2 = max(int(landmarks[i].y * h) for i in MOUTH) + 8
    cv2.rectangle(processed, (mx1, my1), (mx2, my2), boja, 2)
    cv2.putText(processed, f"MAR:{mar:.2f}", (mx1, my2+15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, boja, 1)

    return processed, status

# ============================================================
# GRADIO INTERFEJS
# ============================================================

with gr.Blocks(title="Driver Monitoring System") as app:

    gr.Markdown("**Driver Monitoring System**")
    gr.Markdown("Sistem za detekciju umora vozača koristeći Machine Learning")

    with gr.Tabs():

        # ---- TAB 1: Logistička Regresija ----
        with gr.Tab("📈 Logistička Regresija"):
            gr.Markdown("### Logistička Regresija model")
            gr.Markdown("Koristi **MediaPipe** za izvlačenje EAR i MAR vrijednosti, pa ih klasificira pomoću Logističke Regresije.")

            with gr.Row():
                with gr.Column():
                    lr_input = gr.Image(label="Uploadaj sliku vozača", type="pil")
                    lr_button = gr.Button("🔍 Analiziraj", variant="primary")

                with gr.Column():
                    lr_rezultat = gr.Textbox(label="Rezultat", lines=2)
                    lr_ear_mar = gr.Textbox(label="EAR i MAR vrijednosti")
                    lr_proba = gr.Textbox(label="Vjerovatnoće")

            lr_button.click(
                fn=predict_lr,
                inputs=lr_input,
                outputs=[lr_rezultat, lr_ear_mar, lr_proba]
            )

        # ---- TAB 2: Random Forest ----
        with gr.Tab("🌲 Random Forest"):
            gr.Markdown("### Random Forest model")
            gr.Markdown("Koristi **MediaPipe** za izvlačenje EAR i MAR vrijednosti, pa ih klasificira pomoću Random Forest.")

            with gr.Row():
                with gr.Column():
                    rf_input = gr.Image(label="Uploadaj sliku vozača", type="pil")
                    rf_button = gr.Button("🔍 Analiziraj", variant="primary")

                with gr.Column():
                    rf_rezultat = gr.Textbox(label="Rezultat", lines=2)
                    rf_ear_mar = gr.Textbox(label="EAR i MAR vrijednosti")
                    rf_proba = gr.Textbox(label="Vjerovatnoće")

            rf_button.click(
                fn=predict_rf,
                inputs=rf_input,
                outputs=[rf_rezultat, rf_ear_mar, rf_proba]
            )

        # ---- TAB 3: CNN ----
        with gr.Tab("🧠 CNN"):
            gr.Markdown("### CNN model")
            gr.Markdown("Koristi **Convolutional Neural Network** koji direktno analizira piksele slike.")

            with gr.Row():
                with gr.Column():
                    cnn_input = gr.Image(label="Uploadaj sliku vozača", type="pil")
                    cnn_button = gr.Button("🔍 Analiziraj", variant="primary")

                with gr.Column():
                    cnn_rezultat = gr.Textbox(label="Rezultat", lines=2)
                    cnn_proba = gr.Textbox(label="Vjerovatnoće po klasama")

            cnn_button.click(
                fn=predict_cnn,
                inputs=cnn_input,
                outputs=[cnn_rezultat, cnn_proba]
            )

        # ---- TAB 4: Kamera uživo ----
        with gr.Tab("📷 Kamera uživo"):
            gr.Markdown("### Kamera uživo")

            with gr.Row():

                with gr.Column(scale=1):

                    kamera_input = gr.Image(
                        sources=["webcam"],
                        streaming=True,
                        type="numpy",
                        height=250,
                    )

                    gr.Markdown("### 📊 Rezultati")
                    kamera_status = gr.Textbox(label="Status vozača", lines=3)

                with gr.Column(scale=2):
                    kamera_output = gr.Image(
                        show_label=False,
                        height=480
                    )
                    
            kamera_input.stream(
                fn=predict_kamera,
                inputs=kamera_input,
                outputs=[kamera_output, kamera_status]
            )

    gr.Markdown("---")
    gr.Markdown("**Projekat:** Driver Monitoring System | **Model LR Accuracy:** 85.94% | **Model RF Accuracy:** 87.21% | **Model CNN Accuracy:** 97.08%")

app.launch()