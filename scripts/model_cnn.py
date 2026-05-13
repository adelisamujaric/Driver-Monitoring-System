import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ============================================================
# 1. POSTAVKE
# ============================================================

DATASET_DIR = "/Users/adelisa/FIT/ML/dataset_organized"
IMG_SIZE = 64
MAX_SLIKE_PO_KLASI = 2000
KLASE = ['alert', 'microsleep', 'yawning']
EPOCHS = 10
BATCH_SIZE = 32

print("=" * 50)
print("CNN MODEL - Driver Monitoring System")
print("=" * 50)
print(f"Veličina slike: {IMG_SIZE}x{IMG_SIZE}")
print(f"Slika po klasi: {MAX_SLIKE_PO_KLASI}")
print(f"Klase: {KLASE}")

# ============================================================
# 2. UČITAVANJE SLIKA
# ============================================================

print("\n" + "=" * 50)
print("UČITAVANJE SLIKA")
print("=" * 50)

X = []
y = []

for label, klasa in enumerate(KLASE):
    folder = os.path.join(DATASET_DIR, klasa)
    slike = os.listdir(folder)[:MAX_SLIKE_PO_KLASI]

    print(f"Učitavam {klasa}: {len(slike)} slika...")

    for slika_naziv in slike:
        putanja = os.path.join(folder, slika_naziv)
        try:
            slika = load_img(putanja, target_size=(IMG_SIZE, IMG_SIZE))
            slika_array = img_to_array(slika) / 255.0  # normalizacija
            X.append(slika_array)
            y.append(label)
        except Exception as e:
            pass

X = np.array(X)
y = np.array(y)

print(f"\nUkupno učitano slika: {len(X)}")
print(f"Shape X: {X.shape}")
print(f"Distribucija klasa: {np.bincount(y)}")

# ============================================================
# 3. TRAIN/TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n" + "=" * 50)
print("TRAIN/TEST SPLIT")
print("=" * 50)
print(f"Trening set: {len(X_train)} slika")
print(f"Test set:    {len(X_test)} slika")

# One-hot encoding za CNN
y_train_cat = to_categorical(y_train, num_classes=3)
y_test_cat = to_categorical(y_test, num_classes=3)

# ============================================================
# 4. CNN ARHITEKTURA
# ============================================================

print("\n" + "=" * 50)
print("CNN ARHITEKTURA")
print("=" * 50)

model = Sequential([
    # Prvi konvolucijski blok
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),

    # Drugi konvolucijski blok
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # Treći konvolucijski blok
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # Fully connected slojevi
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 klase
])

model.summary()

# ============================================================
# 5. KOMPAJLIRANJE MODELA
# ============================================================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ============================================================
# 6. TRENIRANJE
# ============================================================

print("\n" + "=" * 50)
print("TRENIRANJE CNN MODELA")
print("=" * 50)

history = model.fit(
    X_train, y_train_cat,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    verbose=1
)

# ============================================================
# 7. EVALUACIJA
# ============================================================

print("\n" + "=" * 50)
print("EVALUACIJA MODELA")
print("=" * 50)

y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=KLASE))

# ============================================================
# 8. GRAFOVI
# ============================================================

# Accuracy i Loss tokom treniranja
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("CNN Treniranje", fontsize=14)

axes[0].plot(history.history['accuracy'], label='Train Accuracy')
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
axes[0].set_title("Accuracy tokom treniranja")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()

axes[1].plot(history.history['loss'], label='Train Loss')
axes[1].plot(history.history['val_loss'], label='Val Loss')
axes[1].set_title("Loss tokom treniranja")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()

plt.tight_layout()
plt.savefig("/Users/adelisa/FIT/ML/outputs/cnn_treniranje.png")
plt.show()

# Confusion Matrix
matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=KLASE,
            yticklabels=KLASE)
plt.title(f"Confusion Matrix - CNN\nAccuracy: {accuracy*100:.2f}%")
plt.xlabel("Predviđeno")
plt.ylabel("Stvarno")
plt.tight_layout()
plt.savefig("/Users/adelisa/FIT/ML/outputs/confusion_matrix_cnn.png")
plt.show()

# ============================================================
# 9. SNIMANJE MODELA
# ============================================================

model.save("/Users/adelisa/FIT/ML/outputs/driver_model_cnn.h5")
print("\nCNN model snimljen u: /Users/adelisa/FIT/ML/outputs/driver_model_cnn.h5")

print("\n" + "=" * 50)
print("ZAVRŠENO!")
print(f"CNN Accuracy: {accuracy*100:.2f}%")
print("=" * 50)