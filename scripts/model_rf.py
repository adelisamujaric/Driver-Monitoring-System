import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ============================================================
# 1. UČITAVANJE ČISTOG DATASETA
# ============================================================

df = pd.read_csv("/Users/adelisa/FIT/ML/outputs/features_clean.csv")

print("=" * 50)
print("DATASET INFO")
print("=" * 50)
print(df.info())
print()
print(df.head())
print()
print("Distribucija klasa:")
print(df['label'].value_counts())

# ============================================================
# 2. PRIPREMA PODATAKA
# ============================================================

# Features (ulazne varijable)
X = df[['ear', 'mar']]

# Target (šta model treba predvidjeti)
y = df['label']

print("\n" + "=" * 50)
print("PRIPREMA PODATAKA")
print("=" * 50)
print(f"Broj uzoraka: {len(df)}")
print(f"Features: {list(X.columns)}")
print(f"Klase: {y.unique()}")

# ============================================================
# 3. TRAIN/TEST SPLIT (80% trening, 20% testiranje)
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrening set: {len(X_train)} uzoraka")
print(f"Test set:    {len(X_test)} uzoraka")

print("\nDistribucija klasa u test setu:")
print(y_test.value_counts())

# ============================================================
# 4. SKALIRANJE PODATAKA
# ============================================================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================================================
# 5. TRENIRANJE MODELA
# ============================================================

print("\n" + "=" * 50)
print("TRENIRANJE RANDOM FOREST MODELA")
print("=" * 50)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Model uspješno istreniran!")
print(f"Broj stabala: {model.n_estimators}")

# ============================================================
# 6. EVALUACIJA MODELA
# ============================================================

y_pred = model.predict(X_test)

print("\n" + "=" * 50)
print("EVALUACIJA MODELA")
print("=" * 50)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['alert', 'umoran']))

# ============================================================
# 7. CONFUSION MATRIX
# ============================================================

matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(matrix)

plt.figure(figsize=(7, 6))
sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['alert', 'umoran'],
            yticklabels=['alert', 'umoran'])
plt.title(f"Confusion Matrix - Random Forest\nAccuracy: {accuracy*100:.2f}%")
plt.xlabel("Predviđeno")
plt.ylabel("Stvarno")
plt.tight_layout()
plt.savefig("/Users/adelisa/FIT/ML/outputs/confusion_matrix_rf.png")
plt.show()

# ============================================================
# 8. VAŽNOST FEATURESOVA
# ============================================================

feature_importance = model.feature_importances_
features = ['EAR', 'MAR']

print("\n" + "=" * 50)
print("VAŽNOST FEATURESA")
print("=" * 50)
for f, imp in zip(features, feature_importance):
    print(f"{f}: {imp:.4f} ({imp*100:.2f}%)")

plt.figure(figsize=(6, 4))
plt.bar(features, feature_importance, color=['steelblue', 'coral'])
plt.title("Važnost featuresa - Random Forest")
plt.xlabel("Feature")
plt.ylabel("Važnost")
plt.ylim(0, 1)
for i, v in enumerate(feature_importance):
    plt.text(i, v + 0.01, f"{v*100:.1f}%", ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig("/Users/adelisa/FIT/ML/outputs/feature_importance_rf.png")
plt.show()

import pickle
with open("/Users/adelisa/FIT/ML/outputs/driver_model_rf.pkl", "wb") as f:
    pickle.dump((model, scaler), f)
print("RF model snimljen!")

print("\n" + "=" * 50)
print("ZAVRŠENO!")
print(f"Accuracy: {accuracy*100:.2f}%")
print("Grafovi snimljeni u /Users/adelisa/FIT/ML/outputs/")
print("=" * 50)