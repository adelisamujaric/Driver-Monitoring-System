import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ============================================================
# 1. UČITAVANJE ČISTOG DATASETA
# ============================================================

df = pd.read_csv("/Users/adelisa/FIT/ML/outputs/features_clean.csv")

print("=" * 50)
print("LOGISTIČKA REGRESIJA - Driver Monitoring")
print("=" * 50)
print(f"Broj uzoraka: {len(df)}")
print(f"Distribucija klasa:")
print(df['label'].value_counts())

# ============================================================
# 2. PRIPREMA PODATAKA
# ============================================================

X = df[['ear', 'mar']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrening set: {len(X_train)} uzoraka")
print(f"Test set:    {len(X_test)} uzoraka")

# ============================================================
# 3. SKALIRANJE
# ============================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 4. TRENIRANJE MODELA
# ============================================================

print("\n" + "=" * 50)
print("TRENIRANJE LOGISTIČKE REGRESIJE")
print("=" * 50)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

print("Model uspješno istreniran!")
print(f"\nKoeficijenti (težine):")
print(f"  EAR težina: {model.coef_[0][0]:.4f}")
print(f"  MAR težina: {model.coef_[0][1]:.4f}")
print(f"  Bias (b):   {model.intercept_[0]:.4f}")

# ============================================================
# 5. EVALUACIJA
# ============================================================

y_pred = model.predict(X_test_scaled)

print("\n" + "=" * 50)
print("EVALUACIJA MODELA")
print("=" * 50)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['alert', 'umoran']))

# ============================================================
# 6. CONFUSION MATRIX
# ============================================================

matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 6))
sns.heatmap(matrix, annot=True, fmt='d', cmap='Greens',
            xticklabels=['alert', 'umoran'],
            yticklabels=['alert', 'umoran'])
plt.title(f"Confusion Matrix - Logistička Regresija\nAccuracy: {accuracy*100:.2f}%")
plt.xlabel("Predviđeno")
plt.ylabel("Stvarno")
plt.tight_layout()
plt.savefig("/Users/adelisa/FIT/ML/outputs/confusion_matrix_lr.png")
plt.show()

# ============================================================
# 7. VIZUALIZACIJA DECISION BOUNDARY
# ============================================================

x_min, x_max = X_test_scaled[:, 0].min() - 1, X_test_scaled[:, 0].max() + 1
y_min, y_max = X_test_scaled[:, 1].min() - 1, X_test_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(9, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlGn')

boje = {0: 'green', 1: 'red'}
oznake = {0: 'alert', 1: 'umoran'}
for klasa in [0, 1]:
    mask = y_test == klasa
    plt.scatter(X_test_scaled[mask, 0], X_test_scaled[mask, 1],
                c=boje[klasa], label=oznake[klasa], alpha=0.6, s=20)

plt.title("Decision Boundary - Logistička Regresija")
plt.xlabel("EAR (skaliran)")
plt.ylabel("MAR (skaliran)")
plt.legend()
plt.tight_layout()
plt.savefig("/Users/adelisa/FIT/ML/outputs/decision_boundary_lr.png")
plt.show()

# ============================================================
# 8. POREĐENJE MODELA
# ============================================================

print("\n" + "=" * 50)
print("POREĐENJE MODELA")
print("=" * 50)
print(f"Logistička regresija: {accuracy*100:.2f}%")
print(f"Random Forest:         87.21%")
print(f"CNN:                   97.08%")

# ============================================================
# 9. SNIMANJE MODELA
# ============================================================

with open("/Users/adelisa/FIT/ML/outputs/driver_model_lr.pkl", "wb") as f:
    pickle.dump((model, scaler), f)

print("\nLR model snimljen u: /Users/adelisa/FIT/ML/outputs/driver_model_lr.pkl")
print("\n" + "=" * 50)
print("ZAVRŠENO!")
print("=" * 50)