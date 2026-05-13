import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# 1. UČITAVANJE PODATAKA
# ============================================================

df = pd.read_csv("/Users/adelisa/FIT/ML/outputs/features.csv")

print("=" * 50)
print("OSNOVNE INFORMACIJE O DATASETU")
print("=" * 50)
print(df.info())
print()
print(df.head())


# ============================================================
# 2. DESKRIPTIVNA STATISTIKA (PRIJE ČIŠĆENJA)
# ============================================================

print("\n" + "=" * 50)
print("DESKRIPTIVNA STATISTIKA (PRIJE ČIŠĆENJA)")
print("=" * 50)

print(f"\nEAR - mean:   {df['ear'].mean():.4f}")
print(f"EAR - median: {df['ear'].median():.4f}")
print(f"EAR - std:    {df['ear'].std():.4f}")
print(f"EAR - min:    {df['ear'].min():.4f}")
print(f"EAR - max:    {df['ear'].max():.4f}")

print(f"\nMAR - mean:   {df['mar'].mean():.4f}")
print(f"MAR - median: {df['mar'].median():.4f}")
print(f"MAR - std:    {df['mar'].std():.4f}")
print(f"MAR - min:    {df['mar'].min():.4f}")
print(f"MAR - max:    {df['mar'].max():.4f}")

# Kvantili
q1_ear = df['ear'].quantile(0.25)
q2_ear = df['ear'].quantile(0.50)
q3_ear = df['ear'].quantile(0.75)
iqr_ear = q3_ear - q1_ear

print(f"\nEAR kvantili -> Q1: {q1_ear:.4f}, Q2: {q2_ear:.4f}, Q3: {q3_ear:.4f}, IQR: {iqr_ear:.4f}")

q1_mar = df['mar'].quantile(0.25)
q2_mar = df['mar'].quantile(0.50)
q3_mar = df['mar'].quantile(0.75)
iqr_mar = q3_mar - q1_mar

print(f"MAR kvantili -> Q1: {q1_mar:.4f}, Q2: {q2_mar:.4f}, Q3: {q3_mar:.4f}, IQR: {iqr_mar:.4f}")

# Distribucija klasa
print("\n" + "=" * 50)
print("DISTRIBUCIJA KLASA (driver_state)")
print("=" * 50)
print(df['driver_state'].value_counts())
print()
print(df['driver_state'].value_counts(normalize=True))


# ============================================================
# 3. VIZUALIZACIJA PRIJE ČIŠĆENJA
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Distribucija EAR i MAR - PRIJE čišćenja", fontsize=14)

axes[0].hist(df['ear'], bins=50, color='steelblue', edgecolor='black')
axes[0].axvline(q1_ear, color='r', linestyle='--', label=f'Q1={q1_ear:.2f}')
axes[0].axvline(q2_ear, color='g', linestyle='--', label=f'Q2={q2_ear:.2f}')
axes[0].axvline(q3_ear, color='r', linestyle='--', label=f'Q3={q3_ear:.2f}')
axes[0].set_title("EAR distribucija")
axes[0].set_xlabel("EAR vrijednost")
axes[0].set_ylabel("Broj uzoraka")
axes[0].legend()

axes[1].hist(df['mar'], bins=50, color='coral', edgecolor='black')
axes[1].axvline(q1_mar, color='r', linestyle='--', label=f'Q1={q1_mar:.2f}')
axes[1].axvline(q2_mar, color='g', linestyle='--', label=f'Q2={q2_mar:.2f}')
axes[1].axvline(q3_mar, color='r', linestyle='--', label=f'Q3={q3_mar:.2f}')
axes[1].set_title("MAR distribucija")
axes[1].set_xlabel("MAR vrijednost")
axes[1].set_ylabel("Broj uzoraka")
axes[1].legend()

plt.tight_layout()
plt.savefig("/Users/adelisa/FIT/ML/outputs/histogram_prije_ciscenja.png")
plt.show()

# Pie chart - distribucija klasa
klase = df['driver_state'].value_counts()
plt.figure(figsize=(7, 7))
plt.pie(klase, labels=klase.index, autopct='%1.1f%%', colors=['green', 'red', 'orange'])
plt.title("Distribucija klasa - PRIJE čišćenja")
plt.savefig("/Users/adelisa/FIT/ML/outputs/pie_chart_klase_prije.png")
plt.show()

# Bar chart - distribucija klasa
plt.figure(figsize=(7, 5))
plt.bar(klase.index, klase.values, color=['green', 'red', 'orange'])
plt.title("Broj uzoraka po klasi - PRIJE čišćenja")
plt.xlabel("Klasa")
plt.ylabel("Broj uzoraka")
plt.savefig("/Users/adelisa/FIT/ML/outputs/bar_chart_klase_prije.png")
plt.show()


# ============================================================
# 4. ČIŠĆENJE PODATAKA - uklanjanje outlitera
# ============================================================

print("\n" + "=" * 50)
print("ČIŠĆENJE PODATAKA")
print("=" * 50)

print(f"\nBroj uzoraka PRIJE čišćenja: {len(df)}")
print(f"Outlieri EAR >= 0.6: {(df['ear'] >= 0.6).sum()}")
print(f"Outlieri MAR >= 4.0: {(df['mar'] >= 4.0).sum()}")

# Filtriranje outlitera
df_clean = df[(df['ear'] < 0.6) & (df['mar'] < 4.0)].copy()

print(f"\nBroj uzoraka NAKON čišćenja: {len(df_clean)}")
print(f"Uklonjeno uzoraka: {len(df) - len(df_clean)}")


# ============================================================
# 5. DESKRIPTIVNA STATISTIKA (NAKON ČIŠĆENJA)
# ============================================================

print("\n" + "=" * 50)
print("DESKRIPTIVNA STATISTIKA (NAKON ČIŠĆENJA)")
print("=" * 50)

print(f"\nEAR - mean:   {df_clean['ear'].mean():.4f}")
print(f"EAR - median: {df_clean['ear'].median():.4f}")
print(f"EAR - std:    {df_clean['ear'].std():.4f}")
print(f"EAR - min:    {df_clean['ear'].min():.4f}")
print(f"EAR - max:    {df_clean['ear'].max():.4f}")

print(f"\nMAR - mean:   {df_clean['mar'].mean():.4f}")
print(f"MAR - median: {df_clean['mar'].median():.4f}")
print(f"MAR - std:    {df_clean['mar'].std():.4f}")
print(f"MAR - min:    {df_clean['mar'].min():.4f}")
print(f"MAR - max:    {df_clean['mar'].max():.4f}")

# Distribucija klasa nakon čišćenja
print("\nDistribucija klasa NAKON čišćenja:")
print(df_clean['driver_state'].value_counts())
print()
print(df_clean['driver_state'].value_counts(normalize=True))


# ============================================================
# 6. VIZUALIZACIJA NAKON ČIŠĆENJA
# ============================================================

q1_ear_c = df_clean['ear'].quantile(0.25)
q2_ear_c = df_clean['ear'].quantile(0.50)
q3_ear_c = df_clean['ear'].quantile(0.75)

q1_mar_c = df_clean['mar'].quantile(0.25)
q2_mar_c = df_clean['mar'].quantile(0.50)
q3_mar_c = df_clean['mar'].quantile(0.75)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Distribucija EAR i MAR - NAKON čišćenja", fontsize=14)

axes[0].hist(df_clean['ear'], bins=50, color='steelblue', edgecolor='black')
axes[0].axvline(q1_ear_c, color='r', linestyle='--', label=f'Q1={q1_ear_c:.2f}')
axes[0].axvline(q2_ear_c, color='g', linestyle='--', label=f'Q2={q2_ear_c:.2f}')
axes[0].axvline(q3_ear_c, color='r', linestyle='--', label=f'Q3={q3_ear_c:.2f}')
axes[0].set_title("EAR distribucija")
axes[0].set_xlabel("EAR vrijednost")
axes[0].set_ylabel("Broj uzoraka")
axes[0].legend()

axes[1].hist(df_clean['mar'], bins=50, color='coral', edgecolor='black')
axes[1].axvline(q1_mar_c, color='r', linestyle='--', label=f'Q1={q1_mar_c:.2f}')
axes[1].axvline(q2_mar_c, color='g', linestyle='--', label=f'Q2={q2_mar_c:.2f}')
axes[1].axvline(q3_mar_c, color='r', linestyle='--', label=f'Q3={q3_mar_c:.2f}')
axes[1].set_title("MAR distribucija")
axes[1].set_xlabel("MAR vrijednost")
axes[1].set_ylabel("Broj uzoraka")
axes[1].legend()

plt.tight_layout()
plt.savefig("/Users/adelisa/FIT/ML/outputs/histogram_nakon_ciscenja.png")
plt.show()

# Pie chart - distribucija klasa NAKON čišćenja
klase_clean = df_clean['driver_state'].value_counts()
plt.figure(figsize=(7, 7))
plt.pie(klase_clean, labels=klase_clean.index, autopct='%1.1f%%', colors=['green', 'red', 'orange'])
plt.title("Distribucija klasa - NAKON čišćenja")
plt.savefig("/Users/adelisa/FIT/ML/outputs/pie_chart_klase_nakon.png")
plt.show()

# Bar chart - distribucija klasa NAKON čišćenja
plt.figure(figsize=(7, 5))
plt.bar(klase_clean.index, klase_clean.values, color=['green', 'red', 'orange'])
plt.title("Broj uzoraka po klasi - NAKON čišćenja")
plt.xlabel("Klasa")
plt.ylabel("Broj uzoraka")
plt.savefig("/Users/adelisa/FIT/ML/outputs/bar_chart_klase_nakon.png")
plt.show()


# ============================================================
# 7. SCATTER PLOT - EAR vs MAR po klasama
# ============================================================

colours = {'alert': 'green', 'microsleep': 'red', 'yawning': 'orange'}

plt.figure(figsize=(9, 6))
for klasa, boja in colours.items():
    subset = df_clean[df_clean['driver_state'] == klasa]
    plt.scatter(subset['ear'], subset['mar'], c=boja, label=klasa, alpha=0.4, s=10)

plt.title("Scatter plot: EAR vs MAR po klasama")
plt.xlabel("EAR (Eye Aspect Ratio)")
plt.ylabel("MAR (Mouth Aspect Ratio)")
plt.legend()
plt.savefig("/Users/adelisa/FIT/ML/outputs/scatter_ear_mar.png")
plt.show()

# ============================================================
# 8. SNIMANJE ČISTOG DATASETA
# ============================================================

df_clean.to_csv("/Users/adelisa/FIT/ML/outputs/features_clean.csv", index=False)
print("\nČist dataset snimljen u: /Users/adelisa/FIT/ML/outputs/features_clean.csv")
print("\nEDA završena! Svi grafovi snimljeni u /Users/adelisa/FIT/ML/outputs/")