"""
Tugas Machine Learning - Klasifikasi Data Tidak Seimbang (Imbalanced Data)
Kasus   : Deteksi kanker payudara (ganas vs jinak)
Dataset : Breast Cancer Wisconsin (bawaan scikit-learn, sumber publik UCI)
Metode  : Decision Tree (baseline) vs Random Forest
Topik   : Imbalanced data, akurasi vs precision/recall/F1, confusion matrix,
          overfitting (pembatasan kompleksitas) + oversampling (opsional)

Instal:  pip install scikit-learn pandas matplotlib seaborn
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # hapus baris ini jika ingin tampil di jendela
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, recall_score)
from sklearn.utils import resample

RANDOM_STATE = 42

# ---------------------------------------------------------------
# A. DATASET
# ---------------------------------------------------------------
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target  # 0 = ganas (malignant), 1 = jinak (benign)

# Dataset asli cukup seimbang (37% ganas). Agar kasusnya imbalanced,
# kelas ganas (minoritas) kita kurangi sampai sekitar 10% dari data.
ganas = df[df.target == 0].sample(n=40, random_state=RANDOM_STATE)
jinak = df[df.target == 1]
df = pd.concat([ganas, jinak]).sample(frac=1, random_state=RANDOM_STATE)

print("=== Distribusi kelas ===")
print(df.target.value_counts().rename({0: "Ganas (minoritas)", 1: "Jinak (mayoritas)"}))
print(f"Persentase ganas: {(df.target == 0).mean() * 100:.1f}%\n")

# ---------------------------------------------------------------
# B. PREPROCESSING
# ---------------------------------------------------------------
print("Missing value:", df.isnull().sum().sum())
X = df.drop(columns="target")
y = df["target"]

# stratify menjaga proporsi kelas di data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)   # fit hanya pada data latih
X_test_s = scaler.transform(X_test)


def evaluasi(nama, model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    acc_train = accuracy_score(ytr, model.predict(Xtr))
    acc_test = accuracy_score(yte, pred)
    print(f"\n===== {nama} =====")
    print(f"Akurasi latih: {acc_train:.3f} | Akurasi uji: {acc_test:.3f}")
    print(classification_report(yte, pred, target_names=["Ganas", "Jinak"],
                                zero_division=0))
    cm = confusion_matrix(yte, pred)
    return {
        "nama": nama, "akurasi": acc_test,
        "recall_ganas": recall_score(yte, pred, pos_label=0),
        "f1_ganas": f1_score(yte, pred, pos_label=0),
        "cm": cm, "gap": acc_train - acc_test,
    }


# ---------------------------------------------------------------
# C. TRAINING & EVALUASI
# ---------------------------------------------------------------
hasil = []

# 0) Pembanding "naif": selalu menebak kelas mayoritas
naif = np.ones_like(y_test)
print("=== Model naif (selalu prediksi 'Jinak') ===")
print(f"Akurasi: {accuracy_score(y_test, naif):.3f} "
      f"| Recall ganas: {recall_score(y_test, naif, pos_label=0, zero_division=0):.3f}")
print("-> Akurasi tampak tinggi padahal tidak ada kasus ganas yang terdeteksi.")

# 1) Decision Tree tanpa batasan (rawan overfitting)
hasil.append(evaluasi("Decision Tree (tanpa batas)",
                      DecisionTreeClassifier(random_state=RANDOM_STATE),
                      X_train_s, y_train, X_test_s, y_test))

# 2) Decision Tree dengan pembatasan kompleksitas + class_weight
hasil.append(evaluasi("Decision Tree (max_depth=3, class_weight=balanced)",
                      DecisionTreeClassifier(max_depth=3, min_samples_leaf=3,
                                             class_weight="balanced",
                                             random_state=RANDOM_STATE),
                      X_train_s, y_train, X_test_s, y_test))

# 3) Random Forest dengan oversampling kelas minoritas (hanya pada data latih)
latih = pd.DataFrame(X_train_s, columns=X.columns)
latih["target"] = y_train.values
min_df = latih[latih.target == 0]
maj_df = latih[latih.target == 1]
min_up = resample(min_df, replace=True, n_samples=len(maj_df),
                  random_state=RANDOM_STATE)
latih_bal = pd.concat([maj_df, min_up])
hasil.append(evaluasi("Random Forest + Oversampling",
                      RandomForestClassifier(n_estimators=200, max_depth=5,
                                             random_state=RANDOM_STATE),
                      latih_bal.drop(columns="target"), latih_bal.target,
                      X_test_s, y_test))

# ---------------------------------------------------------------
# D. VISUALISASI & RINGKASAN
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, len(hasil), figsize=(5 * len(hasil), 4))
for ax, h in zip(axes, hasil):
    sns.heatmap(h["cm"], annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                xticklabels=["Ganas", "Jinak"], yticklabels=["Ganas", "Jinak"])
    ax.set_title(h["nama"], fontsize=9)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("\nGambar confusion matrix disimpan: confusion_matrix.png")

ringkas = pd.DataFrame([{k: v for k, v in h.items() if k != "cm"} for h in hasil])
print("\n=== Ringkasan perbandingan ===")
print(ringkas.round(3).to_string(index=False))

print("""
=== Kesimpulan singkat (sesuaikan dengan hasil yang muncul) ===
1. Pada data tidak seimbang, akurasi saja menyesatkan: model yang selalu
   menebak 'Jinak' tetap berakurasi tinggi tetapi recall kelas ganas = 0.
2. Recall dan F1 kelas minoritas lebih tepat dipakai, apalagi pada kasus
   medis di mana melewatkan kasus ganas (False Negative) sangat berbahaya.
3. Decision Tree tanpa batas cenderung overfitting (akurasi latih jauh di atas
   akurasi uji); max_depth dan min_samples_leaf membantu mengurangi hal ini.
4. Oversampling/class_weight membantu model lebih peka terhadap kelas minoritas.
""")