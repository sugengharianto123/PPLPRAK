# Import library dan package yang diperlukan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Membaca dataset
df = pd.read_csv('heart.csv')
print("Menampilkan dataset:")
print(df)
print("==========================================================")

# Memeriksa nilai yang hilang
print("Memeriksa nilai yang hilang dalam dataset:")
print(df.isnull().sum())
print("==========================================================")

# Memeriksa baris duplikat
print("Memeriksa baris duplikat dalam dataset:")
print(df.duplicated().sum())
print("==========================================================")

# Mengubah variabel kategorikal menjadi numerik
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['ChestPainType'] = le.fit_transform(df['ChestPainType'])
df['RestingECG'] = le.fit_transform(df['RestingECG'])
df['ExerciseAngina'] = le.fit_transform(df['ExerciseAngina'])
df['ST_Slope'] = le.fit_transform(df['ST_Slope'])

# Menampilkan dataset setelah diubah menjadi numerik
print("Dataset setelah variabel kategorikal diubah menjadi numerik:")
print(df.head())
print("==========================================================")

# Melihat statistik deskriptif dari dataset
print("Statistik deskriptif dari dataset:")
print(df.describe())
print("==========================================================")

# Analisis Data Eksploratif (EDA)
sns.countplot(x='HeartDisease', data=df)
plt.title('Distribusi Kejadian Gagal Jantung')
plt.show()

# Menghitung distribusi kejadian gagal jantung
heart_disease_counts = df['HeartDisease'].value_counts()
print("Distribusi Kejadian Gagal Jantung dalam Jumlah Angka:")
print(heart_disease_counts)

# Visualisasi korelasi dengan heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Heatmap Korelasi')
plt.show()

# Eliminasi variabel dengan korelasi tinggi
threshold = 0.8
corr_matrix = df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
df_reduced = df.drop(columns=to_drop)
print("Dataset setelah eliminasi variabel dengan korelasi tinggi:")
print(df_reduced.head())
print("==========================================================")

# Membagi data menjadi fitur dan variabel target
X = df_reduced.drop(columns='HeartDisease')
y = df_reduced['HeartDisease']

# Membagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Penskalaan fitur
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter tuning untuk Decision Tree
param_grid_dtree = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30]
}
grid_search_dtree = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dtree, cv=5)
grid_search_dtree.fit(X_train, y_train)

# Hyperparameter tuning untuk SVM
param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly']
}
grid_search_svm = GridSearchCV(SVC(random_state=42), param_grid_svm, cv=5)
grid_search_svm.fit(X_train, y_train)

# Function to evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'Akurasi': accuracy_score(y_test, y_pred),
        'Presisi': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'Skor F1': f1_score(y_test, y_pred)
    }

# Evaluasi semua kombinasi parameter untuk Decision Tree
print("Decision Tree Evaluations:")
for params in grid_search_dtree.cv_results_['params']:
    model = DecisionTreeClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    scores = evaluate_model(model, X_test, y_test)
    print(f"Parameters: {params}")
    print(f"Scores: {scores}")
    print("==========================================================")

# Evaluasi semua kombinasi parameter untuk SVM
print("SVM Evaluations:")
for params in grid_search_svm.cv_results_['params']:
    model = SVC(**params, random_state=42)
    model.fit(X_train, y_train)
    scores = evaluate_model(model, X_test, y_test)
    print(f"Parameters: {params}")
    print(f"Scores: {scores}")
    print("==========================================================")

# Print best parameters
print(f"Best parameters for Decision Tree: {grid_search_dtree.best_params_}")
print(f"Best parameters for SVM: {grid_search_svm.best_params_}")

# Cetak hasil Klasifikasi untuk kedua model
best_dtree = grid_search_dtree.best_estimator_
best_svm = grid_search_svm.best_estimator_

y_pred_dtree = best_dtree.predict(X_test)
y_pred_svm = best_svm.predict(X_test)

print("Laporan Klasifikasi - Pohon Keputusan")
print(classification_report(y_test, y_pred_dtree))
print("==========================================================")

print("Laporan Klasifikasi - Mesin Vektor Pendukung")
print(classification_report(y_test, y_pred_svm))
print("==========================================================")

# Matriks Confusion untuk Pohon Keputusan
cm_dtree = confusion_matrix(y_test, y_pred_dtree)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_dtree, annot=True, fmt='d', cmap='Blues')
plt.title('Matriks Confusion - Pohon Keputusan')
plt.xlabel('Diprediksi')
plt.ylabel('Aktual')
plt.show()

# Matriks Confusion untuk SVM
cm_svm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriks Confusion - Mesin Vektor Pendukung')
plt.xlabel('Diprediksi')
plt.ylabel('Aktual')
plt.show()

# Cross-Validation untuk evaluasi yang lebih andal (5 lipatan)
dtree_scores_5 = cross_val_score(best_dtree, X_train, y_train, cv=5)
svm_scores_5 = cross_val_score(best_svm, X_train, y_train, cv=5)

print(f"Cross-validated scores (Decision Tree) - 5 folds: {dtree_scores_5}")
print(f"Cross-validated scores (SVM) - 5 folds: {svm_scores_5}")
print("==========================================================")

# Cross-Validation untuk evaluasi yang lebih andal (10 lipatan)
dtree_scores_10 = cross_val_score(best_dtree, X_train, y_train, cv=10)
svm_scores_10 = cross_val_score(best_svm, X_train, y_train, cv=10)

print(f"Cross-validated scores (Decision Tree) - 10 folds: {dtree_scores_10}")
print(f"Cross-validated scores (SVM) - 10 folds: {svm_scores_10}")
print("==========================================================")

# Rata-rata dan standar deviasi dari skor cross-validation
print(f"Decision Tree - Mean accuracy (5 folds): {np.mean(dtree_scores_5):.2f} ± {np.std(dtree_scores_5):.2f}")
print(f"Decision Tree - Mean accuracy (10 folds): {np.mean(dtree_scores_10):.2f} ± {np.std(dtree_scores_10):.2f}")
print("==========================================================")

print(f"SVM - Mean accuracy (5 folds): {np.mean(svm_scores_5):.2f} ± {np.std(svm_scores_5):.2f}")
print(f"SVM - Mean accuracy (10 folds): {np.mean(svm_scores_10):.2f} ± {np.std(svm_scores_10):.2f}")
print("==========================================================")
