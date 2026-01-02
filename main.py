# ============================================================
# PROJET IA – MAINTENANCE PRÉDICTIVE
# PHASE 3 : MODÉLISATION
# PHASE 4 : ÉVALUATION (RMSE)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. NOMS DES COLONNES
# ============================================================
colonnes = [
    'unit_nr', 'time_cycles',
    'op_setting_1', 'op_setting_2', 'op_setting_3'
] + [f'sensor_{i}' for i in range(1, 22)]

# ============================================================
# ===================== PHASE 3 ===============================
# ============================================================

# ============================================================
# 2. CHARGEMENT DES DONNÉES D’ENTRAÎNEMENT
# ============================================================
df = pd.read_csv('train_FD001.txt', sep=r'\s+', header=None, names=colonnes)
print("Train chargé :", df.shape)

# ============================================================
# 3. SUPPRESSION DES CAPTEURS CONSTANTS
# ============================================================
capteurs = [c for c in df.columns if c.startswith('sensor_')]
capteurs_constants = [c for c in capteurs if df[c].nunique() == 1]

print("Capteurs constants supprimés :", capteurs_constants)

df_clean = df.drop(columns=capteurs_constants)

# ============================================================
# 4. CALCUL DU RUL (CIBLE À PRÉDIRE)
# ============================================================
df_clean['RUL'] = (
    df_clean.groupby('unit_nr')['time_cycles'].transform('max')
    - df_clean['time_cycles']
)

# ============================================================
# 5. NORMALISATION MIN-MAX DES CAPTEURS
# ============================================================
from sklearn.preprocessing import MinMaxScaler

capteurs_utiles = [c for c in df_clean.columns if c.startswith('sensor_')]
scaler = MinMaxScaler()

df_clean[capteurs_utiles] = scaler.fit_transform(df_clean[capteurs_utiles])

# ============================================================
# 6. MATRICE DE CORRÉLATION (ANALYSE)
# ============================================================
matrice_corr = df_clean[capteurs_utiles].corr()

with pd.ExcelWriter('train_FD001_clean.xlsx', engine='openpyxl') as writer:
    df_clean.to_excel(writer, sheet_name='Donnees_Nettoyees_RUL', index=False)
    matrice_corr.to_excel(writer, sheet_name='Correlation_Matrix')

print("Fichier Excel train_FD001_clean.xlsx généré")

# ============================================================
# 7. DÉFINITION DES ENTRÉES (X) ET DE LA SORTIE (y)
# ============================================================
# On enlève unit_nr et time_cycles → l’IA apprend via les capteurs
X = df_clean.drop(columns=['unit_nr', 'time_cycles', 'RUL'])
y = df_clean['RUL']

# ============================================================
# 8. SPLIT TRAIN / VALIDATION
# ============================================================
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================
# 9. MODÈLE BASELINE – RÉGRESSION LINÉAIRE
# ============================================================
from sklearn.linear_model import LinearRegression

model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_val)

# ============================================================
# 10. MODÈLE AVANCÉ – RANDOM FOREST
# ============================================================
from sklearn.ensemble import RandomForestRegressor

model_rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_val)

# ============================================================
# 11. ÉVALUATION PHASE 3 (RMSE INTERNE)
# ============================================================
from sklearn.metrics import mean_squared_error

rmse_lr = np.sqrt(mean_squared_error(y_val, y_pred_lr))
rmse_rf = np.sqrt(mean_squared_error(y_val, y_pred_rf))

print("\nPHASE 3 – RÉSULTATS")
print(f"RMSE Régression Linéaire : {rmse_lr:.2f}")
print(f"RMSE Random Forest      : {rmse_rf:.2f}")

# ============================================================
# ===================== PHASE 4 ===============================
# ============================================================

# ============================================================
# 12. CHARGEMENT DU FICHIER TEST
# ============================================================
df_test = pd.read_csv('test_FD001.txt', sep=r'\s+', header=None, names=colonnes)
print("\nTest chargé :", df_test.shape)

# ============================================================
# 13. MÊME NETTOYAGE QUE LE TRAIN
# ============================================================
df_test = df_test.drop(columns=capteurs_constants)
df_test[capteurs_utiles] = scaler.transform(df_test[capteurs_utiles])

# ============================================================
# 14. GARDER LE DERNIER CYCLE DE CHAQUE MOTEUR
# ============================================================
df_last_cycle = df_test.groupby('unit_nr').last()
print("Nombre de moteurs test :", df_last_cycle.shape[0])

# ============================================================
# 15. PRÉDICTION DU RUL AU DERNIER CYCLE
# ============================================================
X_test_final = df_last_cycle.drop(columns=['time_cycles'])
y_pred_test = model_rf.predict(X_test_final)

# ============================================================
# 16. CHARGER LA VÉRITÉ TERRAIN
# ============================================================
y_true = pd.read_csv('RUL_FD001.txt', header=None, names=['RUL'])['RUL'].values

# ============================================================
# 17. CALCUL DU RMSE FINAL
# ============================================================
rmse_test = np.sqrt(mean_squared_error(y_true, y_pred_test))

print("\nPHASE 4 – RÉSULTAT FINAL")
print("RMSE FINAL sur test_FD001 :", rmse_test)
