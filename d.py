# Script: analisis_presupuesto_deforestacion_mejorado.py
# Requisitos: pip install pandas numpy scikit-learn matplotlib seaborn joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, r2_score, mean_absolute_error

# ---------------------------
# 1) Cargar datos históricos
# ---------------------------
pres_wide = pd.read_csv("historico_presupuesto.csv", dtype={"CLAVE_PROGRAMA": str})
def_wide = pd.read_csv("matriz_deforestacion.csv", dtype={"clave_municipio": str})

pres_wide.rename(columns={"CVE_MUNICIPIO": "CVE_MUNICIPIO"}, inplace=True)
def_wide.rename(columns={"clave_municipio": "CVE_MUNICIPIO"}, inplace=True)

# ---------------------------
# 2) Convertir a formato largo
# ---------------------------
years_pres = [c for c in pres_wide.columns if c not in ("CLAVE_PROGRAMA","CVE_MUNICIPIO")]
pres_long = pres_wide.melt(id_vars=["CLAVE_PROGRAMA","CVE_MUNICIPIO"], value_vars=years_pres,
                           var_name="ANIO", value_name="PRESUPUESTO")
pres_long["ANIO"] = pres_long["ANIO"].astype(int)
pres_long["CVE_MUNICIPIO"] = pres_long["CVE_MUNICIPIO"].astype(str)

years_def = [c for c in def_wide.columns if c != "CVE_MUNICIPIO"]
def_long = def_wide.melt(id_vars=["CVE_MUNICIPIO"], value_vars=years_def,
                         var_name="ANIO", value_name="DEFORESTACION_HA")
def_long["ANIO"] = def_long["ANIO"].astype(int)
def_long["CVE_MUNICIPIO"] = def_long["CVE_MUNICIPIO"].astype(str)

# ---------------------------
# 3) Calcular deciles de presupuesto por municipio y año
# ---------------------------
def compute_deciles(df):
    df = df.copy()
    df["DECIL_ANUAL_PRESUP"] = np.nan
    for year, g in df.groupby("ANIO"):
        mask_pos = g["PRESUPUESTO"] > 0
        idx_pos = g.loc[mask_pos].index
        if len(idx_pos) == 0:
            df.loc[g.index, "DECIL_ANUAL_PRESUP"] = 0
            continue
        ranks = g.loc[idx_pos,"PRESUPUESTO"].rank(pct=True, method="first")
        deciles_pos = np.ceil(ranks*10).astype(int).clip(1,10)
        df.loc[idx_pos,"DECIL_ANUAL_PRESUP"] = deciles_pos
        df.loc[g.index.difference(idx_pos),"DECIL_ANUAL_PRESUP"] = 0
    df["DECIL_ANUAL_PRESUP"] = df["DECIL_ANUAL_PRESUP"].astype(int)
    return df

# Sumar presupuesto por municipio antes de decilar
pres_sum = pres_long.groupby(["CVE_MUNICIPIO","ANIO"], as_index=False)["PRESUPUESTO"].sum()
pres_sum = compute_deciles(pres_sum)

# ---------------------------
# 4) Merge con deforestación
# ---------------------------
df = pres_sum.merge(def_long, on=["CVE_MUNICIPIO","ANIO"], how="left")
df["DEFORESTACION_HA"] = df["DEFORESTACION_HA"].fillna(0)

# ---------------------------
# 5) Cambio de decil y clases
# ---------------------------
df = df.sort_values(["CVE_MUNICIPIO","ANIO"])
df["DECIL_ANT"] = df.groupby("CVE_MUNICIPIO")["DECIL_ANUAL_PRESUP"].shift(1)
df["DELTA_DECIL"] = df["DECIL_ANUAL_PRESUP"] - df["DECIL_ANT"]

def label_5clas(delta):
    if pd.isna(delta): return np.nan
    if delta <= -2: return "bajada_fuerte"
    if delta == -1: return "bajada_leve"
    if delta == 0: return "sin_cambio"
    if delta == 1: return "subida_leve"
    return "subida_fuerte"

df["CLASE_5"] = df["DELTA_DECIL"].apply(label_5clas)

# ---------------------------
# 6) Features rezagadas
# ---------------------------
for lag in [1,2]:
    df[f"PRESUPUESTO_L{lag}"] = df.groupby("CVE_MUNICIPIO")["PRESUPUESTO"].shift(lag)
    df[f"DEFORESTACION_L{lag}"] = df.groupby("CVE_MUNICIPIO")["DEFORESTACION_HA"].shift(lag)
    df[f"DECIL_L{lag}"] = df.groupby("CVE_MUNICIPIO")["DECIL_ANUAL_PRESUP"].shift(lag)

# ---------------------------
# 7) Modelo presupuesto (cambio de decil)
# ---------------------------
df_train = df[df["ANIO"]<2024].dropna(subset=["CLASE_5","PRESUPUESTO_L1"])
features_pres = ["PRESUPUESTO_L1","PRESUPUESTO_L2","DECIL_L1","DECIL_L2"]
X_pres = df_train[features_pres]
y_pres = df_train["CLASE_5"]

# Random Forest limitado para no sobreajustar
clf_pres = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=10,
                                  random_state=42, class_weight="balanced")

# Validación cruzada
cv_scores = cross_val_score(clf_pres, X_pres, y_pres, cv=5, scoring="f1_macro")
print("F1 macro CV promedio (presupuesto):", np.mean(cv_scores))

clf_pres.fit(X_pres, y_pres)

# Predicción 2024
df_pred = df[df["ANIO"]==2024].copy()
X_pred_pres = df_pred[features_pres]
df_pred["CLASE_5_pred"] = clf_pres.predict(X_pred_pres)

print("=== Predicción cambio decil 2024 ===")
print(df_pred[["CVE_MUNICIPIO","CLASE_5_pred"]])

# ---------------------------
# 8) Modelo deforestación
# ---------------------------
df_train_def = df_train.dropna(subset=["DEFORESTACION_HA"] + features_pres)
df_train_def["CLASE_5_pred_num"] = clf_pres.predict(df_train_def[features_pres])
df_train_def["CLASE_5_pred_num"] = df_train_def["CLASE_5_pred_num"].map({
    "bajada_fuerte": -2, "bajada_leve": -1, "sin_cambio":0,
    "subida_leve":1, "subida_fuerte":2
})

# Log-transform para no predecir ceros absolutos
df_train_def["DEFORESTACION_LOG"] = np.log1p(df_train_def["DEFORESTACION_HA"])

features_def = ["PRESUPUESTO_L1","PRESUPUESTO_L2","CLASE_5_pred_num","DEFORESTACION_L1","DEFORESTACION_L2"]
X_def = df_train_def[features_def]
y_def = df_train_def["DEFORESTACION_LOG"]

reg_def = RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_leaf=5, random_state=42)

cv_def = cross_val_score(reg_def, X_def, y_def, cv=5, scoring="r2")
print("R2 CV promedio (deforestación):", np.mean(cv_def))

reg_def.fit(X_def, y_def)

# Predicción deforestación 2024
df_pred_def = df_pred.copy()
df_pred_def["CLASE_5_pred_num"] = df_pred_def["CLASE_5_pred"].map({
    "bajada_fuerte": -2, "bajada_leve": -1, "sin_cambio":0,
    "subida_leve":1, "subida_fuerte":2
})
X_pred_def = df_pred_def[["PRESUPUESTO_L1","PRESUPUESTO_L2","CLASE_5_pred_num","DEFORESTACION_L1","DEFORESTACION_L2"]]
df_pred_def["DEFORESTACION_PRED"] = np.expm1(reg_def.predict(X_pred_def))

print("=== Predicción deforestación 2024 ===")
print(df_pred_def[["CVE_MUNICIPIO","DEFORESTACION_PRED"]])

# ---------------------------
# 9) Guardar resultados
# ---------------------------
df_pred_def.to_csv("predicciones_presupuesto_deforestacion_2024.csv", index=False)
