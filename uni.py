# Script simplificado: Predicción deforestación 2024
# Requisitos: pip install pandas numpy scikit-learn matplotlib seaborn joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, r2_score, mean_absolute_error

# ---------------------------
# 1) Cargar datos
# ---------------------------
pres_wide = pd.read_csv("matriz_presupuesto.csv", dtype={"CLAVE_PROGRAMA": str})
def_wide = pd.read_csv("matriz_deforestacion.csv", dtype={"clave_municipio": str})

pres_wide.rename(columns={"CVE_MUNICIPIO": "CVE_MUNICIPIO"}, inplace=True)
def_wide.rename(columns={"clave_municipio": "CVE_MUNICIPIO"}, inplace=True)

# ---------------------------
# 2) Convertir a formato largo
# ---------------------------
years_pres = [c for c in pres_wide.columns if c not in ("CLAVE_PROGRAMA", "CVE_MUNICIPIO")]
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
# 3) Sumar presupuesto por municipio
# ---------------------------
pres_sum = pres_long.groupby(["CVE_MUNICIPIO","ANIO"], as_index=False)["PRESUPUESTO"].sum()

# ---------------------------
# 4) Merge con deforestación
# ---------------------------
df = pres_sum.merge(def_long, on=["CVE_MUNICIPIO","ANIO"], how="left")
df["DEFORESTACION_HA"] = df["DEFORESTACION_HA"].fillna(0)

# ---------------------------
# 5) Cambio simple: hay cambio o no hay cambio
# ---------------------------
df = df.sort_values(["CVE_MUNICIPIO","ANIO"])
df["PRESUPUESTO_ANT"] = df.groupby("CVE_MUNICIPIO")["PRESUPUESTO"].shift(1)
df["CAMBIO_PRESUPUESTO"] = np.where(df["PRESUPUESTO"] != df["PRESUPUESTO_ANT"], "cambio", "sin_cambio")

# ---------------------------
# 6) Features rezagadas
# ---------------------------
df["DEFORESTACION_L1"] = df.groupby("CVE_MUNICIPIO")["DEFORESTACION_HA"].shift(1)

# ---------------------------
# 7) Separar entrenamiento (2019-2023) y predicción (2024)
# ---------------------------
train = df[df["ANIO"] <= 2023].dropna(subset=["CAMBIO_PRESUPUESTO","DEFORESTACION_L1"])
predict = df[df["ANIO"] == 2024].copy()

features_pres = ["PRESUPUESTO_ANT"]
X_train = train[features_pres]
y_train = train["CAMBIO_PRESUPUESTO"]
X_pred = predict[features_pres]

# ---------------------------
# 8) Modelo presupuesto (cambio / sin cambio)
# ---------------------------
clf_pres = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
clf_pres.fit(X_train, y_train)
predict["CAMBIO_PRESUPUESTO_PRED"] = clf_pres.predict(X_pred)

print("=== Predicción cambio de presupuesto 2024 ===")
print(predict[["CVE_MUNICIPIO","CAMBIO_PRESUPUESTO_PRED"]].head())

# ---------------------------
# 9) Modelo deforestación
# ---------------------------
features_def = ["PRESUPUESTO_ANT","DEFORESTACION_L1"]
X_train_def = train[features_def]
y_train_def = train["DEFORESTACION_HA"]
X_pred_def = predict[features_def]

reg_def = RandomForestRegressor(n_estimators=200, random_state=42)
reg_def.fit(X_train_def, y_train_def)
predict["DEFORESTACION_PRED"] = reg_def.predict(X_pred_def)

print("=== Predicción deforestación 2024 ===")
print(predict[["CVE_MUNICIPIO","DEFORESTACION_PRED"]].head())

# ---------------------------
# 10) Gráfica escenario presupuesto vs deforestación
# ---------------------------
presupuestos = np.linspace(0, 200000, 20)
def_pred_list = []

DEFORESTACION_L1_prom = train["DEFORESTACION_L1"].mean()

for p in presupuestos:
    nuevo_def = pd.DataFrame({
        "PRESUPUESTO_ANT":[p],
        "DEFORESTACION_L1":[DEFORESTACION_L1_prom]
    })
    def_pred_list.append(reg_def.predict(nuevo_def)[0])

plt.figure(figsize=(8,5))
sns.lineplot(x=presupuestos, y=def_pred_list)
plt.xlabel("Presupuesto año anterior")
plt.ylabel("Deforestación predicha (ha)")
plt.title("Escenario: efecto del presupuesto sobre deforestación")
plt.show()
