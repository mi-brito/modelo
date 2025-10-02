# Script: analisis_presupuesto_deforestacion_historico.py
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
# 1) Cargar datos y filtrar histórico
# ---------------------------
pres_wide = pd.read_csv("matriz_presupuesto.csv", dtype={"CLAVE_PROGRAMA": str})
def_wide = pd.read_csv("matriz_deforestacion.csv", dtype={"clave_municipio": str})

pres_wide.rename(columns={"CVE_MUNICIPIO":"CVE_MUNICIPIO"}, inplace=True)
def_wide.rename(columns={"clave_municipio":"CVE_MUNICIPIO"}, inplace=True)

# Filtrar solo años 2019-2024
years_pres = [c for c in pres_wide.columns if c not in ("CLAVE_PROGRAMA","CVE_MUNICIPIO")]
years_pres = [y for y in years_pres if int(y) <= 2024]

years_def = [c for c in def_wide.columns if c != "CVE_MUNICIPIO"]
years_def = [y for y in years_def if int(y) <= 2024]

# ---------------------------
# 2) Convertir a formato largo
# ---------------------------
pres_long = pres_wide.melt(id_vars=["CLAVE_PROGRAMA","CVE_MUNICIPIO"],
                           value_vars=years_pres,
                           var_name="ANIO", value_name="PRESUPUESTO")
pres_long["ANIO"] = pres_long["ANIO"].astype(int)
pres_long["CVE_MUNICIPIO"] = pres_long["CVE_MUNICIPIO"].astype(str)

def_long = def_wide.melt(id_vars=["CVE_MUNICIPIO"],
                         value_vars=years_def,
                         var_name="ANIO", value_name="DEFORESTACION_HA")
def_long["ANIO"] = def_long["ANIO"].astype(int)
def_long["CVE_MUNICIPIO"] = def_long["CVE_MUNICIPIO"].astype(str)

# ---------------------------
# 3) Calcular deciles de presupuesto
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
# 7) Separar datos entrenamiento y predicción
# ---------------------------
# Entrenamiento: 2019-2023
df_train = df[df["ANIO"] <= 2023].dropna(subset=["CLASE_5","PRESUPUESTO_L1"])
# Predicción: 2024
df_pred = df[df["ANIO"] == 2024].dropna(subset=["PRESUPUESTO_L1"])

# ---------------------------
# 8) Modelo presupuesto (cambio de decil)
# ---------------------------
features_pres = ["PRESUPUESTO_L1","PRESUPUESTO_L2","DECIL_L1","DECIL_L2"]
X_train_pres = df_train[features_pres]
y_train_pres = df_train["CLASE_5"]
X_pred_pres = df_pred[features_pres]

clf_pres = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
clf_pres.fit(X_train_pres, y_train_pres)

# Predicción cambio decil 2024
df_pred["CLASE_5_pred"] = clf_pres.predict(X_pred_pres)
print("=== Predicción cambio decil 2024 ===")
print(df_pred[["CVE_MUNICIPIO","CLASE_5_pred"]])

# ---------------------------
# 9) Modelo deforestación
# ---------------------------
df_pred["CLASE_5_pred_num"] = df_pred["CLASE_5_pred"].map({
    "bajada_fuerte": -2, "bajada_leve": -1, "sin_cambio":0, "subida_leve":1, "subida_fuerte":2
})
df_train["CLASE_5_pred_num"] = clf_pres.predict(df_train[features_pres])
df_train["CLASE_5_pred_num"] = df_train["CLASE_5_pred_num"].map({
    "bajada_fuerte": -2, "bajada_leve": -1, "sin_cambio":0, "subida_leve":1, "subida_fuerte":2
})

features_def = ["PRESUPUESTO_L1","PRESUPUESTO_L2","CLASE_5_pred_num","DEFORESTACION_L1","DEFORESTACION_L2"]

X_train_def = df_train[features_def]
y_train_def = df_train["DEFORESTACION_HA"]
X_pred_def = df_pred[features_def]

reg_def = RandomForestRegressor(n_estimators=200, random_state=42)
reg_def.fit(X_train_def, y_train_def)

# Predicción deforestación 2024
df_pred["DEFORESTACION_PRED"] = reg_def.predict(X_pred_def)
print("=== Predicción deforestación 2024 ===")
print(df_pred[["CVE_MUNICIPIO","DEFORESTACION_PRED"]])

# ---------------------------
# 10) Guardar modelos y resultados
# ---------------------------
joblib.dump(clf_pres, "modelo_presupuesto_historico.pkl")
joblib.dump(reg_def, "modelo_deforestacion_historico.pkl")
df_pred.to_csv("predicciones_2024.csv", index=False)

# ---------------------------
# 11) Escenarios y gráficas
# ---------------------------
presupuestos = np.linspace(0, 200000, 20)
def_pred_list = []

DEFORESTACION_L1_prom = df_train["DEFORESTACION_L1"].mean()

for p in presupuestos:
    clase_pred = clf_pres.predict(pd.DataFrame({"PRESUPUESTO_L1":[p], "PRESUPUESTO_L2":[p],
                                                "DECIL_L1":[5], "DECIL_L2":[5]}))
    clase_pred_num = pd.Series(clase_pred).map({
        "bajada_fuerte": -2, "bajada_leve": -1, "sin_cambio":0, "subida_leve":1, "subida_fuerte":2
    })
    nuevo_def = pd.DataFrame({
        "PRESUPUESTO_L1":[p],
        "PRESUPUESTO_L2":[p],
        "CLASE_5_pred_num": clase_pred_num,
        "DEFORESTACION_L1":[DEFORESTACION_L1_prom],
        "DEFORESTACION_L2":[DEFORESTACION_L1_prom]
    })
    def_pred_list.append(reg_def.predict(nuevo_def)[0])

plt.figure(figsize=(8,5))
sns.lineplot(x=presupuestos, y=def_pred_list)
plt.xlabel("Presupuesto del año anterior")
plt.ylabel("Deforestación predicha (ha)")
plt.title("Escenario: efecto del presupuesto en deforestación 2024")
plt.show()
