# Script: analisis_presupuesto_deforestacion_final.py
# Requisitos: pip install pandas numpy scikit-learn matplotlib seaborn joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, r2_score, mean_absolute_error

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
pres_long = pres_wide.melt(id_vars=["CLAVE_PROGRAMA", "CVE_MUNICIPIO"], value_vars=years_pres,
                           var_name="ANIO", value_name="PRESUPUESTO")
pres_long["ANIO"] = pres_long["ANIO"].astype(int)
pres_long["CVE_MUNICIPIO"] = pres_long["CVE_MUNICIPIO"].astype(str)

years_def = [c for c in def_wide.columns if c != "CVE_MUNICIPIO"]
def_long = def_wide.melt(id_vars=["CVE_MUNICIPIO"], value_vars=years_def,
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
        ranks = g.loc[idx_pos, "PRESUPUESTO"].rank(pct=True, method="first")
        deciles_pos = np.ceil(ranks*10).astype(int).clip(1,10)
        df.loc[idx_pos, "DECIL_ANUAL_PRESUP"] = deciles_pos
        df.loc[g.index.difference(idx_pos), "DECIL_ANUAL_PRESUP"] = 0
    df["DECIL_ANUAL_PRESUP"] = df["DECIL_ANUAL_PRESUP"].astype(int)
    return df

# Sumar presupuesto por municipio antes de decilar si hay varios programas
pres_sum = pres_long.groupby(["CVE_MUNICIPIO","ANIO"], as_index=False)["PRESUPUESTO"].sum()
pres_sum = compute_deciles(pres_sum)

# ---------------------------
# Verificación de deciles
# ---------------------------
# Resumen de deciles por año
deciles_por_anio = pres_sum.groupby("ANIO")["DECIL_ANUAL_PRESUP"].value_counts().unstack(fill_value=0)
print("=== Resumen de deciles por año ===")
print(deciles_por_anio)


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

# Cambios de decil
df_check = df.groupby("CLASE_5").size()
print("\n=== Conteo por clase de cambio de decil ===")
print(df_check)

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
df_modelo_pres = df.dropna(subset=["CLASE_5","PRESUPUESTO_L1"]).copy()
features_pres = ["PRESUPUESTO_L1","PRESUPUESTO_L2","DECIL_L1","DECIL_L2"]
X_pres = df_modelo_pres[features_pres]
y_pres = df_modelo_pres["CLASE_5"]

X_train, X_test, y_train, y_test = train_test_split(X_pres, y_pres, test_size=0.2, random_state=42, stratify=y_pres)
clf_pres = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
clf_pres.fit(X_train, y_train)
y_pred_pres = clf_pres.predict(X_test)

print("=== Presupuesto: Clasificación cambio decil ===")
print(classification_report(y_test, y_pred_pres))

df_modelo_pres["CLASE_5_pred"] = clf_pres.predict(df_modelo_pres[features_pres])

# ---------------------------
# 8) Modelo deforestación
# ---------------------------
df_modelo_def = df_modelo_pres.copy()
df_modelo_def["CLASE_5_pred_num"] = df_modelo_def["CLASE_5_pred"].map({
    "bajada_fuerte": -2, "bajada_leve": -1, "sin_cambio":0, "subida_leve":1, "subida_fuerte":2
})

features_def = ["PRESUPUESTO_L1","PRESUPUESTO_L2","CLASE_5_pred_num","DEFORESTACION_L1","DEFORESTACION_L2"]
df_modelo_def = df_modelo_def.dropna(subset=features_def + ["DEFORESTACION_HA"])
X_def = df_modelo_def[features_def]
y_def = df_modelo_def["DEFORESTACION_HA"]

X_train_def, X_test_def, y_train_def, y_test_def = train_test_split(X_def, y_def, test_size=0.2, random_state=42)
reg_def = RandomForestRegressor(n_estimators=300, random_state=42)
reg_def.fit(X_train_def, y_train_def)
y_pred_def = reg_def.predict(X_test_def)

print("=== Deforestación: Evaluación ===")
print("R2 score:", r2_score(y_test_def, y_pred_def))
print("MAE:", mean_absolute_error(y_test_def, y_pred_def))

# ---------------------------
# 9) Guardar modelos y resultados
# ---------------------------
joblib.dump(clf_pres, "modelo_presupuesto.pkl")
joblib.dump(reg_def, "modelo_deforestacion.pkl")

df_modelo_def.loc[X_test_def.index, "DEFORESTACION_PRED"] = y_pred_def
df_modelo_def.to_csv("predicciones_presupuesto_deforestacion_final.csv", index=False)

importances_def = pd.Series(reg_def.feature_importances_, index=features_def).sort_values(ascending=False)
importances_def.to_csv("importancia_variables_deforestacion.csv")

# ---------------------------
# 10) Escenarios y gráficas realistas
# ---------------------------
presupuestos = np.linspace(0, 200000, 20)
def_pred_list = []

# Usamos DEFORESTACION_L1 como promedio histórico para simular
DEFORESTACION_L1_prom = df_modelo_def["DEFORESTACION_L1"].mean()

for p in presupuestos:
    clase_pred = clf_pres.predict(pd.DataFrame({"PRESUPUESTO_L1":[p], "PRESUPUESTO_L2":[p],
                                                "DECIL_L1":[5], "DECIL_L2":[5]}))
    clase_pred_num = pd.Series(clase_pred).map({
        "bajada_fuerte": -2, "bajada_leve": -1, "sin_cambio":0, "subida_leve":1, "subida_fuerte":2
    })
    nuevo_def = pd.DataFrame({
        "PRESUPUESTO_L1": [p],
        "PRESUPUESTO_L2": [p],
        "CLASE_5_pred_num": clase_pred_num,
        "DEFORESTACION_L1": [DEFORESTACION_L1_prom],
        "DEFORESTACION_L2": [DEFORESTACION_L1_prom]
    })
    def_pred_list.append(reg_def.predict(nuevo_def)[0])

plt.figure(figsize=(8,5))
sns.lineplot(x=presupuestos, y=def_pred_list)
plt.xlabel("Presupuesto del año anterior")
plt.ylabel("Deforestación predicha (ha)")
plt.title("Escenario: efecto del presupuesto en deforestación")
plt.show()
