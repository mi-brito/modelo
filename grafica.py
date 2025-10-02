# Script: deciles_presupuesto_grafica.py
# Requisitos: pip install pandas matplotlib seaborn numpy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# 1) Cargar datos
# ---------------------------
pres_wide = pd.read_csv("matriz_presupuesto.csv", dtype={"CLAVE_PROGRAMA": str})
def_wide = pd.read_csv("matriz_deforestacion.csv", dtype={"clave_municipio": str})

# ---------------------------
# 2) Transformar a formato largo
# ---------------------------
years_pres = [c for c in pres_wide.columns if c not in ("CLAVE_PROGRAMA", "CVE_MUNICIPIO")]
pres_long = pres_wide.melt(id_vars=["CLAVE_PROGRAMA", "CVE_MUNICIPIO"],
                           value_vars=years_pres,
                           var_name="ANIO", value_name="PRESUPUESTO")
pres_long["ANIO"] = pres_long["ANIO"].astype(int)
pres_long["CVE_MUNICIPIO"] = pres_long["CVE_MUNICIPIO"].astype(str)

years_def = [c for c in def_wide.columns if c != "clave_municipio"]
def_long = def_wide.melt(id_vars=["clave_municipio"],
                         value_vars=years_def,
                         var_name="ANIO", value_name="DEFORESTACION_HA")
def_long.rename(columns={"clave_municipio": "CVE_MUNICIPIO"}, inplace=True)
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

# Sumar presupuesto por municipio y año antes de decilar
pres_sum = pres_long.groupby(["CVE_MUNICIPIO","ANIO"], as_index=False)["PRESUPUESTO"].sum()
pres_sum = compute_deciles(pres_sum)

# ---------------------------
# 4) Merge con deforestación
# ---------------------------
df = pres_sum.merge(def_long, on=["CVE_MUNICIPIO","ANIO"], how="left")
df["DEFORESTACION_HA"] = df["DEFORESTACION_HA"].fillna(0)

# ---------------------------
# 5) Gráfica deciles promedio por año
# ---------------------------
decil_avg = df.groupby("ANIO")["DECIL_ANUAL_PRESUP"].mean().reset_index()

plt.figure(figsize=(10,6))
sns.lineplot(data=decil_avg, x="ANIO", y="DECIL_ANUAL_PRESUP", marker="o")
plt.title("Decil promedio de presupuesto por año")
plt.ylabel("Decil promedio")
plt.xlabel("Año")
plt.xticks(decil_avg["ANIO"])
plt.grid(True)
plt.show()
