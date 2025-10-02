import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================
# 1️⃣ Leer datos
# ==========================
df_def = pd.read_csv("matriz_deforestacion.csv")  # columnas: clave_municipio,2001,2002,...
df_def = df_def.rename(columns={"clave_municipio": "CVE_MUNICIPIO"})

df_pres = pd.read_csv("matriz_presupuesto.csv")  # columnas: CLAVE_PROGRAMA,CVE_MUNICIPIO,2019,2020,...
df_pres_long = df_pres.melt(id_vars=["CLAVE_PROGRAMA","CVE_MUNICIPIO"],
                            var_name="ANIO", value_name="PRESUPUESTO")
df_pres_long["ANIO"] = df_pres_long["ANIO"].astype(int)

# ==========================
# 2️⃣ Transformar deforestación a formato largo
# ==========================
df_def_long = df_def.melt(id_vars=["CVE_MUNICIPIO"], var_name="ANIO", value_name="DEFORESTACION_HA")
df_def_long["ANIO"] = df_def_long["ANIO"].astype(int)
df_def_long = df_def_long.sort_values(["CVE_MUNICIPIO","ANIO"])

# ==========================
# 3️⃣ Calcular deforestación anterior y Xn
# ==========================
df_def_long["DEFORESTACION_ANT"] = df_def_long.groupby("CVE_MUNICIPIO")["DEFORESTACION_HA"].shift(1)
df_def_long["DEFORESTACION_ANT"] = df_def_long["DEFORESTACION_ANT"].fillna(0)  # opcional para primer año

df_def_long["Xn"] = df_def_long["DEFORESTACION_HA"] - df_def_long["DEFORESTACION_ANT"]

# ==========================
# 4️⃣ Unir con presupuesto
# ==========================
df = df_def_long.merge(df_pres_long, on=["CVE_MUNICIPIO","ANIO"], how="left")

# ==========================
# 5️⃣ Calcular deciles de presupuesto por año
# ==========================
df["DECIL_PRES"] = df.groupby("ANIO")["PRESUPUESTO"].transform(
    lambda x: pd.qcut(x, 10, labels=False, duplicates="drop")+1
)

# ==========================
# 6️⃣ Calcular score dentro de cada decil y año
# ==========================
df["score"] = df.groupby(["ANIO","DECIL_PRES"])["Xn"].transform(lambda x: (x - x.mean()) / x.std())

# ==========================
# 7️⃣ Verificación
# ==========================
print("Columnas disponibles:", df.columns)
print("Valores únicos DECIL_PRES:", df["DECIL_PRES"].unique())
print("Número de filas:", len(df))
df_valid = df[df["DECIL_PRES"].notna()]
print(df_valid.head())

# ==========================
# 8️⃣ Graficar
# ==========================
plt.figure(figsize=(12,6))
sns.scatterplot(data=df, x="ANIO", y="score", hue="DECIL_PRES", palette="tab10", alpha=0.5)

# Línea de tendencia promedio por decil
df_mean = df.groupby(["ANIO","DECIL_PRES"])["score"].mean().reset_index()
sns.lineplot(data=df_mean, x="ANIO", y="score", hue="DECIL_PRES", palette="tab10", linewidth=2, legend=False)

plt.title("Score de deforestación por decil de presupuesto")
plt.ylabel("Score (Xn estandarizado)")
plt.xlabel("Año")
plt.legend(title="Decil Presupuesto", bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()
plt.show()
