import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

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
df_def_long["DEFORESTACION_ANT"] = df_def_long["DEFORESTACION_ANT"].fillna(0)

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
# 6️⃣ Calcular score estandarizado (z-score) dentro de cada decil
# ==========================
df["score"] = df.groupby(["ANIO","DECIL_PRES"])["Xn"].transform(lambda x: (x - x.mean()) / x.std())

# ==========================
# 7️⃣ Guardar CSV general
# ==========================
df = df[df["DECIL_PRES"].notna()]
df.to_csv("resultados_deforestacion_general.csv", index=False)
print("CSV general generado: resultados_deforestacion_general.csv")
# ==========================
# 8️⃣ Gráfica general
# ==========================
plt.figure(figsize=(12,6))
sns.scatterplot(data=df, x="ANIO", y="score", hue="DECIL_PRES", palette="tab10", alpha=0.5)

df_mean = df.groupby(["ANIO","DECIL_PRES"])["score"].mean().reset_index()
sns.lineplot(data=df_mean, x="ANIO", y="score", hue="DECIL_PRES", palette="tab10", linewidth=2, legend=False)

plt.title("Score de deforestación por decil de presupuesto")
plt.ylabel("Score (Xn estandarizado)")
plt.xlabel("Año")
plt.legend(title="Decil Presupuesto", bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()
plt.savefig("grafica_general.png")
plt.show()

# ==========================
# 9️⃣ CSV y gráfica por programa
# ==========================
programas = df["CLAVE_PROGRAMA"].unique()
os.makedirs("programas", exist_ok=True)

for prog in programas:
    df_prog = df[df["CLAVE_PROGRAMA"]==prog].copy()
    
    # Filtrar filas con DECIL_PRES válido
    df_prog = df_prog[df_prog["DECIL_PRES"].notna()]
    
    df_prog.to_csv(f"programas/resultados_{prog}.csv", index=False)
    
    plt.figure(figsize=(12,6))
    sns.scatterplot(data=df_prog, x="ANIO", y="score", hue="DECIL_PRES", palette="tab10", alpha=0.5)
    df_prog_mean = df_prog.groupby(["ANIO","DECIL_PRES"])["score"].mean().reset_index()
    sns.lineplot(data=df_prog_mean, x="ANIO", y="score", hue="DECIL_PRES", palette="tab10", linewidth=2, legend=False)
    
    plt.title(f"Score de deforestación - Programa {prog}")
    plt.ylabel("Score (Xn estandarizado)")
    plt.xlabel("Año")
    plt.legend(title="Decil Presupuesto", bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"programas/grafica_{prog}.png")
    plt.close()

    
print("CSV y gráficas por programa generadas en carpeta 'programas'")
