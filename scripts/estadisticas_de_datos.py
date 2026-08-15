import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

datadir = "../data/"
datafilename = "data.csv"
targetfilename = "target.csv"
statsdir = "../estadisticas/"

# Importamos los datos
data = pd.read_csv(datadir + datafilename, index_col=0)
target = pd.read_csv(datadir + targetfilename, index_col=0)

# Resumen de los datos
data_summary = data.describe()
target_summary = target.describe()

# Exportamos los resultados
data_summary.to_csv(statsdir + "data_summary.csv")
target_summary.to_csv(statsdir + "target_summary.csv")

# Distribución de target
plt.figure(figsize=(7, 4))
plt.hist(target, bins=50, color='steelblue', edgecolor='white')
plt.axvline(target.max()["MedHouseVal"], color='red', linestyle='--', \
        label=f"Techo censurado ({target.max()["MedHouseVal"]:.2f})"
        # label="Techo censurado"
            )
plt.xlabel('Valor medio de la vivienda (cientos de miles USD)')
plt.ylabel('Frecuencia')
plt.title('Distribución del target')
plt.legend()
plt.tight_layout()
plt.savefig(statsdir + "target_dist.png")

# Coeficientes de correlación con target
df_corr = data.copy()
df_corr['MedHouseVal'] = target

correlaciones = df_corr.corr()['MedHouseVal'].sort_values(ascending=False)
correlaciones.to_csv(statsdir + "correlaciones.csv")

plt.figure(figsize=(8, 5))
correlaciones.drop('MedHouseVal').sort_values().plot(kind='barh', color='steelblue')
plt.title('Correlación de cada variable con el valor medio de la vivienda')
plt.xlabel('Coeficiente de correlación de Pearson')
plt.axvline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig(statsdir + "correlaciones_graf.png")

