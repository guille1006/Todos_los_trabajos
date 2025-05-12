import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import statsmodels.api as sm

# Lista de datos de ejemplo
data = [1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4, 4, 5]

# 1. Calcular la ECDF (empirical cumulative distribution function)
ecdf = sm.distributions.ECDF(data)

# 2. Crear la CDF teórica de una distribución normal
x = np.linspace(min(data), max(data), 1000)  # Puntos en el rango de los datos
mean, std = np.mean(data), np.std(data)  # Media y desviación estándar de los datos
cdf_normal = norm.cdf(x, loc=mean, scale=std)  # CDF normal teórica

data_unique = sorted(set(data))

# Calcular frecuencias acumuladas
ecdf_y = np.arange(1, len(data) + 1) / len(data)
print(ecdf_y)
indices_cambio = [i for i in range(1, len(data)) if data[i] != data[i-1]]
indices_cambio = np.array(indices_cambio)-1
ecdf_y = ecdf_y[ indices_cambio]
print(ecdf_y)

# Graficar ECDF (pasos)
plt.step(ecdf.x, ecdf.y, where='post', label="ECDF (Empírica)", color='blue')
# Graficar CDF normal teórica
plt.plot(x, cdf_normal, label="CDF Normal Teórica", color='red')

# Configuración de la gráfica
plt.xlabel('Valor')
plt.ylabel('Probabilidad acumulada')
plt.title('Comparación entre ECDF y CDF Normal')
plt.legend()
plt.grid()
plt.show()
