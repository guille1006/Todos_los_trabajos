
import os # Proporciona funciones para interactuar con el sistema operativo.
import pandas as pd # Manipulación y análisis de datos tabulares (filas y columnas).
import numpy as np # Operaciones numéricas y matriciales.
import seaborn as sns # Visualización estadística de datos.
import matplotlib.pyplot as plt # Creación de gráficos y visualizaciones.

  
# Matplotlib es una herramienta versátil para crear gráficos desde cero,
# mientras que Seaborn simplifica la creación de gráficos estadísticos.

from sklearn.decomposition import PCA # Implementación del Análisis de Componentes Principales (PCA).
from sklearn.preprocessing import StandardScaler # Estandarización de datos para análisis estadísticos.

#Definimos nuestro entorno de trabajo.
os.chdir('G:/Mi unidad/Apuntes titulo propio1/MasterDani/Modulo Portela/Dani') 

# Cargo las funciones que voy a utilizar
from FuncionesMineria2 import (plot_varianza_explicada, plot_cos2_heatmap, plot_corr_cos, plot_cos2_bars,
                               plot_contribuciones_proporcionales, plot_pca_scatter, plot_pca_scatter_with_vectors,
                               plot_pca_scatter_with_categories)

# Cargar un archivo Excel llamado 'iris.xlsx' en un DataFrame llamado notas.
datos = pd.read_excel('iris.xlsx') 
# No todas las categoricas estan como queremos
datos.dtypes

# Genera una lista con los nombres de las variables.
variables = list(datos.columns)   

# Seleccionar las columnas numéricas del DataFrame
numericas = datos.select_dtypes(include=['int', 'int32', 'int64','float', 'float32', 'float64']).columns

# Seleccionar las columnas categóricas del DataFrame
categoricas = [variable for variable in variables if variable not in numericas]

# Guarda la variable el indice y 'is_genuine' en un dataframe
Var_categ = datos.iloc[:,4]

#Elimina la variable 'is_genuine' del DataFrame 'datos'.

eliminar = ['class']
datos = datos.drop(eliminar, axis=1)

# Genera una lista con los nombres de las variables.
variables = list(datos.columns)  
# Cálculo de los estadísticos descriptivos.

# Calcula las estadísticas descriptivas para cada variable y crea un DataFrame con los resultados.
estadisticos = pd.DataFrame({
    'Mínimo': datos[variables].min(),
    'Percentil 25': datos[variables].quantile(0.25),
    'Mediana': datos[variables].median(),
    'Percentil 75': datos[variables].quantile(0.75),
    'Media': datos[variables].mean(),
    'Máximo': datos[variables].max(),
    'Desviación Estándar': datos[variables].std(),
    'Varianza': datos[variables].var(),
    'Coeficiente de Variación': (datos[variables].std() / datos[variables].mean()),
    'Datos Perdidos': datos[variables].isna().sum()  # Cuenta los valores NaN por variable.
})

# Calcula y representación de la matriz de correlación entre las 
# variables del DataFrame 'datos'.
R = datos.corr()

# Crea una nueva figura de tamaño 10x8 pulgadas para el gráfico.
plt.figure(figsize=(10, 8))

# Genera un mapa de calor (heatmap) de la matriz de correlación 'R' utilizando Seaborn.
# 'annot=True' agrega los valores de correlación en las celdas.
# 'cmap' establece el esquema de colores (en este caso, 'coolwarm' para colores fríos y cálidos).
# 'fmt' controla el formato de los números en las celdas ('.2f' para dos decimales).
# 'linewidths' establece el ancho de las líneas que separan las celdas.
sns.heatmap(R, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)



# -------------------------------- Analisis PCA:
    
# Estandarizamos los datos:
# Utilizamos StandardScaler() para estandarizar (normalizar) las variables.
# - StandardScaler calcula la media y la desviación estándar de las variables en 'datos' durante el ajuste.
# - Luego, utiliza estos valores para transformar 'datos' de manera que tengan media 0 y desviación estándar 1.
# - El método fit_transform() realiza ambas etapas de ajuste y transformación en una sola llamada.
# Finalmente, convertimos la salida en un DataFrame usando pd.DataFrame().
datos_estandarizados = pd.DataFrame(
    StandardScaler().fit_transform(datos),  # Datos estandarizados
    columns=['{}_z'.format(variable) for variable in variables],  # Nombres de columnas estandarizadas
    index=datos.index  # Índices (etiquetas de filas) del DataFrame
)

# Crea una instancia de Análisis de Componentes Principales (ACP):
# - Utilizamos PCA(n_components=7) para crear un objeto PCA que realizará un análisis de componentes principales.
# - Establecemos n_components en 7 para retener el maximo de las componentes principales (maximo= numero de variables).
pca = PCA(n_components=4)

# Aplicar el Análisis de Componentes Principales (ACP) a los datos estandarizados:
# - Usamos pca.fit(datos_estandarizados) para ajustar el modelo de ACP a los datos estandarizados.
fit = pca.fit(datos_estandarizados)

# Obtener los autovalores asociados a cada componente principal.
autovalores = fit.explained_variance_

# Obtener la varianza explicada por cada componente principal como un porcentaje de la varianza total.
var_explicada = fit.explained_variance_ratio_*100

# Calcular la varianza explicada acumulada a medida que se agregan cada componente principal.
var_acumulada = np.cumsum(var_explicada)

# Crear un DataFrame de pandas con los datos anteriores y establecer índice.
data = {'Autovalores': autovalores, 'Variabilidad Explicada': var_explicada, 'Variabilidad Acumulada': var_acumulada}
tabla = pd.DataFrame(data, index=['Componente {}'.format(i) for i in range(1, fit.n_components_+1)]) 

# Imprimir la tabla
print(tabla)

resultados_pca = pd.DataFrame(fit.transform(datos_estandarizados), 
                              columns=['Componente {}'.format(i) for i in range(1, fit.n_components_+1)],
                              index=datos_estandarizados.index)


# Representacion de la variabilidad explicada:   

plot_varianza_explicada(var_explicada, fit.n_components_)


# Crea una instancia de ACP con las dos primeras componentes que nos interesan y aplicar a los datos.
pca = PCA(n_components=2)
fit = pca.fit(datos_estandarizados)

# Obtener los autovalores asociados a cada componente principal.
autovalores = fit.explained_variance_

# Obtener los autovectores asociados a cada componente principal y transponerlos.
autovectores = pd.DataFrame(pca.components_.T, 
                            columns = ['Autovector {}'.format(i) for i in range(1, fit.n_components_+1)],
                            index = ['{}_z'.format(variable) for variable in variables])

# Calculamos las dos primeras componentes principales
resultados_pca = pd.DataFrame(fit.transform(datos_estandarizados), 
                              columns=['Componente {}'.format(i) for i in range(1, fit.n_components_+1)],
                              index=datos_estandarizados.index)

# Añadimos las componentes principales a la base de datos estandarizada.
datos_z_cp = pd.concat([datos_estandarizados, resultados_pca], axis=1)


# Cálculo de las correlaciones entre las variables originales y las componentes seleccionadas.
# Guardamos el nombre de las variables del archivo conjunto (variables y componentes).
variables_cp = datos_z_cp.columns

# Calculamos las correlaciones y seleccionamos las que nos interesan (variables contra componentes).
correlacion = pd.DataFrame(np.corrcoef(datos_estandarizados.T, resultados_pca.T), 
                           index = variables_cp, columns = variables_cp)

n_variables = fit.n_features_in_
correlaciones_datos_con_cp = correlacion.iloc[:fit.n_features_in_, fit.n_features_in_:]

#####################################################################################################

cos2 = correlaciones_datos_con_cp **2
plot_cos2_heatmap(cos2)
#######################################################################################################

plot_corr_cos(fit.n_components, correlaciones_datos_con_cp)

##################################################################################################

plot_cos2_bars(cos2)

contribuciones_proporcionales = plot_contribuciones_proporcionales(cos2,autovalores,fit.n_components)
######################################################################################################
            
plot_pca_scatter(pca, datos_estandarizados, fit.n_components)
################################################################################

plot_pca_scatter_with_vectors(pca, datos_estandarizados, fit.n_components, fit.components_)

##################################################


# Cargar un archivo Excel llamado 'datos.xlsx' en un DataFrame llamado datos.
datos_S = pd.read_excel('iris_S.xlsx') 



# Guarda la variable el indice y 'is_genuine' en un dataframe
Var_categ_S = datos_S.iloc[:,4]

#Elimina la variable 'is_genuine' del DataFrame 'datos'.

eliminar = ['class']
datos_S = datos_S.drop(eliminar, axis=1)


# Calcular la media y la desviación estándar de 'datos'
media_datos = datos.mean()
desviacion_estandar_datos = datos.std()

# Estandarizar 'datos_S' utilizando la media y la desviación estándar de 'datos'
datos_S_estandarizados = pd.DataFrame(((datos_S - media_datos) / desviacion_estandar_datos))

datos_S_estandarizados.columns = ['{}_z'.format(variable) for variable in variables]

# Agregar las observaciones estandarizadas a 'datos'
datos_sup = pd.concat([datos_estandarizados, datos_S_estandarizados])

# Calcular las componentes principales para el conjunto de datos combinado
componentes_principales_sup = pca.transform(datos_sup)

# Calcular las componentes principales para el conjunto de datos combinado
# y renombra las componentes
resultados_pca_sup = pd.DataFrame(fit.transform(datos_sup), 
                              columns=['Componente {}'.format(i) for i in range(1, fit.n_components_+1)],
                              index=datos_sup.index)

# Representacion observaciones + suplementarios
plot_pca_scatter(pca, datos_sup, fit.n_components)

######################################################
# Añadimos la variable categórica "Luxury" en los datos
datos_componentes_sup= pd.concat([datos_sup, resultados_pca_sup], axis=1)  

extra = Var_categ

extra_S = Var_categ_S

extra_sup = pd.concat([extra, extra_S], axis=0)
datos_componentes_sup_extra= pd.concat([datos_componentes_sup,
                                               extra_sup], axis=1)  

#################################################################################################


Luxury = 'class'      
plot_pca_scatter_with_categories(datos_componentes_sup_extra, componentes_principales_sup, fit.n_components, Luxury)
