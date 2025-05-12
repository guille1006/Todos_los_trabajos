import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from itertools import product
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from itertools import chain
import scipy.stats as stats
import warnings
import math

#--------------------------------------------------------------------------------------------------------------
def analizar_variables_categoricas(datos):
    """
    Analiza variables categóricas en un DataFrame.

    Args:
        datos (DataFrame): El DataFrame que contiene los datos.

    Returns:
        dict: Un diccionario donde aparecen las diferentes categorias, sus frecuencias
        absolutas y relativas.
    """
    # Inicializar un diccionario para almacenar los resultados
    resultados = {}
    
    # Genera una lista con los nombres de las variables.
    variables = list(datos.columns) 
    
    # Seleccionar las columnas numéricas en el DataFrame
    numericas = datos.select_dtypes(include=['int', 'int32', 'int64','float', 'float32', 'float64']).columns

    # Seleccionar las columnas categóricas en el DataFrame
    categoricas = [variable for variable in variables if variable not in numericas]
    
    # Iterar a través de las variables categóricas
    for categoria in categoricas:
        # Verificar si la variable categórica existe en el DataFrame
        if categoria in datos.columns:
            # Crear un DataFrame de resumen para la variable categórica
            resumen = pd.DataFrame({
                'n': datos[categoria].value_counts(),             # Conteo de frecuencias
                '%': datos[categoria].value_counts(normalize=True)  # Porcentaje de frecuencias
            })
            resultados[categoria] = resumen  # Almacenar el resumen en el diccionario
        else:
            # Si la variable no existe en los datos, almacenar None en el diccionario
            resultados[categoria] = None
    
    return resultados

#--------------------------------------------------------------------------------------------------------------
def cuentaDistintos(datos):
    """
    Cuenta valores distintos en cada variable numerica de un DataFrame.

    Args:
        datos (DataFrame): El DataFrame que contiene los datos.

    Returns:
        Dataframe: Un DataFrame con las variables y valores distintos en cada una de ellas
    """
    # Seleccionar las columnas numéricas en el DataFrame
    numericas = datos.select_dtypes(include=['int', 'int32', 'int64','float', 'float32', 'float64'])
    
    # Calcular la cantidad de valores distintos en cada columna numérica
    resultados = numericas.apply(lambda x: len(x.unique()))
    
    # Crear un DataFrame con los resultados
    resultado = pd.DataFrame({'Columna': resultados.index, 'Distintos': resultados.values})
    
    return resultado

#--------------------------------------------------------------------------------------------------------------
def frec_variables_num(datos, NumCat):
    """
    Calcula las frecuencias de los diferentes valores de variables numericas (tratadas como categóricas).
    Args:
        datos: DataFrame de datos.
        NumCat: Lista de nombres de variables númericas a analizar.
        :return: Un diccionario donde las claves son los nombres de las variables numericas y los valores son DataFrames
             con el resumen de frecuencias y porcentajes.
    """
    resultados = {}

    for categoria in NumCat:
        # Verificar si la variable categórica existe en el DataFrame
        if categoria in datos.columns:
            # Crear un DataFrame de resumen para la variable categórica
            resumen = pd.DataFrame({
                'n': datos[categoria].value_counts(),             # Conteo de frecuencias
                '%': datos[categoria].value_counts(normalize=True)  # Porcentaje de frecuencias
            })
            resultados[categoria] = resumen  # Almacenar el resumen en el diccionario
        else:
            # Si la variable no existe en los datos, almacenar None en el diccionario
            resultados[categoria] = None
    
    return resultados

#--------------------------------------------------------------------------------------------------------------
def atipicosAmissing(varaux):
    """
    Esta función identifica valores atípicos en una serie de datos y los reemplaza por NaN.
    
    Datos de entrada:
    - varaux: Serie de datos en la que se buscarán valores atípicos.
    
    Datos de salida:
    - Una nueva serie de datos con valores atípicos reemplazados por NaN.
    - El número de valores atípicos identificados.
    """
    
    # Verifica si la distribución de los datos es simétrica o asimétrica
    if abs(varaux.skew()) < 1:
        # Si es simétrica, calcula los valores atípicos basados en la desviación estándar
        criterio1 = abs((varaux - varaux.mean()) / varaux.std()) > 3
    else:
        # Si es asimétrica, calcula la Desviación Absoluta de la Mediana (MAD) y los valores atípicos
        mad = sm.robust.mad(varaux, axis=0)
        criterio1 = abs((varaux - varaux.median()) / mad) > 8
    
    # Calcula los cuartiles 1 (Q1) y 3 (Q3) para determinar el rango intercuartílico (H)
    qnt = varaux.quantile([0.25, 0.75]).dropna()
    Q1 = qnt.iloc[0]
    Q3 = qnt.iloc[1]
    H = 3 * (Q3 - Q1)
    
    # Identifica valores atípicos que están fuera del rango intercuartílico
    criterio2 = (varaux < (Q1 - H)) | (varaux > (Q3 + H))
    
    # Crea una copia de la serie original y reemplaza los valores atípicos por NaN
    var = varaux.copy()
    var[criterio1 & criterio2] = np.nan
    
    # Retorna la serie con valores atípicos reemplazados y el número de valores atípicos identificados
    return [var, sum(criterio1 & criterio2)]

#--------------------------------------------------------------------------------------------------------------
def patron_perdidos(datos_input):
    """
    Visualiza un mapa de calor que muestra la matriz de correlación de valores ausentes en el conjunto de datos.

    Args:
        datos_input (DataFrame): El conjunto de datos de entrada.

    """
    # Calculo una matriz de correlación de los valores ausentes en las columnas con al menos un missing
    correlation_matrix = datos_input[datos_input.columns[datos_input.isna().sum() > 0]].isna().corr()
    
    # Creo una máscara para ocultar la mitad superior de la matriz (simetría)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    # Configuro el tamaño de la figura y el tamaño de la fuente en el gráfico
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    
    # Genero un mapa de calor (heatmap) de la matriz de correlación de valores ausentes
    # 'annot=True' muestra los valores dentro de las celdas
    # 'cmap='coolwarm'' establece la paleta de colores del mapa de calor
    # 'fmt=".2f"' formatea los valores como números de punto flotante con dos decimales
    # 'cbar=False' oculta la barra de color (escala) en el lado derecho
    # 'mask=mask' aplica la máscara para ocultar la mitad superior de la matriz
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar=False, mask=mask)
    
    # Establezco el título del gráfico
    plt.title("Matriz de correlación de valores ausentes")
    plt.show()

#--------------------------------------------------------------------------------------------------------------
def ImputacionCuant(var, tipo):
    """
    Esta función realiza la imputación de valores faltantes en una variable cuantitativa.

    Datos de entrada:
    - var: Serie de datos cuantitativos con valores faltantes a imputar.
    - tipo: Tipo de imputación ('media', 'mediana' o 'aleatorio').

    Datos de salida:
    - Una nueva serie con valores faltantes imputados.
    """

    # Realiza una copia de la variable para evitar modificar la original
    vv = var.copy()

    if tipo == 'media':
        # Imputa los valores faltantes con la media de la variable
        vv[np.isnan(vv)] = round(np.nanmean(vv), 4)
    elif tipo == 'mediana':
        # Imputa los valores faltantes con la mediana de la variable
        vv[np.isnan(vv)] = round(np.nanmedian(vv), 4)
    elif tipo == 'aleatorio':
        # Imputa los valores faltantes de manera aleatoria basada en la distribución de valores existentes
        x = vv[~np.isnan(vv)]
        frec = x.value_counts(normalize=True).reset_index()
        frec.columns = ['Valor', 'Frec']
        frec = frec.sort_values(by='Valor')
        frec['FrecAcum'] = frec['Frec'].cumsum()
        random_values = np.random.uniform(min(frec['FrecAcum']), 1, np.sum(np.isnan(vv)))
        imputed_values = list(map(lambda x: list(frec['Valor'][frec['FrecAcum'] <= x])[-1], random_values))
        vv[np.isnan(vv)] = [round(x, 4) for x in imputed_values]

    return vv
#--------------------------------------------------------------------------------------------------------------
def ImputacionCuali(var, tipo):
    """
    Esta función realiza la imputación de valores faltantes en una variable cualitativa.

    Datos de entrada:
    - var: Serie de datos cualitativos con valores faltantes a imputar.
    - tipo: Tipo de imputación ('moda' o 'aleatorio').

    Datos de salida:
    - Una nueva serie con valores faltantes imputados.
    """

    # Realiza una copia de la variable para evitar modificar la original
    vv = var.copy()

    if tipo == 'moda':
        # Imputa los valores faltantes con la moda (valor más frecuente)
        frecuencias = vv[~vv.isna()].value_counts()
        moda = frecuencias.index[np.argmax(frecuencias)]
        vv[vv.isna()] = moda
    elif tipo == 'aleatorio':
        # Imputa los valores faltantes de manera aleatoria a partir de valores no faltantes
        vv[vv.isna()] = np.random.choice(vv[~vv.isna()], size=np.sum(vv.isna()), replace=True)

    return vv