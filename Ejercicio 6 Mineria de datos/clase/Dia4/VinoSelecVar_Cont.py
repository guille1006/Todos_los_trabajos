# Cargo las librerias 
import os
import pickle
from sklearn.model_selection import train_test_split
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

os.chdir(r'C:\Users\danie\Desktop\Universidad\Docencia\Título Propio Minería\TP_MDMP_1\Datos')

# Cargo las funciones que voy a utilizar
from FuncionesMineria import (Rsq, lm, lm_forward, lm_backward, lm_stepwise, validacion_cruzada_lm,
                           crear_data_modelo)

# Cargo los datos depurados
with open('Vinotodo_cont.pickle', 'rb') as f:
    todo = pickle.load(f)

# Identifico la variable objetivo y la elimino del conjunto de datos
varObjCont = todo['Beneficio']
todo = todo.drop('Beneficio', axis = 1)

# Identifico las variables continuas
var_cont = ['Acidez', 'AcidoCitrico', 'Azucar', 'CloruroSodico', 'Densidad', 'pH', 
            'Sulfatos', 'Alcohol',  'PrecioBotella', 'sqrxAcidez', 
            'expxAcidoCitrico', 'logxAzucar', 'sqrxCloruroSodico', 'xDensidad', 
            'sqrxpH', 'xSulfatos', 'xAlcohol', 'xPrecioBotella']

# Identifico las variables continuas sin transformar
var_cont_sin_transf = ['Acidez', 'AcidoCitrico', 'Azucar', 'CloruroSodico', 'Densidad', 'pH', 
                       'Sulfatos', 'Alcohol',  'PrecioBotella']

# Identifico las variables categóricas
var_categ = ['Etiqueta', 'CalifProductor', 'Clasificacion', 'Region', 'prop_missings']

# Hago la particion
x_train, x_test, y_train, y_test = train_test_split(todo, varObjCont, test_size = 0.2, random_state = 1234567)

# Construyo el modelo ganador del dia 2
modeloManual = lm(y_train, x_train, [], ['Clasificacion', 'Etiqueta', 'CalifProductor'])
# Resumen del modelo
modeloManual['Modelo'].summary()
# R-squared del modelo para train
Rsq(modeloManual['Modelo'], y_train, modeloManual['X'])
# Preparo datos test
x_test_modeloManual = crear_data_modelo(x_test, [], ['Clasificacion', 'Etiqueta', 'CalifProductor'])
# R-squared del modelo para test
Rsq(modeloManual['Modelo'], y_test, x_test_modeloManual)

# Posible unión de categorías (no del todo necesario en este ejemplo)
todo['CalifProductor'] = todo['CalifProductor'].replace({'2': '2-3', '3': '2-3'})

# Repitp la particion y modelo Inicial
x_train, x_test, y_train, y_test = train_test_split(todo, varObjCont, test_size = 0.2, random_state = 1234567)



# Seleccion de variables Stepwise, métrica AIC
modeloStepAIC = lm_stepwise(y_train, x_train, var_cont_sin_transf, var_categ, [], 'AIC')
# Resumen del modelo
modeloStepAIC['Modelo'].summary()
# Preparo datos test
x_test_modeloStepAIC = crear_data_modelo(x_test, modeloStepAIC['Variables']['cont'], 
                                                modeloStepAIC['Variables']['categ'], 
                                                modeloStepAIC['Variables']['inter'])
# R-squared del modelo para test
Rsq(modeloStepAIC['Modelo'], y_test, x_test_modeloStepAIC)

# Seleccion de variables Backward, métrica AIC
modeloBackAIC = lm_backward(y_train, x_train, var_cont_sin_transf, var_categ, [], 'AIC')
modeloBackAIC['Modelo'].summary()
x_test_modeloBackAIC = crear_data_modelo(x_test, modeloBackAIC['Variables']['cont'], 
                                                modeloBackAIC['Variables']['categ'], 
                                                modeloBackAIC['Variables']['inter'])
Rsq(modeloBackAIC['Modelo'], y_test, x_test_modeloBackAIC)

# Comparo número de parámetros (iguales)
len(modeloStepAIC['Modelo'].params)
len(modeloBackAIC['Modelo'].params)


# Mismas variables seleccionadas, mismos parámetros, mismo modelo.
modeloStepAIC['Modelo'].params
modeloBackAIC['Modelo'].params

# Seleccion de variables Stepwise, métrica BIC
modeloStepBIC = lm_stepwise(y_train, x_train, var_cont_sin_transf, var_categ, [], 'BIC')
# Resumen del modelo
modeloStepBIC['Modelo'].summary()
# Preparo datos test
x_test_modeloStepBIC = crear_data_modelo(x_test, modeloStepBIC['Variables']['cont'], 
                                                modeloStepBIC['Variables']['categ'], 
                                                modeloStepBIC['Variables']['inter'])
# R-squared del modelo para test
Rsq(modeloStepBIC['Modelo'], y_test, x_test_modeloStepBIC)

# Seleccion de variables Backward, métrica BIC
modeloBackBIC = lm_backward(y_train, x_train, var_cont_sin_transf, var_categ, [], 'BIC')
# Resumen del modelo
modeloBackBIC['Modelo'].summary()
# Preparo datos test
x_test_modeloBackBIC = crear_data_modelo(x_test, modeloBackBIC['Variables']['cont'], 
                                                modeloBackBIC['Variables']['categ'], 
                                                modeloBackBIC['Variables']['inter'])
# R-squared del modelo para test
Rsq(modeloBackBIC['Modelo'], y_test, x_test_modeloBackBIC)

# Comparo número de parámetros
len(modeloStepBIC['Modelo'].params)
len(modeloStepBIC['Modelo'].params)


# Mismas variables seleccionadas, mismos parámetros, mismo modelo.
# Los metodos Stepwise y Backward han resultado ser iguales.
modeloStepBIC['Modelo'].params
modeloBackBIC['Modelo'].params

# Comparo (R-squared)
Rsq(modeloStepAIC['Modelo'], y_test, x_test_modeloStepAIC)
Rsq(modeloStepBIC['Modelo'], y_test, x_test_modeloStepBIC)

# Nos quedamos con modeloStepBIC=modeloBackBIC, tienen similar R-squared pero menos parámetros


# Interacciones 2 a 2 de todas las variables (excepto las continuas transformadas)
interacciones = var_cont_sin_transf + var_categ
interacciones_unicas = list(itertools.combinations(interacciones, 2)) 
  
# Seleccion de variables Stepwise, métrica AIC, con interacciones
modeloStepAIC_int = lm_stepwise(y_train, x_train, var_cont_sin_transf, var_categ, 
                                interacciones_unicas, 'AIC')
# Resumen del modelo
modeloStepAIC_int['Modelo'].summary()
# Preparo datos test
x_test_modeloStepAIC_int = crear_data_modelo(x_test, modeloStepAIC_int['Variables']['cont'], 
                                                    modeloStepAIC_int['Variables']['categ'], 
                                                    modeloStepAIC_int['Variables']['inter'])
# R-squared del modelo para test
Rsq(modeloStepAIC_int['Modelo'], y_test, x_test_modeloStepAIC_int)

# Seleccion de variables Stepwise, métrica BIC, con interacciones
modeloStepBIC_int = lm_stepwise(y_train, x_train, var_cont_sin_transf, var_categ,
                                interacciones_unicas, 'BIC')
# Resumen del modelo
modeloStepBIC_int['Modelo'].summary()
# Preparo datos test
x_test_modeloStepBIC_int = crear_data_modelo(x_test, modeloStepBIC_int['Variables']['cont'], 
                                                    modeloStepBIC_int['Variables']['categ'], 
                                                    modeloStepBIC_int['Variables']['inter'])
# R-squared del modelo para test
Rsq(modeloStepBIC_int['Modelo'], y_test, x_test_modeloStepBIC_int)
  
# Comparo número de parámetros  
# Por el principio de parsimonia, es preferible el modeloStepBIC_int
len(modeloStepAIC_int['Modelo'].params)
len(modeloStepBIC_int['Modelo'].params)


# Pruebo con todas las transf y las variables originales, métrica AIC
modeloStepAIC_trans = lm_stepwise(y_train, x_train, var_cont, var_categ, [], 'AIC')
# Resumen del modelo
modeloStepAIC_trans['Modelo'].summary()
# Preparo datos test
x_test_modeloStepAIC_trans = crear_data_modelo(x_test, modeloStepAIC_trans['Variables']['cont'], 
                                                      modeloStepAIC_trans['Variables']['categ'], 
                                                      modeloStepAIC_trans['Variables']['inter'])
# R-squared del modelo para test
Rsq(modeloStepAIC_trans['Modelo'], y_test, x_test_modeloStepAIC_trans)

# Pruebo con todas las transf y las variables originales, métrica BIC
modeloStepBIC_trans = lm_stepwise(y_train, x_train, var_cont, var_categ, [], 'BIC')
# Resumen del modelo
modeloStepBIC_trans['Modelo'].summary()
# Preparo datos test
x_test_modeloStepBIC_trans = crear_data_modelo(x_test, modeloStepBIC_trans['Variables']['cont'], 
                                                      modeloStepBIC_trans['Variables']['categ'], 
                                                      modeloStepBIC_trans['Variables']['inter'])
# R-squared del modelo para test
Rsq(modeloStepBIC_trans['Modelo'], y_test, x_test_modeloStepBIC_trans)

# Comparo número de parámetros  
# No está claro cual es mejor
len(modeloStepAIC_trans['Modelo'].params)
len(modeloStepBIC_trans['Modelo'].params)

# Pruebo modelo con las Transformaciones y las interacciones, métrica AIC
modeloStepAIC_transInt = lm_stepwise(y_train, x_train, var_cont, var_categ, interacciones_unicas, 'AIC')
# Resumen del modelo
modeloStepAIC_transInt['Modelo'].summary()
# Preparo datos test
x_test_modeloStepAIC_transInt = crear_data_modelo(x_test, modeloStepAIC_transInt['Variables']['cont'], 
                                                         modeloStepAIC_transInt['Variables']['categ'], 
                                                         modeloStepAIC_transInt['Variables']['inter'])
# R-squared del modelo para test
Rsq(modeloStepAIC_transInt['Modelo'], y_test, x_test_modeloStepAIC_transInt)
# Pruebo modelo con las Transformaciones y las interacciones, métrica BIC
modeloStepBIC_transInt = lm_stepwise(y_train, x_train, var_cont, var_categ, interacciones_unicas, 'BIC')
# Resumen del modelo
modeloStepBIC_transInt['Modelo'].summary()
# Preparo datos test
x_test_modeloStepBIC_transInt = crear_data_modelo(x_test, modeloStepBIC_transInt['Variables']['cont'], 
                                                         modeloStepBIC_transInt['Variables']['categ'], 
                                                         modeloStepBIC_transInt['Variables']['inter'])
# R-squared del modelo para test
Rsq(modeloStepBIC_transInt['Modelo'], y_test, x_test_modeloStepBIC_transInt)

# Comparo número de parámetros  
# Por el principio de parsimonia, es preferible el modeloStepBIC_transInt
len(modeloStepAIC_transInt['Modelo'].params)
len(modeloStepBIC_transInt['Modelo'].params)


# Hago validacion cruzada repetida para ver que modelo es mejor
# Crea un DataFrame vacío para almacenar resultados
results = pd.DataFrame({
    'Rsquared': []
    , 'Resample': []
    , 'Modelo': []
})
# Realiza el siguiente proceso 20 veces (representado por el bucle `for rep in range(20)`)

for rep in range(20):
    # Realiza validación cruzada en cuatro modelos diferentes y almacena sus R-squared en listas separadas

    modelo_manual = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloManual['Variables']['cont']
        , modeloManual['Variables']['categ']
    )
    modelo_stepBIC = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloStepBIC['Variables']['cont']
        , modeloStepBIC['Variables']['categ']
    )
    modelo_stepBIC_int = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloStepBIC_int['Variables']['cont']
        , modeloStepBIC_int['Variables']['categ']
        , modeloStepBIC_int['Variables']['inter']
    )
    modelo_stepAIC_trans = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloStepAIC_trans['Variables']['cont']
        , modeloStepAIC_trans['Variables']['categ']
    )
    modelo_stepBIC_trans = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloStepBIC_trans['Variables']['cont']
        , modeloStepBIC_trans['Variables']['categ']
    )
    modelo_stepBIC_transInt = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloStepBIC_transInt['Variables']['cont']
        , modeloStepBIC_transInt['Variables']['categ']
        , modeloStepBIC_transInt['Variables']['inter']
    )
    # Crea un DataFrame con los resultados de validación cruzada para esta repetición

    results_rep = pd.DataFrame({
        'Rsquared': modelo_manual + modelo_stepBIC + modelo_stepBIC_int + modelo_stepBIC_trans + modelo_stepBIC_trans + modelo_stepBIC_transInt
        , 'Resample': ['Rep' + str((rep + 1))]*5*6 # Etiqueta de repetición (5 repeticiones 6 modelos)
        , 'Modelo': [1]*5 + [2]*5 + [3]*5 + [4]*5 + [5]*5 + [6]*5 # Etiqueta de modelo (6 modelos 5 repeticiones)
    })
    results = pd.concat([results, results_rep], axis = 0)
    
# Boxplot de la validacion cruzada 
plt.figure(figsize=(10, 6))  # Crea una figura de tamaño 10x6
plt.grid(True)  # Activa la cuadrícula en el gráficoç
# Agrupa los valores de Rsquared por modelo
grupo_metrica = results.groupby('Modelo')['Rsquared']
# Organiza los valores de R-squared por grupo en una lista
boxplot_data = [grupo_metrica.get_group(grupo).tolist() for grupo in grupo_metrica.groups]
# Crea un boxplot con los datos organizados
plt.boxplot(boxplot_data, labels=grupo_metrica.groups.keys())  # Etiqueta los grupos en el boxplot
# Etiqueta los ejes del gráfico
plt.xlabel('Modelo')  # Etiqueta del eje x
plt.ylabel('Rsquared')  # Etiqueta del eje y
plt.show()  # Muestra el gráfico  

# Calcular la media de las métricas R-squared por modelo
media_r2 = results.groupby('Modelo')['Rsquared'].mean()
# Calcular la desviación estándar de las métricas R-squared por modelo
std_r2 = results.groupby('Modelo')['Rsquared'].std()
# Contar el número de parámetros en cada modelo
num_params = [len(modeloManual['Modelo'].params), len(modeloStepAIC['Modelo'].params), len(modeloStepBIC_int['Modelo'].params), 
 len(modeloStepAIC_trans['Modelo'].params), len(modeloStepBIC_trans['Modelo'].params), 
 len(modeloStepBIC_transInt['Modelo'].params)]

# Todos los modelos son parecidos en cuanto a R-squared y su desviación estandar
# descartamos el modeloStepBIC_int y modeloStepBIC_transInt por su elevado número de parámetros
# elijo el modeloStepBIC_trans por su reducido número de parámetros.

## Seleccion aleatoria (se coge la submuestra de los datos de entrenamiento)

# Inicializar un diccionario para almacenar las fórmulas y variables seleccionadas.
variables_seleccionadas = {
    'Formula': [],
    'Variables': []
}

# Realizar 30 iteraciones de selección aleatoria.
for x in range(30):
    print('---------------------------- iter: ' + str(x))
    
    # Dividir los datos de entrenamiento en conjuntos de entrenamiento y prueba.
    x_train2, x_test2, y_train2, y_test2 = train_test_split(x_train, y_train, 
                                                            test_size = 0.3, random_state = 1234567 + x)
    
    # Realizar la selección stepwise utilizando el criterio BIC en la submuestra.
    modelo = lm_stepwise(y_train2.astype(int), x_train2, var_cont, var_categ, interacciones_unicas, 'BIC')
    
    # Almacenar las variables seleccionadas y la fórmula correspondiente.
    variables_seleccionadas['Variables'].append(modelo['Variables'])
    variables_seleccionadas['Formula'].append(sorted(modelo['Modelo'].model.exog_names))

# Unir las variables en las fórmulas seleccionadas en una sola cadena.
variables_seleccionadas['Formula'] = list(map(lambda x: '+'.join(x), variables_seleccionadas['Formula']))
    
# Calcular la frecuencia de cada fórmula y ordenarlas por frecuencia.
frecuencias = Counter(variables_seleccionadas['Formula'])
frec_ordenada = pd.DataFrame(list(frecuencias.items()), columns = ['Formula', 'Frecuencia'])
frec_ordenada = frec_ordenada.sort_values('Frecuencia', ascending = False).reset_index()

# Identificar las tres fórmulas más frecuentes y las variables correspondientes.
var_1 = variables_seleccionadas['Variables'][variables_seleccionadas['Formula'].index(
    frec_ordenada['Formula'][0])]
var_2 = variables_seleccionadas['Variables'][variables_seleccionadas['Formula'].index(
    frec_ordenada['Formula'][1])]
var_3 = variables_seleccionadas['Variables'][variables_seleccionadas['Formula'].index(
    frec_ordenada['Formula'][2])]

# ============================================================================
# De las 30 repeticiones, las 3 que más se repiten son:
#   1)  Clasificacion', 'CalifProductor', ('Etiqueta', 'Clasificacion'), ('Densidad', 'Etiqueta')
#   2)  'CalifProductor', Alcohol, ('Densidad', 'Clasificacion'), ('Etiqueta', 'Clasificacion'), ('Densidad', 'Etiqueta')
#   3) 'Clasificacion', 'CalifProductor', ('Etiqueta', 'Clasificacion'), ('Densidad', 'Etiqueta'), ('Acidez', 'pH')


## Comparacion final, tomo el ganador de antes y los nuevos candidatos
results = pd.DataFrame({
    'Rsquared': []
    , 'Resample': []
    , 'Modelo': []
})
for rep in range(20):
    modelo1 = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloStepBIC_trans['Variables']['cont']
        , modeloStepBIC_trans['Variables']['categ']
        , modeloStepBIC_trans['Variables']['inter']
    )
    modelo2 = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , var_1['cont']
        , var_1['categ']
        , var_1['inter']
    )
    modelo3 = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , var_2['cont']
        , var_2['categ']
        , var_2['inter']
    )
    modelo4 = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , var_3['cont']
        , var_3['categ']
        , var_3['inter']
    )
    results_rep = pd.DataFrame({
        'Rsquared': modelo1 + modelo2 + modelo3 + modelo4
        , 'Resample': ['Rep' + str((rep + 1))]*5*4
        , 'Modelo': [1]*5 + [2]*5 + [3]*5 + [4]*5
    })
    results = pd.concat([results, results_rep], axis = 0)
     

# Boxplot de la validacion cruzada 
plt.figure(figsize=(10, 6))  # Crea una figura de tamaño 10x6
plt.grid(True)  # Activa la cuadrícula en el gráficoç
# Agrupa los valores de Rsquared por modelo
grupo_metrica = results.groupby('Modelo')['Rsquared']
# Organiza los valores de R-squared por grupo en una lista
boxplot_data = [grupo_metrica.get_group(grupo).tolist() for grupo in grupo_metrica.groups]
# Crea un boxplot con los datos organizados
plt.boxplot(boxplot_data, labels=grupo_metrica.groups.keys())  # Etiqueta los grupos en el boxplot
# Etiqueta los ejes del gráfico
plt.xlabel('Modelo')  # Etiqueta del eje x
plt.ylabel('Rsquared')  # Etiqueta del eje y
plt.show()  # Muestra el gráfico  

# Calcular la media de las métricas R-squared por modelo
media_r2_v2 = results.groupby('Modelo')['Rsquared'].mean()
# Calcular la desviación estándar de las métricas R-squared por modelo
std_r2_v2 = results.groupby('Modelo')['Rsquared'].std()
# Contar el número de parámetros en cada modelo
num_params_v2 = [len(modeloStepBIC_trans['Modelo'].params), 
                 len(frec_ordenada['Formula'][0].split('+')),
                 len(frec_ordenada['Formula'][1].split('+')), 
                 len(frec_ordenada['Formula'][2].split('+'))]

# Una vez decidido el mejor modelo, hay que evaluarlo 
ModeloGanador = modeloStepBIC_trans

# Vemos los coeficientes del modelo ganador
ModeloGanador['Modelo'].summary()

# Evaluamos la estabilidad del modelo a partir de las diferencias en train y test
Rsq(ModeloGanador['Modelo'], y_train, ModeloGanador['X'])

x_test_modeloganador = crear_data_modelo(x_test, ModeloGanador['Variables']['cont'], 
                                                ModeloGanador['Variables']['categ'], 
                                                ModeloGanador['Variables']['inter'])
Rsq(ModeloGanador['Modelo'], y_test, x_test_modeloganador)
    
    
    
