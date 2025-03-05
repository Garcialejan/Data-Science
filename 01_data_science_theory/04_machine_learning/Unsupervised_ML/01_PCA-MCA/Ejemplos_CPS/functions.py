import pandas as pd
import re
import numpy as np

from scipy.stats import chi2_contingency
from itertools import combinations


def estandarizar_columnas(df):
    """
    Función para estandarizar los nombres de las columnas de un DataFrame.
    - Reemplaza caracteres especiales y acentuados.
    - Convierte todo a minúsculas.
    - Elimina saltos de línea y otros caracteres especiales.
    
    Parámetros:
    df : DataFrame cuyas columnas serán estandarizadas.

    Retorna:
    df : Dataframe con las columnas estandarizadas
    
    """
    df.columns = (
        df.columns
        .str.replace("\n", " ", regex=False)   # Eliminar saltos de línea
        .str.lower()                           # Convertir todo a minúsculas
        .str.replace("á", "a", regex=False)    # Reemplazar caracteres con acentos
        .str.replace("é", "e", regex=False)
        .str.replace("í", "i", regex=False)
        .str.replace("ó", "o", regex=False)
        .str.replace("ú", "u", regex=False)
        .str.replace("ñ", "n", regex=False)
        .str.replace("ç", "c", regex=False)
        .str.replace("Á", "a", regex=False)    # Reemplazar caracteres en mayúsculas
        .str.replace("É", "e", regex=False)
        .str.replace("Í", "i", regex=False)
        .str.replace("Ó", "o", regex=False)
        .str.replace("Ú", "u", regex=False)
        .str.replace("Ñ", "n", regex=False)
        .str.replace("Ç", "c", regex=False)
    )
    return df



def drop_columns_df(df, lista_columns):
    """
    Función para eliminar columnas a partir de una lista.
    
    Parámetros:
        df : DataFrame cuyas columnas serán eliminadas.
        lista_columns : lista de columnas a eliminar.
    
    Retorna:
        df : Dataframe con las columnas finales.
    """
    df_good = df.drop(columns = lista_columns, axis = 1, inplace = True)
    return df_good



def redondear_decena(valor):
    """
    Función para redondear un número a su decena más proxima.
    
    Parámetros:
        valor : float o integer que queremos redondear 

    Retorna:
        valor : valor redondeado a la decena más proxima.
    """
    if valor % 10 != 0:
        return round(valor, -1)
    return valor  # Si ya está redondeado, devolver el valor original



def replace_nan(df, lista_columnas, valor_reemplazo):
    """
    Función para reemplazar los valores nulos de las columnas definidas.
    
    Parámetros:
        df : DataFrame a tratar
        lista_columnas : lista de columnas del dataframe en las que se van a reemplazar los nan
        valor_reemplazo : valor por el que los Nan va a ser reemplazados

    Retorna:
        df : Dataframe con los valores nulos reemplazados.
    """
    df[lista_columnas] = df[lista_columnas].fillna(valor_reemplazo)
    return df



def clasificar_vehiculos(df, tipos_vehiculo, nombre_columna_tipo_vehiculo):
    """
    Función para clasificar los tipos de vehículos en el DataFrame.
    
    Esta función agrega columnas al DataFrame indicando si cada vehículo pertenece
    a una categoría específica. Para cada tipo de vehículo proporcionado, se 
    asigna un valor de 1 si el tipo de vehículo está en la lista correspondiente
    y 0 en caso contrario. Elimina la columna original con los tipos de vehículos
    
    Parámetros:
    df : pd.DataFrame
        DataFrame que contiene una columna 'tipo de vehiculo'.
        
    tipos_vehiculo : dict
        Diccionario donde las claves son los nombres de las categorías y los
        valores son listas de tipos de vehículos que pertenecen a cada categoría.
    
    Retorna:
    pd.DataFrame
        DataFrame con columnas adicionales que indican la clasificación de vehículos.
    """
    for tipo, nombres in tipos_vehiculo.items():
        df[tipo] = np.where(df[nombre_columna_tipo_vehiculo].isin(nombres), 1, 0)

    return df.info()



def clasificar_víctimas(df, tipos_victima, nombre_columna_tipo_lesion):
    """
    Función para clasificar los tipos de víctiams en el DataFrame de personas.
    
    Esta función agrega columnas al DataFrame indicando si un tipo de lesión pertenece
    a una categoría específica(mortal, grave, leve). Para cada persona se 
    asigna un valor de 1 si el tipo de lesión está en la lista correspondiente
    y 0 en caso contrario. Elimina la columna original con los tipos de vehículos
    
    Parámetros:
    df : pd.DataFrame
        DataFrame que contiene una columna 'tipo de vehiculo'.
        
    tipos_victimas : dict
        Diccionario donde las claves son los nombres del tipo de víctima y los
        valores son listas de tipos de lesiones que pertenecen a cada categoría.
    
    Retorna:
    pd.DataFrame
        DataFrame con columnas adicionales que indican la clasificación de víctimas.
    """
    for tipo, nombres in tipos_victima.items():
        df[tipo] = np.where(df[nombre_columna_tipo_lesion].isin(nombres), 1, 0)

    return df



# Función que evalúa el tipo de víctima por fila
def asignar_tipo_victima(row):
    if row['mortal'] > 0:
        return 'victima mortal'
    elif row['grave'] > 0:
        return 'victima grave'
    elif row['leve'] > 0:
        return 'victima leve'
    elif row['ileso'] > 0:
        return 'ileso'
    elif row['se desconoce'] > 0:
        return 'se desconoce'
    


# Función que evalúa el tipo de víctima por fila
def asignar_tipo_victima_2(row):
    if row['victimas mortales']:
        return 'victima mortal'
    elif row['victimas graves']:
        return 'victima grave'
    elif row['victimas leves']:
        return 'victima leve'
    elif row['ilesos']:
        return 'ileso'
    else:
        return 'se desconoce'
    


# Función que asigna un rango en función del total de personas de un accidente
def asignar_rango_personas(personas):
    if personas <= 3:
        return "Menos de 3"
    elif personas <= 6:
        return "Entre 3 y 6"
    else:
        return "Más de 6"



# Función que asigna un rango en función del total de víctimas de un accidente
def asignar_rango_victimas(personas):
    if personas <= 3:
        return "Menos de 3"
    elif personas <= 6:
        return "Entre 3 y 6"
    else:
        return "Más de 6"
    


# Función que asigna un rango en función de la hora del dia
def asignar_rango_horas(hora):
    if hora <= 6:
        return "Entre las 00 y las 6"
    elif hora <= 10:
        return "Entre las 7 y las 10"
    elif hora <= 14:
        return "Entre las 11 y las 14"
    elif hora <= 17:
        return "Entre las 15 y las 17"
    elif hora <= 21:
        return "Entre las 18 y las 21"
    else:
        return "Entre las 22 y las 00"
    


# Función que asigna un rango de velocidad en función del límite de velocidad
def asignar_rango_velocidad(velocidad):
    if velocidad <= 30:
        return "Hasta 30 Km/h"
    elif velocidad <= 60:
        return "Entre 30 y 60 km/h"
    elif velocidad <= 90:
        return "Entre 60 y 90 km/h"
    else:
        return "Limite de 120 km/h"



# Función para definir el rango de edad de las personas implicadas
def rango_edad(df, columna_edad, columna_rango="rango edad"):
    def clasificar_edad(edad):
        if edad <= 13:
            return "Menor de 13 años"
        elif edad <= 17:
            return "De 14 a 17 años"
        elif edad <= 24:
            return "De 18 a 24 años"
        elif edad <= 44:
            return "De 25 a 44 años"
        elif edad <= 64:
            return "De 45 a 64 años"
        elif edad <= 74:
            return "De 65 a 74 años"
        else:
            return "Más de 74 años"

    df[columna_rango] = df[columna_edad].apply(clasificar_edad)
    return df



# Función para definir el rango de edad de las personas implicadas
def rango_alcohol(df, columna_tasa, columna_rango="rango alcohol"):
    def clasificar_alcohol(tasa):
        if tasa == 0:
            return "No prueba/no positivo"
        elif tasa <= 0.15:
            return "Inferior a 0.15 mg/l"
        elif tasa <= 0.25:
            return "De 0.16 a 0.25 mg/l"
        elif tasa <= 0.5:
            return "De 0.26 a 0.5 mg/l"
        elif tasa <= 1:
            return "De 0.51 a 1 mg/l"
        else:
            return "Más de 1 mg/l"

    df[columna_rango] = df[columna_tasa].apply(clasificar_alcohol)
    return df



def agrupar_vehiculo(df, vehicle_columns):
    """
    Función para agrupar los tipos de vehículo en una sola columna
    categórica.
    
    Parámetros:
    df : DataFrame.
    vehicle_columns = lista de columnas que contienen booleanos con los vehículos

    Retorna:
    df : Dataframe con la columna de vehículos categorica y el resto eliminadas
    
    """
    # Crear una columna vacía para el tipo de vehículo
    df['tipo de vehiculo'] = ''

    # Iterar por cada fila y columna de vehículos
    for i, row in df.iterrows():
        for col in vehicle_columns:
            if row[col] == 1:
                df.at[i, 'tipo de vehiculo'] = col
                break  # Salir del bucle una vez se encuentra el primer tipo de vehículo

    # Eliminar las columnas de vehículos originales
    df = df.drop(columns=vehicle_columns)

    return df



def cramers(x, y):
    '''
    Función de Cramér's la cual se utiliza para medir la asociación entre dos variables categóricas.
    Es una medida basada en la Chi-cuadrado que normaliza el valor, obteniendo un rango de 0 a 1, 
    donde 0 indica independencia y 1 una relación fuerte entre las categorías. El test de Chi-cuadrado 
    compara las frecuencias observadas con las frecuencias esperadas bajo la hipótesis de que las 
    variables son independientes
    '''

    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))



def calculate_cramers_v_matrix(df):
    '''
    Función para calcular todas las correlaciones Cramér's V entre columnas categóricas
    '''
    
    # Crear una matriz de resultados
    result = pd.DataFrame(index=df.columns, columns=df.columns)

    # Iterar sobre todas las combinaciones de columnas
    for col1, col2 in combinations(df.columns, 2):
        # Calcular Cramér's V solo si ambas columnas son categóricas
        if df[col1].dtype == 'object' or df[col1].dtype.name == 'category':
            if df[col2].dtype == 'object' or df[col2].dtype.name == 'category':
                result.loc[col1, col2] = cramers(df[col1], df[col2])
                result.loc[col2, col1] = result.loc[col1, col2]  # Matriz simétrica
            else:
                result.loc[col1, col2] = np.nan
                result.loc[col2, col1] = np.nan
        else:
            result.loc[col1, col2] = np.nan
            result.loc[col2, col1] = np.nan
    
    # Rellenar la diagonal con 1 (correlación perfecta)
    np.fill_diagonal(result.values, 1)

    return result