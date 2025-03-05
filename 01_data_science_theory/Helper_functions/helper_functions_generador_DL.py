import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.ensemble import IsolationForest

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K


def read_pickle_df(path):
   with open(path, "rb") as file:
      df = pickle.load(file)
      return df


def encode_ciclicas(column_series,
                    n_valores: int):
    '''
    Function to encode the ciclical variables like days
    hours, months,... using a sinus and cosinus functions.

    Args:
    ---------------------------------------------------
    - column_series: column of a pd.Dataframe or np.series to
    encode.
    - n_valores: space between values. For example, 7 for day
    week.
    '''
    angulo_entre_valores = 2*np.pi/n_valores # 360 grados = 2pi radianes
    x = np.sin(column_series*angulo_entre_valores)
    y = np.cos(column_series*angulo_entre_valores)
    return x, y


def get_time_position_vars_updt(df_input, oe_salida, oe_retorno, oe_resto):
    # Agregar fecha de operaciones especiales TRAIN
    fechas_series_tr = pd.Series(df_input.index.date, index = df_input.index)

    # Agregar fechas de operaciones especiales de saldia, retorno y resto
    df_input["oe_salida"] = 0
    df_input.loc[fechas_series_tr.where(fechas_series_tr.isin(oe_salida)).dropna().index, "oe_salida"] = 1

    df_input["oe_retorno"] = 0
    df_input.loc[fechas_series_tr.where(fechas_series_tr.isin(oe_retorno)).dropna().index, "oe_retorno"] = 1

    df_input["oe_resto"] = 0
    df_input.loc[fechas_series_tr.where(fechas_series_tr.isin(oe_resto)).dropna().index, "oe_resto"] = 1

    # Agregar fechas Covid TRAIN
    df_input["covid"] = 0
    df_input.loc[((df_input.index >="2020-03-15")&(df_input.index < "2020-06-22"))|((df_input.index >="2020-10-25")&(df_input.index < "2021-05-09")), "covid"] = 1

    # Agregar variables relacionadas con el tiempo TRAIN
    df_input["dayofWeek"] = df_input.index.dayofweek
    df_input["month"] = df_input.index.month
    # df_input["hour"] = df_input.index.hour

    # Agregar variables cíclicas para hour, dayofWeek y month 
    def encode_ciclicas(v, n_valores):
        angulo_entre_valores = 2*np.pi/n_valores # 360 grados = 2pi radianes
        x = np.sin(v*angulo_entre_valores)
        y = np.cos(v*angulo_entre_valores)
        return x, y

    cols_cicl = [("dayofWeek", 7), ("month", 12) ] #, ("hour", 24)]
    for col, num, in cols_cicl:
        df_input[col+"_X"], df_input[col+"_Y"] = encode_ciclicas(df_input[col], num)
    # eliminar columnas categóricas
    df_input.drop(["month", "dayofWeek"], axis = 1 , inplace = True)

    return df_input


def hampel_filter(series,
                  window_size=15,
                  n_sigmas=5):
    '''
    Function to apply the Hampel filter over a 
    series (pd.Series or np.Series)

    Args:
    -------------------------------------------------
    - window_size:  The size of the moving window for 
    outlier detection (default 15 days).
    - n_sigmas: number of standard deviations for outlier
    detection (default is 5.0). By tuning this parameter 
    we can have more or less tolerance to possible outliers
    '''
    rolling_median = series.rolling(window=window_size, center=True).median()
    rolling_mad = (series - rolling_median).abs().rolling(window=window_size, center=True).median()
    threshold = n_sigmas * (1/0.6745) * rolling_mad
    outliers = (series - rolling_median).abs() > threshold

    return outliers.fillna(False)


def isolation_forest_outliers(data,
                              n_estimators:int=100,
                              contamination:float=0.05,
                              random_state:int=42) -> pd.Series:
    """
    Aplica el Isolation Forest para detectar outliers en una serie de datos.
    
    :param data: Serie de datos (pandas.Series).
    :param contamination: Proporción estimada de outliers en los datos.
    :param random_state: Semilla para la reproducibilidad.
    :return: Serie de enteros indicando si un valor es outlier (1) o no (0).
    """
    # Reshape los datos para que estén en el formato adecuado para Isolation Forest
    data_reshaped = data.values.reshape(-1, 1)
    
    # Inicializar el modelo de Isolation Forest
    model = IsolationForest(n_estimators=n_estimators,
                            contamination=contamination,
                            random_state=random_state,
                            n_jobs=-1)
    
    # Entrenar el modelo
    model.fit(data_reshaped)
    
    # Predecir outliers
    predictions = model.predict(data_reshaped)
    
    # Convertir las predicciones a 1 (outlier) y 0 (no outlier)
    outliers = np.where(predictions == -1, 1, 0)
    
    return pd.Series(outliers, index=data.index)

@tf.keras.utils.register_keras_serializable()
def r2_score(y_true, y_pred):
    """Calcula el R² score."""
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - ss_res / (ss_tot + tf.keras.backend.epsilon())


def inicializar_decoder(model_version:str,
                        nodes_combination: list[int],
                        input_shape: int,
                        output_shape: int, 
                        L1: float = 0,
                        L2: float = 0,
                        DROPOUT: float = 0,
                        learning_rate: float = 0.001):
    '''
    Esta función permite la creación y compilación de un modelo
    decoder para la generación de matrices OD a partir de datos
    de de viajes de detectores.

    Args:
    -----------------------------------------------------------
    - model_version: nombre de la versión que queramos darle al
    modelo cada vez que llamamos a la función.
    - NODES_COMBI: combinaciones de nodos para cada una de las
    capas de la red neuronal
    - L1: valor del factor de penalización de Lasso (reducción
    de overfitting)
    - L2: valor del factor de penalización de Ridge (reducción
    de overfitting)
    - DROPOUT: valor del factor de apagado aleatorio de nodos
    durante el entrenamiento (reducción de overfitting)

    Returns:
    -----------------------------------------------------------
    - modelo de tensorflow creado y compilado, listo para su
    entrenamiento 
    '''
    # Reiniciar el estado de Keras
    K.clear_session()
    tf.random.set_seed(42)
    tf.config.experimental.enable_op_determinism()

    INPUT_SHAPE = input_shape
    new_model = Sequential(name=f"Modelo_inicial_{model_version}")

    # Bucle para añadir capas a nuestro modelo
    for i, NODES in enumerate(nodes_combination):
        # Capa de entrada
        if i == 0:
            new_model.add(Dense(units=NODES, activation='relu', input_shape=INPUT_SHAPE,
                        kernel_regularizer=regularizers.L1L2(l1=L1, l2=L2)))
            if DROPOUT != 0:
                new_model.add(Dropout(DROPOUT))
        else:
            # Capas ocultas
            new_model.add(Dense(units=NODES, activation='relu',
                        kernel_regularizer=regularizers.L1L2(l1=L1, l2=L2)))
            if DROPOUT != 0:
                new_model.add(Dropout(DROPOUT))

    # Capa de salida (output layer)
    new_model.add(Dense(units=output_shape))

    # Compilar el modelo
    # Definimos el rmse y mape como nuestras métricas de validación junto al r2_score
    rmse = tf.keras.metrics.RootMeanSquaredError(name='rmse')
    mape = tf.keras.metrics.MeanAbsolutePercentageError(name='mape')

    new_model.compile(
        loss='mean_absolute_error',
        optimizer=tf.keras.optimizers.Adam(learning_rate= learning_rate), # RMSProp, SGD, Nadam, otro Stochastic...
        metrics=[rmse, r2_score, mape])
    return new_model


def train_decoder_model_unbatch(
        model,
        X_train, y_train,
        X_test, y_test,
        epochs: int,
        callbacks: list,
        verbose:int = 0):
    '''
    Función que inicializa el entrenamiento del modelo.
    Previamente se debe llamar a inicializar_decoder.

    Args:
    --------------------------------------------------
    - model: nombre del modelo a utilizar
    - X_train: features de entrenamiento
    - X_test: features de testeo
    - y_train: target de entrenamiento
    - y_test: target de testeo

    Returns:
    --------------------------------------------------
    - historia: objeto obtenido tras entrenamiento con
    información sobre métricas y función de pérdida. 
    - principales métricas de evaluación del modelo
    '''
    model_history = model.fit(
        x = X_train,
        y = y_train,
        epochs = epochs,
        validation_data = (X_test, y_test), 
        verbose = verbose,
        callbacks = callbacks
        )
    
    return model, model_history, {"mape_train": model_history.history['mape'][-1],
                                  "mape_tests":model_history.history['val_mape'][-1],
                                  "rmse_train": model_history.history['rmse'][-1],
                                  "rmse_tests":model_history.history['val_rmse'][-1],
                                  "R2_train":model_history.history['r2_score'][-1],
                                  "R2_test": model_history.history['val_r2_score'][-1]}


def plot_training_history(history):
  '''
  Función que permite plotear la función de pérdida y
  las métricas de los modelos que se han entrenado. En
  concreto se representa el RMSE Y EL R^2.

  Args:
  ---------------------------------------------------
  - model_history: historia del modelo que se quiere
  representar
  '''
  history_dict = history.history
  epochs = range(1, len(history_dict['loss']) + 1)

  # Graficar la pérdida de entrenamiento y validación
  plt.figure(figsize=(12, 3))

  # Pérdida de entrenamiento y validación
  plt.subplot(1, 2, 1)
  plt.plot(epochs, history_dict['loss'], 'b', label='Entrenamiento Loss')
  plt.plot(epochs, history_dict['val_loss'], 'r', label='Validación Loss')
  plt.title('Pérdida en Entrenamiento y Validación')
  plt.xlabel('Épocas')
  plt.ylabel('Pérdida')
  plt.legend()

  # Métrica de entrenamiento y validación (en este caso, MSE)
  plt.subplot(1, 2, 2)
  plt.plot(epochs, history_dict['r2_score'], 'b', label='Entrenamiento MSE')
  plt.plot(epochs, history_dict['val_r2_score'], 'r', label='Validación MSE')
  plt.title('R2 de Entrenamiento y Validación')
  plt.xlabel('Épocas')
  plt.ylabel('R2')
  plt.legend()

  plt.tight_layout()
  plt.show()



def plot_one_model_history(model_history,
                           metrics_rmse:str="rmse",
                           val_metrics_rmse:str="val_rmse"):
    """
    Devuelve curvas de pérdida separadas para
    las métricas de entrenamiento y validación.

    Args:
    - model_history: Objeto hystory del modelo 
    TensorFlow 
    """ 
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    r2_score = model_history.history['r2_score']
    val_r2_score = model_history.history['val_r2_score']

    rmse = model_history.history[metrics_rmse]
    val_rmse = model_history.history[val_metrics_rmse]

    epochs = range(len(model_history.history['loss']))

    # Plot loss
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot r2_score
    plt.figure()
    plt.plot(epochs, r2_score, label='training_r2_score')
    plt.plot(epochs, val_r2_score, label='val_r2_score')
    plt.title('R^2 coeficent')
    plt.xlabel('Epochs')
    plt.legend()
    
    # Plot rmse
    plt.figure()
    plt.plot(epochs, rmse, label='training_rmse')
    plt.plot(epochs, val_rmse, label='val_rmse')
    plt.title('RMSE score')
    plt.xlabel('Epochs')
    plt.legend();


def kde_plot_seaborn_comp(df_comprobacion:pd.DataFrame,
                          column_real:str,
                          column_pred:str,
                          x_lim:int):
    '''
    Function for plotting a KDE (Kernel Density Estimation) of
    real total_trips and predicted total_trips withg the decoder
    model.

    Args:
    ------------------------------------------------
    df_comprobacion: dataframe with the real and pred trips.
    column_real: name of column with real trips.
    column_pred: name of column with pred trips.
    x_lim: limit value of total_trips.
    '''
    # No mostramos todos los valores ya que si no se pierde información
    df_comprobacion = df_comprobacion[
    (df_comprobacion[column_real] < x_lim) & (df_comprobacion[column_pred] < x_lim)
    ]
    
    sns.kdeplot(df_comprobacion[column_real], label='Reales', fill=True, color='red', alpha=0.5)
    sns.kdeplot(df_comprobacion[column_pred], label='Predichos', fill=True, color='green', alpha=0.5)
    plt.title(f'Comparación de distribución de Viajes modelo decoder (total_trips < {x_lim})')
    plt.xlabel('Total de Viajes')
    plt.ylabel('Densidad')
    plt.legend(loc = "upper right")

    return plt.show()



def kde_provincial_plot_seaborn_comp(df_comprobacion:pd.DataFrame,
                                     column_real:str,
                                     column_pred:str,
                                     provincia_origen: str,
                                     provincia_destino: str):
    '''
    Function for plotting a KDE (Kernel Density Estimation) of
    real total_trips and predicted total_trips with the decoder
    model, both filtered for a P_O and P_D.  

    Args:
    ------------------------------------------------
    df_comprobacion: dataframe with the real and pred trips.
    column_real: name of column with real trips.
    column_pred: name of column with pred trips.
    '''
    df_comprobacion = df_comprobacion[df_comprobacion["P_O"].eq(provincia_origen)&\
                                       df_comprobacion["P_D"].eq(provincia_destino)]

    sns.kdeplot(df_comprobacion[column_real], label='Reales', fill=True, color='red', alpha=0.5)
    sns.kdeplot(df_comprobacion[column_pred], label='Predichos', fill=True, color='green', alpha=0.5)
    plt.title(f'Comparativa distribución de viajes entre P_O:{provincia_origen} y P_D:{provincia_destino}')
    plt.xlabel('Total de Viajes')
    plt.ylabel('Densidad')
    plt.legend(loc = "upper right")

    return plt.show()


def kde_date_plot_seaborn_comp(df_comprobacion:pd.DataFrame,
                               column_real:str,
                               column_pred:str,
                               date: str,
                               x_lim:int):
    '''
    Function for plotting a KDE (Kernel Density Estimation) of
    real total_trips and predicted total_trips with the decoder
    model, both filtered by date an x_lim.  

    Args:
    ------------------------------------------------
    df_comprobacion: dataframe with the real and pred trips.
    column_real: name of column with real trips.
    column_pred: name of column with pred trips.
    x_lim: limit value of total_trips.
    
    '''
    df_comprobacion = df_comprobacion[df_comprobacion["date"].eq(date)]

    # No mostramos todos los valores ya que si no se pierde información
    df_comprobacion = df_comprobacion[
    (df_comprobacion[column_real] < x_lim) & (df_comprobacion[column_pred] < x_lim)
    ]

    sns.kdeplot(df_comprobacion[column_real], label='Reales', fill=True, color='red', alpha=0.5)
    sns.kdeplot(df_comprobacion[column_pred], label='Predichos', fill=True, color='green', alpha=0.5)
    plt.title(f'Comparación de distribución de Viajes modelo decoder para el día: {date}')
    plt.xlabel('Total de Viajes')
    plt.ylabel('Densidad')
    plt.legend(loc = "upper right")

    return plt.show()



def single_kde_plot(df:pd.DataFrame,
                    trips_column:str,
                    x_lim:int):
    '''
    Function for plotting a KDE (Kernel Density Estimation) of
    real total_trips and predicted total_trips withg the decoder
    model.

    Args:
    ------------------------------------------------
    df: dataframe with trips.
    trips_column: name of column with total trips.
    x_lim: limit value of total_trips.
    '''
    # No mostramos todos los valores ya que si no se pierde información
    df = df[df[trips_column] < x_lim]
    
    sns.kdeplot(df[trips_column], label='Reales', fill=True, color='red', alpha=0.5)
    plt.title(f'Distribución de viajes (total_trips < {x_lim})')
    plt.xlabel('Total de Viajes')
    plt.ylabel('Densidad')
    plt.legend(loc = "upper right")
    return plt.show()

# * Función para modelo a escala logarítmica --------------------------------------------------------
@tf.keras.utils.register_keras_serializable()
def rmse_log(y_true, y_pred):
    y_true = K.exp(y_true) - 1  # Revertimos la transformación logarítmica
    y_pred = K.exp(y_pred) - 1
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

@tf.keras.utils.register_keras_serializable()
def mape_log(y_true, y_pred):
    y_true = K.exp(y_true) - 1  # Revertimos la transformación logarítmica
    y_pred = K.exp(y_pred) - 1
    return tf.reduce_mean(tf.abs((y_true - y_pred) / (y_true + K.epsilon()))) * 100

@tf.keras.utils.register_keras_serializable()
def mae_log(y_true, y_pred):
    # Volvemos a la escala original
    y_true_original = K.exp(y_true) - 1
    y_pred_original = K.exp(y_pred) - 1
    return K.mean(K.abs(y_true_original - y_pred_original))

def inicializar_decoder_log(model_version:str,
                        nodes_combination: list[int],
                        input_shape: int,
                        output_shape: int, 
                        L1: float = 0,
                        L2: float = 0,
                        DROPOUT: float = 0,
                        learning_rate: float = 0.001):
    '''
    Inicializa un modelo de red neuronal para predecir viajes en su escala original.
    '''

    # Reiniciar el estado de Keras
    K.clear_session()
    tf.random.set_seed(42)
    tf.config.experimental.enable_op_determinism()

    INPUT_SHAPE = input_shape
    new_model = tf.keras.Sequential(name=f"Modelo_inicial_{model_version}")

    # Añadir capas al modelo
    for i, NODES in enumerate(nodes_combination):
        if i == 0:
            new_model.add(tf.keras.layers.Dense(units=NODES, activation='relu', input_shape=INPUT_SHAPE,
                        kernel_regularizer=tf.keras.regularizers.L1L2(l1=L1, l2=L2)))
            if DROPOUT != 0:
                new_model.add(tf.keras.layers.Dropout(DROPOUT))
        else:
            new_model.add(tf.keras.layers.Dense(units=NODES, activation='relu',
                        kernel_regularizer=tf.keras.regularizers.L1L2(l1=L1, l2=L2)))
            if DROPOUT != 0:
                new_model.add(tf.keras.layers.Dropout(DROPOUT))

    # Capa de salida
    new_model.add(tf.keras.layers.Dense(units=output_shape))

    # Métricas personalizadas en escala original
    def rmse_original(y_true, y_pred):
        return K.sqrt(K.mean(K.square(K.exp(y_true) - 1 - (K.exp(y_pred) - 1))))

    def mape_original(y_true, y_pred):
        return K.mean(K.abs((K.exp(y_true) - 1 - (K.exp(y_pred) - 1)) / (K.exp(y_true) - 1 + K.epsilon()))) * 100

    # Compilar el modelo con la nueva función de pérdida y métricas en escala original
    new_model.compile(
        loss=mae_log,  # Ahora la pérdida se calcula en la escala original
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=[rmse_log, r2_score, mape_log]
    )
    
    return new_model

def train_decoder_model_unbatch_log(model,
                                    X_train, y_train,
                                    X_test, y_test,
                                    epochs: int,
                                    callbacks: list,
                                    verbose:int = 0):
    '''
    Función que inicializa el entrenamiento del modelo
    en base logarítmica.
    '''
    model_history = model.fit(
        x = X_train,
        y = y_train,
        epochs = epochs,
        validation_data = (X_test, y_test), 
        verbose = verbose,
        callbacks = callbacks
        )
    
    return model, model_history


#! #################################################################################################################################

# def create_model_checkpoint(model_name, save_path="model_experiments"):
#   '''
#     Funcion que nos permite crear un checkpoint de nuestros
#     modelos. Se define en esta función que solo se guarden
#     los patrones (pesos) con los que se obtiene la mejor
#     precisión posible.
#
#     Args:
#     ---------------------------------------------------
#     - model_name: nombre del modelo con el que queremos que se
#     guarde.
#     - save_path: ruta para guardar los pesos del modelo (no se
#     guarda la arquitectura, solo los pesos).
#     '''
#   return tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_path, model_name), 
#                                             save_weights_only=True,
#                                             save_best_only=True,
#                                             verbose=0) # limitamos el output por pantalla


# def train_decoder_model_prefetch(
#         model,
#         train_dataset,
#         test_dataset,
#         epochs: int,
#         callbacks: list,
#         batch_size = 32):
#     '''
#     Función que inicializa el entrenamiento del modelo.
#     Previamente se debe llamar a inicializar_decoder.
#
#     Args:
#     --------------------------------------------------
#     - model: nombre del modelo a utilizar
#     - X_train: features de entrenamiento
#     - X_test: features de testeo
#     - y_train: target de entrenamiento
#     - y_test: target de testeo
#
#     Returns:
#     --------------------------------------------------
#     - historia: objeto obtenido tras entrenamiento con
#     información sobre métricas y función de pérdida. 
#     - principales métricas de evaluación del modelo
#     '''
#     model_history = model.fit(
#         train_dataset,
#         validation_data= test_dataset,
#         epochs = epochs, 
#         verbose=1,
#         callbacks = callbacks
#         )
#
#     return model, model_history, {"rmse_train": model_history.history['rmse'][-1],
#                                   "rmse_tests":model_history.history['val_rmse'][-1],
#                                   "R2_train":model_history.history['r2_score'][-1],
#                                   "R2_test": model_history.history['val_r2_score'][-1]}