import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle


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



def data_imputer_KNN_parallel(df: pd.DataFrame,
                              columns_for_imputation: list[str],
                              n_neighbors: int) -> pd.DataFrame:
    '''
    This function is an updated function of "data_imputer". We use
    multithreading to process each P_O and P_D group in parallel
    and we impute data with with the KNNImputer.

    Args:
    ------------------------------------------------------------
    - df: dataframe for imputation
    - columns_for_imputation: the columns we use for the
    computation (previously encoded)
    - n_neigbhbors: number of neighbors used for the imputation
    '''
    imputed_data = []

    def impute_group(P_O, P_D, group):
        try:
            group_imputed = group.copy()
            # Verificar si hay columnas con solo valores NaN en este grupo
            if group[columns_for_imputation].isnull().all().any():
                return group_imputed 
            
            imputer = KNNImputer(n_neighbors=n_neighbors)
            group_imputed[columns_for_imputation] = imputer.fit_transform(group[columns_for_imputation])
            return group_imputed
        
        except Exception as e:
            print(f"Error imputing group ({P_O}, {P_D}): {e}")
            return group  # Devolver el grupo sin cambios en caso de error

    # Usamos ThreadPoolExecutor para procesar los grupos en paralelo
    with ThreadPoolExecutor() as executor:
        futures = []
        # Iterar sobre cada grupo de P_O y P_D
        for (P_O, P_D), group in df.groupby(['P_O', 'P_D']):
            future = executor.submit(impute_group, P_O, P_D, group)
            futures.append(future)
        
        # Recoger los resultados de los hilos
        for future in as_completed(futures):
            imputed_data.append(future.result())

    # Concatenar todos los grupos imputados en un solo DataFrame
    df_imputed = pd.concat(imputed_data).sort_index().reset_index(drop=True)

    return df_imputed


def kde_plot_seaborn(df_imputed:pd.DataFrame,
                     df_no_imputed:pd.DataFrame,
                     column_x:str,
                     x_lim: int):
    '''
    Function for plotting a KDE (Kernel Density Estimation) of
    total_trips before and after imputation with the KNNImputer
    with a filter in total_trips.

    Args:
    ------------------------------------------------
    df_imputed: data after imputation.
    df_no_imputed: data before imputation.
    column_x: column of the dfs which represents x axis
    x_lim: limit value of total_trips 
    '''
    df_no_imputado_filtered = df_no_imputed[df_no_imputed[column_x] < x_lim]
    df_imputado_filtered = df_imputed[df_imputed[column_x] < x_lim]

    sns.kdeplot(df_imputado_filtered[column_x], label=f'Imputados', fill=True, color='red', alpha=0.5)
    sns.kdeplot(df_no_imputado_filtered[column_x], label='No Imputados', fill=True, color='blue', alpha=0.5)
    plt.title(f'Distribución de Viajes Antes y después de la Imputación (total_trips < {x_lim})')
    plt.xlabel('Total de Viajes')
    plt.ylabel('Densidad')
    plt.legend(loc = "upper right")

    return plt.show()



def kde_plot_seaborn_comp(df_imputed_1:pd.DataFrame,
                          df_imputed_2:pd.DataFrame,
                          df_no_imputed:pd.DataFrame,
                          column_x:str,
                          x_lim: int,
                          n_vecinos_1:int,
                          n_vecinos_2:int):
    '''
    Function for plotting a KDE (Kernel Density Estimation) of
    total_trips before and after imputation with the KNNImputer
    with a filter in total_trips. Plots diferents KDE based on
    the number of neighbors

    Args:
    ------------------------------------------------
    df_imputed_1: data after imputation with n_vecinos_1.
    df_imputed_2: data after imputation with n_vecinos_2.
    df_no_imputed: data before imputation.
    column_x: column of the dfs which represents x axis
    x_lim: limit value of total_trips 
    '''
    df_no_imputado_filtered = df_no_imputed[df_no_imputed[column_x] < x_lim]
    df_imputado_filtered_1 = df_imputed_1[df_imputed_1[column_x] < x_lim]
    df_imputado_filtered_2 = df_imputed_2[df_imputed_2[column_x] < x_lim]

    sns.kdeplot(df_imputado_filtered_1[column_x], label=f'Imputados {n_vecinos_1}', fill=True, color='red', alpha=0.5)
    sns.kdeplot(df_imputado_filtered_2[column_x], label=f'Imputados {n_vecinos_2}', fill=True, color='green', alpha=0.2)
    sns.kdeplot(df_no_imputado_filtered[column_x], label='No Imputados', fill=True, color='blue', alpha=0.5)
    plt.title(f'Distribución de Viajes Antes y después de la Imputación (total_trips < {x_lim})')
    plt.xlabel('Total de Viajes')
    plt.ylabel('Densidad')
    plt.legend(loc = "upper right")

    return plt.show()



def kde_provincial_plot_seaborn_comp(df_imputed_1:pd.DataFrame,
                                     df_imputed_2:pd.DataFrame,
                                     df_no_imputed:pd.DataFrame,
                                     column_x:str,
                                     provincia_origen: str,
                                     provincia_destino: str,
                                     n_vecinos_1:int,
                                     n_vecinos_2:int):
    '''
    Function for plotting a KDE (Kernel Density Estimation) of
    total_trips before and after imputation with the KNNImputer
    with a filter in total_trips and filtered by PO-PD. Added
    new functionality: compare between data imputed

    Args:
    ------------------------------------------------
    df_imputed_1: data after imputation with n_vecinos_1.
    df_imputed_2: data after imputation with n_vecinos_2.
    df_no_imputed: data before imputation.
    column_x: column of the dfs which represents x axis
    '''
    df_no_imputed_filt = df_no_imputed[df_no_imputed["P_O"].eq(provincia_origen)&\
                                       df_no_imputed["P_D"].eq(provincia_destino)]
    
    df_imputed_filt_1 = df_imputed_1[df_imputed_1["P_O"].eq(provincia_origen)&\
                                 df_imputed_1["P_D"].eq(provincia_destino)]
    
    df_imputed_filt_2 = df_imputed_2[df_imputed_2["P_O"].eq(provincia_origen)&\
                                 df_imputed_2["P_D"].eq(provincia_destino)]

    sns.kdeplot(df_imputed_filt_1[column_x], label=f'Imputados {n_vecinos_1}', fill=True, color='red', alpha=0.5)
    sns.kdeplot(df_imputed_filt_2[column_x], label=f'Imputados {n_vecinos_2}', fill=True, color='green', alpha=0.2)
    sns.kdeplot(df_no_imputed_filt[column_x], label='No Imputados', fill=True, color='blue', alpha=0.5)
    plt.title(f'Comparativa distribución de viajes entre P_O:{provincia_origen} y P_D:{provincia_destino}')
    plt.xlabel('Total de Viajes')
    plt.ylabel('Densidad')
    plt.legend(loc = "upper right")

    return plt.show()



def comparacion_imputacion_provincias(provincia_origen: str,
                                      provincia_destino: str,
                                      df_imputed: pd.DataFrame,
                                      df_no_imputed:pd.DataFrame,
                                      valor_min: int,
                                      valor_max:int,
                                      nbins:int):
    '''
    Function to plot a histogram filtered by the PO-PD. The objective
    is see how imputations works for an especific PO-PD.
    '''
    df_no_imputed_filt = df_no_imputed[df_no_imputed["P_O"].eq(provincia_origen)&\
                                       df_no_imputed["P_D"].eq(provincia_destino)]
    
    df_imputed_filt = df_imputed[df_imputed["P_O"].eq(provincia_origen)&\
                                 df_imputed["P_D"].eq(provincia_destino)]

    data_no_imputed = df_no_imputed_filt.total_trips
    data_imputed = df_imputed_filt.total_trips

    plt.hist(data_imputed, range = (valor_min, valor_max), bins=nbins, alpha=1, label='Imputados', color='red')
    plt.hist(data_no_imputed,  range = (valor_min, valor_max), bins=nbins, alpha=1, label='No imputados', color='blue')
    plt.title(f'Comparación distribución de viajes entre {provincia_origen}-{provincia_destino}')
    plt.xlabel('Total de viajes')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    return plt.show()



def comparacion_global(df_imputed: pd.DataFrame,
                       df_no_imputed:pd.DataFrame,
                       valor_min: int,
                       valor_max:int,
                       nbins:int):
    '''
    Function for plotting a histogram of total_trips 
    before and after imputation with the KNNImputer.
    
    Args:
    ------------------------------------------------
    df_imputed: data after imputation.
    df_no_imputed: data before imputation.
    valor_min: minimum value of the histogram bins values.
    valor_max:int maximum value of the histogram bins values.
    nbins: number of bins of the histogram.
    '''
    data_no_imputed = df_no_imputed.total_trips
    data_imputed = df_imputed.total_trips

    plt.hist(data_imputed, range = (valor_min, valor_max), bins=nbins, alpha=1, label='Imputados', color='red')
    plt.hist(data_no_imputed,  range = (valor_min, valor_max), bins=nbins, alpha=1, label='No imputados', color='blue')
    plt.title(f'Comparación distribución de viajes')
    plt.xlabel('Total de viajes')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    return plt.show()


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


def hampel_filter_parallel(df: pd.DataFrame, 
                               columns_for_filtering: list[str], 
                               window_size: int = 15, 
                               n_sigmas: int = 6) -> pd.DataFrame:
    '''
    Applies the Hampel filter to detect and remove outliers for each
    Province Origin (P_O) - Province Destination (P_D) group in parallel.

    Outliers detected will be replaced with NaN values.

    Args:
    ------------------------------------------------------------
    - df: DataFrame containing the data.
    - columns_for_filtering: List of columns to check for outliers.
    - window_size: Size of the rolling window for the Hampel filter.
    - n_sigmas: Number of standard deviations to consider as outliers.
    '''
    
    filtered_data = []

    def hampel_filter_group(P_O, P_D, group):
        try:
            group_filtered = group.copy()
            
            for column in columns_for_filtering:
                # Cálculo de la mediana y MAD (Median Absolute Deviation)
                rolling_median = group[column].rolling(window=window_size, center=True).median()
                rolling_mad = (group[column] - rolling_median).abs().rolling(window=window_size, center=True).median()
                # Umbral de detección de outliers basado en MAD
                threshold = n_sigmas * (1/0.6745) * rolling_mad
                # Detectar outliers **positivos** (valores que exceden el umbral)
                outliers = (group[column] - rolling_median).abs() > threshold
                # Reemplazar outliers con NaN
                group_filtered.loc[outliers, column] = np.nan

            return group_filtered
        
        except Exception as e:
            print(f"Error processing group ({P_O}, {P_D}): {e}")
            return group  # Devolver el grupo sin cambios en caso de error

    # Procesamiento en paralelo con ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        futures = []
        # Iterar sobre cada grupo de P_O y P_D
        for (P_O, P_D), group in df.groupby(['P_O', 'P_D']):
            future = executor.submit(hampel_filter_group, P_O, P_D, group)
            futures.append(future)

        # Recoger los resultados procesados
        for future in as_completed(futures):
            filtered_data.append(future.result())

    # Concatenar todos los grupos filtrados en un solo DataFrame
    df_filtered = pd.concat(filtered_data).sort_index().reset_index(drop=True)

    return df_filtered