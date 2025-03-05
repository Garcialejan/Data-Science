import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from matplotlib.patches import Ellipse


def plot_correlation_heatmap(df, titulo, threshold = 0.2):
    """
    Función para plotear un heatmap de correlación a partir de un DataFrame.
    
    Args:
        df (pd.DataFrame): El DataFrame que contiene los datos a analizar.
        titulo (str): Título del gráfico.
    """
    # Verificar si el argumento 'df' es un DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("El argumento 'df' debe ser un pd.DataFrame.")

    # Calcular la matriz de correlación
    corr = df.corr()

    # Aplicar el umbral: convertir en NaN los valores de correlación menores a threshold (en valor absoluto)
    corr = corr.mask(corr < threshold)

    # Crear una máscara para ocultar la parte superior del heatmap
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Configurar el tamaño de la figura
    fig = plt.figure(figsize=(16, 9))

    # Definir la paleta de colores
    cmap = sns.diverging_palette(210, 5, as_cmap=True)

    # Generar el heatmap
    sns.heatmap(corr, mask=mask, cmap=cmap,
                vmax=1, vmin=-1, center=0,
                annot=True, annot_kws={"size": 8},
                square=True, linewidths=.5,
                cbar_kws={"shrink": .5})

    # Añadir el título
    plt.title(titulo, fontsize=16)

    # Mostrar la gráfica
    return plt.show()



def plot_feature_importance(rf_model, X, title='Importancia de las Variables'):
    """
    Función para obtener y graficar la importancia de las variables de un modelo de clasificación.

    Args:
        rf_model: Modelo de Random Forest ya entrenado.
        X: DataFrame que contiene las variables predictoras (feature variables).
        title: Título del gráfico.
    """
    # Obtener los coeficientes (importancia de las variables)
    coeficientes = rf_model.feature_importances_

    # Relacionar los coeficientes con los nombres de las columnas
    pesos_variables = pd.DataFrame({
        'Variable': X.columns,
        'Coeficiente': coeficientes
    })

    # Ordenar los coeficientes por magnitud de mayor a menor
    pesos_variables = pesos_variables.sort_values(by='Coeficiente', ascending=False)

    # Crear el gráfico de barras
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Coeficiente', y='Variable', data=pesos_variables, palette='viridis')

    # Personalizar el gráfico
    plt.title(title, fontsize=16)
    plt.xlabel('Peso (Coeficiente)', fontsize=14)
    plt.ylabel('Variable', fontsize=14)

    # Mostrar el gráfico
    return plt.show()



def plot_confusion_matrix(y_true, y_pred, title='Matriz de Confusión'):
    """
    Función para calcular y graficar la matriz de confusión.

    Args:
        y_true: Las etiquetas verdaderas.
        y_pred: Las etiquetas predichas por el modelo.
        title: Título del gráfico.
    """
    # Calcular la matriz de confusión
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Crear el gráfico de la matriz de confusión
    plt.figure(figsize=(9, 9))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                cbar=True, annot_kws={"size": 14})
    
    # Personalizar el gráfico
    plt.ylabel('Etiquetas Verdaderas', fontsize=14)
    plt.xlabel('Predicciones', fontsize=14)
    plt.title(title, fontsize=16)
    
    # Mostrar el gráfico
    return plt.show()



def plot_cluster_scatter(coord, labels, tittle):
    # Crear el scatter plot
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(coord.iloc[:, 0], coord.iloc[:, 1], 
                        c=labels, cmap='viridis', alpha=0.7, edgecolor='k', s=100)

    # Mejorar el formato
    plt.title(tittle, fontsize=16, fontweight='bold')
    plt.xlabel('Eje X', fontsize=14)
    plt.ylabel('Eje Y', fontsize=14)
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    # Personalizar el fondo
    plt.gca().set_facecolor('#f9f9f9')  # Cambiar el color de fondo
    plt.gca().spines['top'].set_visible(False)  # Ocultar el borde superior
    plt.gca().spines['right'].set_visible(False)  # Ocultar el borde derecho

    # Ajustar los ticks
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Añadir anotaciones opcionales
    for i, txt in enumerate(labels):
        plt.annotate(txt, (coord.iloc[i, 0], coord.iloc[i, 1]), 
                    fontsize=9, ha='right', va='bottom')

    # Añadir elipses para cada clúster
    for cluster in np.unique(labels):
        # Obtener los puntos del clúster actual
        cluster_points = coord[labels == cluster]
        
        # Calcular la media y la covarianza
        mean = cluster_points.mean(axis=0).values
        cov = np.cov(cluster_points.T)  # Covarianza de los datos
        
        # Calcular los valores propios y vectores propios
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # Calcular el ancho y alto de la elipse
        width, height = 4 * np.sqrt(eigenvalues)  # Aumentar el tamaño de la elipse
        angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]) * 180 / np.pi  # Convertir a grados
        
        # Obtener el color del clúster
        cluster_color = plt.cm.viridis(cluster / (len(np.unique(labels)) - 1))
        
        # Crear la elipse
        ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, edgecolor=cluster_color, fill=False, linewidth=2)
        plt.gca().add_patch(ellipse)

    plt.show()