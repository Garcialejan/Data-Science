import datetime as dt
from airflow import DAG, Dataset
from airflow.decorators import task

my_file = Dataset("/tmp/my_file.txt") # Definimos un dataset de Airflow
my_file_2 = Dataset("/tmp/my_file2.txt") 

with DAG(
    dag_id = "producer", 
    start_date = dt.datetime(2025, 3, 17),
    schedule = "@daily",
    catchup = False) as dag:

    # El parámetro outlets se usa para definir los datasets o recursos que una tarea produce como salida.
    # Puede ser archivos, tablas en bases de datos, directorios, URLs, etc
    # Al declarar un dataset como "outlet", estás indicando a Airflow que esta tarea modifica o actualiza ese recurso
    @task(outlets = [my_file]) 
    def update_dataset():
        with open(my_file.uri, "a+") as f:
            f.write("Producer update")

    @task(outlets = [my_file_2]) 
    def update_dataset_2():
        with open(my_file_2.uri, "a+") as f:
            f.write("Producer update")

    update_dataset() >> update_dataset_2()