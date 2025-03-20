from airflow.decorators import dag, task
import datetime as dt
import time

@dag(dag_id = 'group_dag',
     start_date=dt.datetime(2022, 1, 1),
     schedule='@daily',
     catchup=False
)
def group_dag():

    @task
    def download(task_name: str):
        print(f"Starting {task_name}")
        time.sleep(10)  # Simula un proceso que tarda 10 segundos
        print(f"Finished {task_name}")
        return f"{task_name}_data"
    
    @task
    def check_files(downloaded_data):
        print("Checking files...")
        time.sleep(10)  # Simula un proceso que tarda 10 segundos
        print("Files checked successfully.")
        return "checked_data"
    
    @task
    def transform(task_name: str, data_to_transform: str):
        print(f"Transforming {task_name} with data: {data_to_transform}")
        time.sleep(10)  # Simula un proceso que tarda 10 segundos
        print(f"Finished transforming {task_name}")
        return f"transformed_{task_name}_data"
    
    # Crear las tareas de descarga
    download_a = download("download_a")
    download_b = download("download_b")
    download_c = download("download_c")

    # Unir las tareas de descarga y pasarlas a check_files
    checked_data = check_files([download_a, download_b, download_c])

    # Crear las tareas de transformación
    transform_a = transform("transform_a", checked_data)
    transform_b = transform("transform_b", checked_data)
    transform_c = transform("transform_c", checked_data)

    # Definir las dependencias entre las tareas
    [download_a, download_b, download_c] >> checked_data >> [transform_a, transform_b, transform_c]

group_dag()

#! Using the TaskFLowAPI for tasks building
# from airflow import DAG
# from airflow.operators.bash import BashOperator
# from datetime import datetime

# with DAG('group_dag', start_date=datetime(2022, 1, 1), 
#     schedule='@daily', catchup=False) as dag:

#     download_a = BashOperator(
#         task_id='download_a',
#         bash_command='sleep 10'
#     )

#     download_b = BashOperator(
#         task_id='download_b',
#         bash_command='sleep 10'
#     )

#     download_c = BashOperator(
#         task_id='download_c',
#         bash_command='sleep 10'
#     )

#     check_files = BashOperator(
#         task_id='check_files',
#         bash_command='sleep 10'
#     )

#     transform_a = BashOperator(
#         task_id='transform_a',
#         bash_command='sleep 10'
#     )

#     transform_b = BashOperator(
#         task_id='transform_b',
#         bash_command='sleep 10'
#     )

#     transform_c = BashOperator(
#         task_id='transform_c',
#         bash_command='sleep 10'
#     )

#     [download_a, download_b, download_c] >> check_files >> [transform_a, transform_b, transform_c]