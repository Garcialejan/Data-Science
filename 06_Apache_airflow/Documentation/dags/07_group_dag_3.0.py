from airflow import DAG
from airflow.operators.bash import BashOperator
from groups.groups_downloads import download_tasks
from groups.groups_transforms import transform_tasks

import datetime as dt

with DAG(dag_id = 'group_dag_task_group',
        start_date=dt.datetime(2022, 1, 1),
        schedule='@daily',
        catchup=False) as dag:

    args = {"start_date": dag.start_date,
            "schedule": dag.schedule,
            "catchup": dag.catchup}
    
    downloads = download_tasks()

    check_files = BashOperator(
        task_id='check_files',
        bash_command='sleep 10'
    )

    transforms = transform_tasks()

    downloads >> check_files >> transforms



# from airflow.decorators import dag, task, task_group
# from datetime import datetime
# import time

# @dag(dag_id = 'group_dag_with_taskgroups',
#      start_date=datetime(2022, 1, 1),
#      schedule='@daily',
#      catchup=False
# )
# def group_dag_with_task_groups():
#     @task
#     def download(task_name: str):
#         print(f"Starting {task_name}")
#         time.sleep(10)  # Simula un proceso que tarda 10 segundos
#         print(f"Finished {task_name}")
#         return f"{task_name}_data"

#     @task
#     def check_files(downloaded_data):
#         print("Checking files...")
#         time.sleep(10)  # Simula un proceso que tarda 10 segundos
#         print("Files checked successfully.")
#         return "checked_data"

#     @task
#     def transform(task_name: str, data_to_transform: str):
#         print(f"Transforming {task_name} with data: {data_to_transform}")
#         time.sleep(10)  # Simula un proceso que tarda 10 segundos
#         print(f"Finished transforming {task_name}")
#         return f"transformed_{task_name}_data"

#     # Definir un TaskGroup para las tareas de descarga
#     @task_group(group_id="downloads_group")
#     def downloads_group():
#         download_a = download("download_a")
#         download_b = download("download_b")
#         download_c = download("download_c")
#         return [download_a, download_b, download_c]

#     # Definir un TaskGroup para las tareas de transformación
#     @task_group(group_id="transforms_group")
#     def transforms_group(checked_data):
#         transform_a = transform("transform_a", checked_data)
#         transform_b = transform("transform_b", checked_data)
#         transform_c = transform("transform_c", checked_data)
#         return [transform_a, transform_b, transform_c]

#     # Crear las tareas agrupadas
#     downloads = downloads_group()
#     checked_data = check_files(downloads)
#     transforms = transforms_group(checked_data)

#     # Definir las dependencias entre las tareas
#     downloads >> checked_data >> transforms

# group_dag_with_task_groups()