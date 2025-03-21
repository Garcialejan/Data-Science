from airflow import DAG
from airflow.operators.bash import BashOperator
    
from datetime import datetime

# Recordar que para correr tareas en paralelo debemos modificar el #! AIRFLOW__CORE__EXECUTOR 
# Levantar flower con: docker compose --profile flower up -d
# para comprobar el estado de los tasks

with DAG('parallel_dag',
         start_date = datetime(2022, 1, 1), 
         schedule ='@daily',
         catchup=False) as dag:

        extract_a = BashOperator(
            task_id='extract_a',
            bash_command='sleep 10'
        )

        extract_b = BashOperator(
            task_id='extract_b',
            bash_command='sleep 10'
        )

        load_a = BashOperator(
            task_id='load_a',
            bash_command='sleep 10'
        )

        load_b = BashOperator(
            task_id='load_b',
            bash_command='sleep 10'
        )

        transform = BashOperator(
            task_id='transform',
            queue="high_cpu", #! Para enviar al task a la cola específica del worker
            bash_command='sleep 30'
        )

        extract_a >> load_a
        extract_b >> load_b
        [load_a, load_b] >> transform