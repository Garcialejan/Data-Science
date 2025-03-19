from airflow import DAG
from airflow.decorators import task
from airflow.operators.bash import BashOperator

import datetime as dt

def subdag_downloads(parent_dag_id, child_dag_id, args):
    with DAG(dag_id = f"{parent_dag_id}.{child_dag_id}",
             start_date = args["start_date"],
             schedule = args["schedule"],
             catchup = args["catchup"]) as dag:

        download_a = BashOperator(
        task_id='download_a',
        bash_command='sleep 10'
        )

        download_b = BashOperator(
        task_id='download_b',
        bash_command='sleep 10'
        )

        download_b = BashOperator(
        task_id='download_b',
        bash_command='sleep 10'
        )

        return dag


