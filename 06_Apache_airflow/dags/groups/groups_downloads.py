from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.task_group import TaskGroup

import datetime as dt

def download_tasks(parent_dag_id, child_dag_id, args):
    with TaskGroup(
        group_id = "downloads",
        tooltip = "Download tasks") as group:

        download_a = BashOperator(
        task_id='download_a',
        bash_command='sleep 10')

        download_b = BashOperator(
        task_id='download_b',
        bash_command='sleep 10')

        download_b = BashOperator(
        task_id='download_b',
        bash_command='sleep 10')

        return group
