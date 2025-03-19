from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.subdag import SubDagOperator


import datetime as dt

def subdag_transform(parent_dag_id, child_dag_id, args):
    with DAG(dag_id = f"{parent_dag_id}.{child_dag_id}",
            start_date = args["start_date"],
            schedule = args["schedule"],
            catchup = args["catchup"]) as dag:

        transform_a = BashOperator(
            task_id='transform_a',
            bash_command='sleep 10')

        transform_b = BashOperator(
            task_id='transform_b',
            bash_command='sleep 10')

        transform_c = BashOperator(
            task_id='transform_c',
            bash_command='sleep 10')
        
        return dag