from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator

import datetime as dt


def _t1(ti): # ti significa: task instance (object)
    ti.xcom_push(key = 'my_key', value = 42) #Usnado xcom con ti y xcom_push

def _t2(ti):
    ti.xcom_pull(key = "my_key", task_id = 't1') # Especificas la key de la que quieres realizar el pull y el task_id de donde proviene el xcom

def branch_func(ti):
    value = ti.xcom_pull(key = "my_key", task_id = 't1')
    if (value == 42):
        return "t2"
    return "t3"

with DAG("condition_dag",
        start_date=dt.datetime(2025, 3, 18),
        schedule='@daily',
        catchup=False) as dag:
    
    t1 = PythonOperator(
        task_id='t1',
        python_callable=_t1)
    
    branch = BranchPythonOperator(
        task_id = "branch",
        python_callable = branch_func
    )

    t2 = PythonOperator(
        task_id='t2',
        python_callable=_t2)

    t3 = BashOperator(
        task_id='t3',
        bash_command="echo ''")
    
    t4 = BashOperator( # Esta tarea no se va a ejecutar, ya que para ello, deberías cumplirse t2 y t3 porque no se ha usado una condición. El tigger es all_succes por defecto
        task_id='t4',
        bash_command="echo ''")

    t1 >> branch >> [t2, t3] >> t4