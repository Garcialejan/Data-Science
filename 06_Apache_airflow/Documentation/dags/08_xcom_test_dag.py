from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

from datetime import datetime

def _t1(ti): # ti significa: task instance (object)
    ti.xcom_push(key = 'my_key', value = 42) # Usando xcom con ti y xcom_push

def _t2(ti):
    print(ti.xcom_pull(key = "my_key", task_id = 't1')) # Especificas la key de la que quieres realizar el pull y el task_id de donde proviene el xcom

with DAG("xcom_dag_test",
        start_date=datetime(2022, 1, 1),
        schedule='@daily',
        catchup=False) as dag:
    t1 = PythonOperator(
        task_id='t1',
        python_callable=_t1)

    t2 = PythonOperator(
        task_id='t2',
        python_callable=_t2)

    t3 = BashOperator(
        task_id='t3',
        bash_command="echo ''")

    t1 >> t2 >> t3