from airflow import DAG
from airflow.operators.python import PythonOperator
from hooks.elastics.elastic.hook import ElasticHook
import datetime as dt

def _print_es_info():
    hook = ElasticHook()
    print(hook.info())

with DAG('elastic_dag',
        start_date=dt.datetime(2022, 1, 1),
        schedule='@daily',
        catchup=False) as dag:
    
    print_es_info = PythonOperator(
        task_id='print_es_info',
        python_callable=_print_es_info
    )