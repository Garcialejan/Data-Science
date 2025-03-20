from airflow import DAG
from airflow.operators.python import PythonOperator # Used to run a python function
import datetime as dt

## Define task 1
def preprocess_data():
    print("Preprocessing data...")

## Define task 2
def train_model():
    print("Training model...")

## Define task 3
def evaluate_model():
    print("Evaluate Models...")

## Define the DAG
with DAG(
    dag_id = 'ml_pipeline',
    start_date=dt.datetime(2025,1,1),
    schedule='@weekly',
    catchup= False
) as dag:

    ##Define the task
    preprocess=PythonOperator(task_id="preprocess_task", python_callable=preprocess_data)
    train=PythonOperator(task_id="train_task", python_callable=train_model, trigger_rule='all_success')
    evaluate=PythonOperator(task_id="evaluate_task", python_callable=evaluate_model, trigger_rule='all_success')

    #* To understand well the trigger rules for the tasks https://www.astronomer.io/blog/understanding-airflow-trigger-rules-comprehensive-visual-guide/

    ## set dependencies
    preprocess >> train >> evaluate