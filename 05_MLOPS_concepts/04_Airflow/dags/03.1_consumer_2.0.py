import datetime as dt
from airflow import DAG, Dataset
from airflow.decorators import task

my_file = Dataset("/tmp/my_file.txt") 
my_file_2 = Dataset("/tmp/my_file2.txt") 

with DAG(
    dag_id = "consumer2",
    schedule = [my_file, my_file_2], #! Podemos hacer que airflow espere a la actualziación de dos datasets
    start_date = dt.datetime(2025, 3, 17),
    catchup = False) as dag:

    @task() 
    def read_dataset():
        with open(my_file.uri, "r") as f:
            print(f.read())

    read_dataset()