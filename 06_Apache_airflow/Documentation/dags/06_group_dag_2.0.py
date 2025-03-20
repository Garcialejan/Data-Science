from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.subdag import SubDagOperator
from subdags.subdags_downloads import subdag_downloads
from subdags.subdags_transform import subdag_transform

import datetime as dt

with DAG(dag_id = 'group_dag_2',
        start_date=dt.datetime(2022, 1, 1),
        schedule='@daily',
        catchup=False) as dag:

    args = {"start_date": dag.start_date,
            "schedule": dag.schedule,
            "catchup": dag.catchup}
    
    downloads = SubDagOperator(
        task_id = "downloads",
        subdag = subdag_downloads(dag.dag_id, "downloads", args) #Función que se ha creado en el fichero 06 y que recibe el parent.id y el child.id (el id del subdag)
    )

    check_files = BashOperator(
        task_id='check_files',
        bash_command='sleep 10'
    )

    transforms = SubDagOperator(
        task_id = "transforms",
        subdag = subdag_transform(dag.dag_id, "transforms", args) #Función que se ha creado en el fichero 06 y que recibe el parent.id y el child.id (el id del subdag)
    )

    downloads >> check_files >> transforms