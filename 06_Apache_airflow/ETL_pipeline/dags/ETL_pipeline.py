from airflow import DAG

from airflow.decorators import task, task_group
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.providers.http.sensors.http import HttpSensor
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

import json


#* Los hook son una herramienta que permite al desarrollador abstraerse de
#* la complejidad de integrar los distintos servicios de Airflow entre sí

#* Recordar que un sensor nos permite definir condiciones para que se
#* ejecuten las sigiuentes task. Usamos HttpSensor para comprobar si la
#* conexión a la API se realiza correctamente

with DAG(
    dag_id = "nasa_postgres_dag",
    start_date=days_ago(1),
    schedule= "@daily",
    catchup= False
) as dag:
    
    ## step 1: Create the table if it doesn't exist
    @task()
    def create_table():
        try:
            postgres_hook = PostgresHook(postgres_conn_id = "my_postgres_connection")
            create_table_sql_query = '''
            CREATE TABLE IF NOT EXISTS nasa_data(
                id SERIAL PRIMARY KEY, 
                title VARCHAR(255),
                explanation TEXT,
                url TEXT,
                date DATE,
                media_type VARCHAR(255)
            );
            '''
            # Execution of table creation with the PostgresHook
            postgres_hook.run(create_table_sql_query)
        except Exception as e:
            raise ValueError(f"Error al crear la tabla. {e}")
        
    # https://api.nasa.gov/planetary/apod?api_key=jeaUvKFRHHK8GISeGAQcVKfdCo1d0fcDSCHsrIl1
    ## step 2: Check the API and extract the data
    check_api_availability = HttpSensor(
                    task_id = "check_api_availability",
                    http_conn_id = "nasa_api",
                    endpoint = "planetary/apod",
                    request_params= {'api_key': '{{conn.nasa_api.extra_dejson.api_key}}'},
                    response_check=lambda response: response.status_code == 200,
                    timeout = 60
                    )
    
    extract_data = SimpleHttpOperator(
                task_id = "extract_data",
                http_conn_id = "nasa_api", # Connection ID defined in Airflow for NASA API
                endpoint = "planetary/apod", # NASA API endpoint for the data
                method = "GET",
                data = {"api_key":"{{conn.nasa_api.extra_dejson.api_key}}"},
                response_filter = lambda response: response.json(), # Extract dara and transform to JSON format
                log_response = True,
                )
    
    ## step 3: transform the data
    @task()
    def transform_nasa_data(response):
        if not all(key in response for key in ["title", "explanation", "url", "date", "media_type"]):
            raise ValueError("La respuesta de la API no contiene todos los campos requeridos.")
        nasa_data = {
            "title":response.get("title", ""), # If not "title" gives "". With the previous step this is not necesary
            "explanation":response.get("explanation", ""),
            "url":response.get("url", ""),
            "date":response.get("date", ""),
            "media_type":response.get("media_type", "")
        }
        return nasa_data
    
    ## step 4: upload the data into the PostgreSQL
    @task()
    def load_data_into_postgres(nasa_data):
        postgres_hook = PostgresHook(postgres_conn_id = "my_postgres_connection")
        insert_sql_query = '''
        INSERT INTO nasa_data (title, explanation, url, date, media_type)
        VALUES (%s, %s, %s, %s, %s);
        '''
        #Execute the SQL query
        postgres_hook.run(insert_sql_query, parameters=(
            nasa_data["title"],
            nasa_data["explanation"],
            nasa_data["url"],
            nasa_data["date"],
            nasa_data["media_type"]
        ))
        
    ## step 5: verify the data with DBViewer
    
    ## step 6: Define the task dependencies
    # Extract
    create_table() >> check_api_availability >> extract_data
    api_response = extract_data.output
    # Transform
    transformed_data = transform_nasa_data(api_response)
    # Load
    load_data_into_postgres(transformed_data)