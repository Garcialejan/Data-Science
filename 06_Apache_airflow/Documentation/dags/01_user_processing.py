import datetime as dt
import json
from pandas import json_normalize

from airflow import DAG
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.http.sensors.http import HttpSensor
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.operators.python import PythonOperator

#* Los hook son una herramienta que permite al desarrollador abstraerse de
#* la complejidad de integrar los distintos servicios de Airflow entre sí 
from airflow.providers.postgres.hooks.postgres import PostgresHook

with DAG(dag_id = "user_processing",
         start_date = dt.datetime(2025, 3, 17), # Necesario fijar la fecha de origen como un start date object
         schedule = "@daily", # https://airflow.apache.org/docs/apache-airflow/1.10.10/scheduler.html#dag-runs for CRON preset
         catchup = False) as dag: # Define si debemos ejecutar el dag de días anteriores al actual. True permite re lanzar los DAG para fechas no triggeadas.


        create_table = PostgresOperator(
            task_id = 'create_table',
            postgres_conn_id = "postgres",
            sql =
            '''
            CREATE A TABLE IF NOT EXISTS users(
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                country TEXT NOT NULL,
                user_name TEXT NOT NULL,
                email TEXT NOT NULL,
                password TEXT NOT NULL,
                );
                ''')
        # Para comprobar si un task funciona podemos abrir el bash dentro del contenedor (docker exec -it <nombre_cont> bash)
        # y ejecutar #! airflow tasks <dag_id> [task_name]



        is_api_available = HttpSensor(
                task_id = "is_api_avaiable",
                http_conn_id = "user_api",
                endpoint = "api/" # Es el path de la API/web que queremos revisar si está en funcionamiento para realizar la consulta
                )
        
        extract_user = SimpleHttpOperator(
                task_id = "extract_user",
                http_conn_id = "user_api",
                endpoint = "api/", # Es el path de la API/web que queremos revisar si está en funcionamiento para realizar la consulta
                method = "GET",
                response_filter = lambda response: json.loads(response.txt), # Extraemos los datos y los convertimos a formato JSON
                log_response = True, # Para que los logs se puedan ver desde la UI 
                )
        

        def proces_user_func(ti):
            user = ti.xcom_pull(task_ids = "extract_user") #Xcom es una funcionalidad interna de airflow que permite compartir datos entre tasks.
            user = user["results"][0]
            processed_user = json_normalize({
                "first_name": user["name"]["first"],
                "last_name": user["name"]["last"],
                "country": user["location"]["country"],
                "user_name": user["login"]["username"],
                "password": user["login"]["password"],
                "email": user["email"]["first"]
                })
            processed_user.to_csv("/tmp/processed_user.csv", index = None, header = False)

        process_user = PythonOperator( # Operador que se utiliza para correr código de Python dentro de un DAG
                task_id = "process_user",
                python_callable = proces_user_func # Se define la función de python que se quiere ejecutar
        )

        def store_user_func():
            hook = PostgresHook(postgres_conn_id = "postgres")
            hook.copy_expert(
                sql = "COPY users FROM stdin WITH DELIMITER as ','",
                filename = "/tmp/processed_user.csv"
            )

        store_user = PythonOperator(
             task_id = "store_user",
             python_callable = store_user_func
        )

        create_table >> is_api_available >> extract_user >> process_user >> store_user

        # create_table >> is_api_available >> [extract_user, process_user, store_user] #! Si quisiera ejecutar en paralelo realizando bifurcaciones