from confluent_kafka.admin import AdminClient, NewTopic

admin_client = AdminClient({'bootstrap.servers': 'localhost:9092'})

#Crear Topic con X particiones
nombre_topic = 'testtopic'
particiones = 3
nuevotopic = NewTopic(topic=nombre_topic, num_partitions=particiones)
result = admin_client.create_topics(new_topics=[nuevotopic])

# Imprimir el resultado
for topic, future in result.items():
    try:
        future.result()  # Esto lanzará una excepción si hay un error
        print(f"Se ha creado el topic {topic} con {particiones} particiones.")
    except Exception as e:
        print(f"No se pudo crear el topic {topic}. Error: {e}")