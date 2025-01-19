#importar librerías
from confluent_kafka import Producer
import requests
import json
import time


#Cryptos en seguimiento
cryptos_to_track = ['bitcoin', 'ethereum', 'ripple', 'litecoin', 'cardano', 'polkadot', 'stellar', 'eos', 'tron', 'dogecoin']

#URL de la API de coincap
api_url = 'https://api.coincap.io/v2/assets'

# Configuración de Kafka
producer_conf = {'bootstrap.servers': 'localhost:9092'}
producer = Producer(producer_conf)
topic_name = 'cryptodata'

while True:
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        # Verificar la estructura de la respuesta
        if 'data' in data and isinstance(data['data'], list):
            # Crear una lista de diccionarios con la información
            rows = [{'timestamp': int(time.time()), 'name': crypto['name'], 'symbol': crypto['symbol'], 'price': crypto['priceUsd']} for crypto in data['data'] if crypto['id'] in cryptos_to_track]
            
            # Enviar los datos a Kafka
            for row in rows:
                producer.produce(topic_name, value=json.dumps(row))

            producer.flush()

            print("Datos enviados exitosamente a Kafka")
        
        else:
            print("La estructura de la respuesta no es la esperada")
        
    else:
        print(f"Error al realizar la solicitud a la API. Código de estado: {response.status_code}")

    time.sleep(30)