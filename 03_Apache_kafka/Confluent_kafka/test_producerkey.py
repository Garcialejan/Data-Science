from confluent_kafka import Producer

nombre_topic = 'testtopic'
clave = 'pais'
valor = 'colombia'

p = Producer({'bootstrap.servers': 'localhost:9092'})
p.produce(topic=nombre_topic, key=clave, value=valor)
p.flush()