from confluent_kafka import Producer

p = Producer({'bootstrap.servers': 'localhost:9092'})
p.produce('testtopic', 'hola esto es una prueba')
p.flush()