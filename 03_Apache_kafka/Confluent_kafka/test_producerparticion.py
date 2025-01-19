from confluent_kafka import Producer

nombre_topic = 'testtopic'
mensaje = 'mensaje particion 1'
particion_ID = 1

p = Producer({'bootstrap.servers': 'localhost:9092'})
p.produce(topic=nombre_topic, value=mensaje, partition=particion_ID)
p.flush()