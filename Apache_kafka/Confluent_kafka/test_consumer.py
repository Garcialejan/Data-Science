
from confluent_kafka import Consumer

nombre_topic = 'testtopic'
grupo_consumers = 'testgrupo'

c = Consumer({'bootstrap.servers': 'localhost:9092', 'group.id': grupo_consumers, 'auto.offset.reset': 'earliest'})
c.subscribe([nombre_topic])

try:
    while True:
        msg = c.poll(1.0) #Almacenamos los mensajes que se han recibido en cada segundo.
        if msg is None:
            print("No hay ningún mensaje. Sigo escuchando")
            continue
        print(f"Valor Mensaje: {msg.value().decode('utf-8')}") #El decode se utiliza para decodificar el mensaje original, ya que este se envía en bytes, por lo que debemos decodificarlo.
        print(f"Clave: {msg.key()}")
        print(f"Particion: {msg.partition()}")
        print(f"Offset: {msg.offset()}")

except KeyboardInterrupt:
    pass
finally:
    c.close()