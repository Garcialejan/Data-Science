

###########
#TOPICS
###########

#Crear topic
kafka-topics.sh --bootstrap-server localhost:9092 --topic prueba --create

#Obtener el detalle de un topic
kafka-topics.sh --bootstrap-server localhost:9092 --topic pruebaparticiones --describe

#Lista de topics creados
kafka-topics.sh --bootstrap-server localhost:9092 --list 

#Eliminar un topic 
kafka-topics.sh --bootstrap-server localhost:9092 --topic prueba --delete

#Crear un topic indicando el número de particiones
kafka-topics.sh --bootstrap-server localhost:9092 --topic pruebaparticiones --create --partitions 2

#Crear un topic indicando el número de particiones y réplicas
kafka-topics.sh --bootstrap-server localhost:9092 --topic pruebareplicas --create --partitions 3 --replication-factor 2




###########
#PRODUCERS
###########

#Para ver todos los comandos relacionados con los producers
kafka-console-producer.sh

#Producir para un topic (si el topic no existe se creará automáticamente)
kafka-console-producer.sh --bootstrap-server localhost:9092 --topic pruebaparticiones 
> Mensaje de Prueba

#Producir para un topic con keys
kafka-console-producer.sh --bootstrap-server localhost:9092 --topic pruebaparticiones --property parse.key=true --property key.separator=:
>clave:valor
>deporte:baloncesto
>pais:colombia

#Para recibir información de que el consumer ha consumido el mensaje correctamente\ha podido leerlo sin problemas. Confirmar que no ha habido perdida de información/mensajes.
kafka-console-producer.sh --bootstrap-server localhost:9092 --topic pruebaparticiones --producer-property acks=all



###########
#CONSUMERS
###########

#Consumir los mensajes que se reciben en un topic
kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic pruebaparticiones

#Consumir todos los mensajes de un topic des del inicio
kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic pruebaparticiones --from-beginning

#Mostrar propiedades de los mensajes recibidos
kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic pruebaparticiones --formatter kafka.tools.DefaultMessageFormatter --property print.timestamp=true --property print.key=true --property print.value=true --property print.partition=true --from-beginning




###########
#CONSUMER GROUPS
###########

#Productor balancea aleatoriamente mensajes entre particiones de un mismo topic. No es útil en producción porque no sabemos a qué partición enviamos nuestros mensajes. Mejor trabajar con clave:valor
kafka-console-producer.sh --bootstrap-server localhost:9092 --producer-property partitioner.class=org.apache.kafka.clients.producer.RoundRobinPartitioner --topic topicgrupo

#Iniciar un consumidor asociado a un grupo
kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic topicgrupo --group testgrupoconsumers

#Listar los consumidores que tenemos
kafka-consumer-groups.sh --bootstrap-server localhost:9092 --list

#Mostrar mas detalles de un grupo de consumidores
kafka-consumer-groups.sh --bootstrap-server localhost:9092 --describe --group pruebagrupoconsumers --from -beginning

#Mover el offset a la posición zero (no se formaliza hasta que se lanza el "--execute")
kafka-consumer-groups.sh --bootstrap-server localhost:9092 --group pruebagrupoconsumers --reset-offsets --to-earliest --topic topicgrupo --dry-run

#Mover el offset a la posición zero
kafka-consumer-groups.sh --bootstrap-server localhost:9092 --group pruebagrupoconsumers --reset-offsets --to-earliest --topic topicgrupo --excute