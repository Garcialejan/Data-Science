#Crear nuevo Cluster Kafka con un ID random
kafka-storage.sh random-uuid

#Configurar directorio Logs
kafka-storage.sh format -t KONanyvXTTCIv3WJKOKYpQ -c ~/kafka_2.13-3.7.1/config/kraft/server.properties

#Inicializar Kafka en daemon mode
kafka-server-start.sh ~/kafka_2.13-3.7.1/config/kraft/server.properties

#Mover archivos de Ubuntu a entorno windows con la siguiente ruta. Es la carpeta de Ubuntu dentro de nuestro windows, sonde se encuentran nuestro archivos de Ubunto.
\\wsl.localhost\Ubuntu\home\alejandro_garcia