#Para instalar Linux en Windows con WSL es necesario abrir PowerShell y ejecutar lo siguiente:
wsl --install

#Desactivar ipv6
sudo sysctl -w net.ipv6.conf.all.disable_ipv6=1
sudo sysctl -w net.ipv6.conf.default.disable_ipv6=1

#Instalar Java
wget -O- https://apt.corretto.aws/corretto.key | sudo apt-key add - 
sudo add-apt-repository 'deb https://apt.corretto.aws stable main'
sudo apt-get update; sudo apt-get install -y java-11-amazon-corretto-jdk

#Comprobar versión de Java
java --version

#Instalar Kafka
wget https://downloads.apache.org/kafka/3.6.1/kafka_2.13-3.6.1.tgz

#Extraer el contenido y moverlo al directorio principal
tar -xvzf kafka_2.13-3.6.1.tgz
mv kafka_2.13-3.6.1 ~

#Configuración PATH
vim .bashrc
PATH="$PATH:~/kafka_2.13-3.6.1/bin"