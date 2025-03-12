### MLFLOW On AWS

## MLflow on AWS Setup:

1. Login to AWS console.
2. Create IAM user with AdministratorAccess
3. Export the credentials in your AWS CLI by running "aws configure"
4. Create a s3 bucket
5. Create EC2 machine (Ubuntu) & add Security groups 5000 port

Run the following command on EC2 machine
```bash
sudo apt update

sudo apt install python3-pip

sudo apt install pipenv

sudo apt install virtualenv

mkdir mlflow

cd mlflow

pipenv install mlflow

pipenv install awscli

pipenv install boto3

pipenv shell


## Then set aws credentials
aws configure


#Finally 
mlflow server -h 0.0.0.0 --default-artifact-root s3://mlflowtracking1

#open Public IPv4 DNS to the port 5000


#set uri in your local terminal and in your code 
export MLFLOW_TRACKING_URI=http://ec2-54-158-152-207.compute-1.amazonaws.com:5000/
```


### Explicación comandos
1. `sudo apt install python3-pip`
- Acción : Instala el gestor de paquetes pip para Python 3 en un sistema basado en Debian (como Ubuntu).
- Propósito : Permite instalar bibliotecas y herramientas de Python desde PyPI (Python Package Index) o repositorios similares.
- Ejemplo : Después de ejecutar este comando, podrás usar pip para instalar módulos como Flask, NumPy, etc.

2. `sudo apt install pipenv`
- Acción : Instala pipenv, una herramienta que combina pip y virtualenv para gestionar dependencias y entornos virtuales de Python.
- Propósito : Simplifica la gestión de proyectos Python al permitirte crear entornos aislados con todas las dependencias necesarias definidas en un archivo Pipfile.
- Ventaja : Automatiza la creación de entornos virtuales y gestiona versiones de paquetes.

3. `sudo apt install virtualenv`
- Acción : Instala virtualenv, una herramienta para crear entornos virtuales de Python.
- Propósito : Aísla las dependencias de tu proyecto de las instaladas globalmente en tu sistema.
- Diferencia con Pipenv : Mientras que virtualenv solo crea el entorno, pipenv también gestiona las dependencias y el archivo Pipfile.

4. `pipenv install awscli`
- Acción : Instala la CLI de AWS (awscli) en el entorno virtual gestionado por pipenv.
- Propósito : Permite interactuar con servicios de Amazon Web Services (AWS) desde la línea de comandos.
- Uso común : Subir datos a S3, gestionar recursos EC2, etc.

5. `pipenv install boto3`
- Acción : Instala boto3, la biblioteca oficial de AWS para Python.
- Propósito : Proporciona una interfaz programática para interactuar con servicios de AWS, como S3, DynamoDB, Lambda, etc.

6. `pipenv shell`
- Acción : Abre un shell dentro del entorno virtual configurado por pipenv. Se utiliza para activar el entorno creado dentro de la shell.
- Propósito : Te permite ejecutar comandos de Python o scripts en un entorno aislado con las dependencias específicas del proyecto cargadas.
- Ejemplo : Si ejecutas `pipenv shell python`, estarás usando la versión de Python y las bibliotecas instaladas en el entorno virtual.

7. `mlflow server -h 0.0.0.0 --default-artifact-root s3://mlflow-project-tracking1`
- **mlflow server**: para iniiar el servidor de MLflow
- **-h 0.0.0.0**: Especifica la dirección IP en la que el servidor escuchará conexiones. En este caso el servidor escuchará en todas las interfaces de red disponibles.
- **--default-artifact-root**:  Define la ubicación predeterminada que MLflow utiliza para saber dónde se almacenarán los artefactos generados por los experimentos. Se requiere tener configuradas las credenciales de AWS correctamente (Claves de acceso del usuario). Puede ser tanto un directorio local, por ejemplo `file://path/to/local/directory` como un bucket de S3, como `s3://mi-bucket-de-s3/`