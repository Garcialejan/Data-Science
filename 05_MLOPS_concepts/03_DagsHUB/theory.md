# DagsHUB: carga de datos y logs de experimentos
Las principales funcionalidades que implemente DagsHUBs son:
- Control de **versiones de nuestros dato**s en un repositorio remoto utilizando **DVC**. Es como GitHUB pero para nuestro versionado de los datos.
- **Control** de nuestros **experimentos** con **MLflow**. Nos permite tener un repositorio remoto en el que podemos compartir y trackear los experimentos que realizamos para nuestros modelo de ML.
- También podemos utilizarlo como si fuera GitHUB para trackear las versiones de nuestro código. Lo recomendable es conectar tu repositorio de GitHUb con el de DagsHUB para que las versiones del código estén unificadas en ambos repositorios sin problemas de replicación.