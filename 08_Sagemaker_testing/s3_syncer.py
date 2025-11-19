import os

# Se puede utilizar la CLI de AWS para ejecutar el comando "aws s3 sync" que
# permite conectarnos con los buckets de S3 para cargar o importar archivos.
# Se ejecuta el comando a través de os.system()
class S3Sync:
    def sync_folder_to_s3(self, folder, aws_bucket_url):
        command = f"aws s3 sync {folder} {aws_bucket_url} "
        os.system(command)

    def sync_folder_from_s3(self,folder,aws_bucket_url):
        command = f"aws s3 sync {aws_bucket_url} {folder} "
        os.system(command)