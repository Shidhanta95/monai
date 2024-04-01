import boto3
import os


def extract_data():
    s3 = boto3.client('s3')
    bucket_name = 'deeplearning-mlops'
    url = s3.generate_presigned_url(
                    ClientMethod='get_object',
                    Params={'Bucket': bucket_name, 'Key': 'workspace.zip'},
                    ExpiresIn=7500  # URL expiration time in seconds (adjust as needed)
                )
    print(url)
    return url

extract_data()
