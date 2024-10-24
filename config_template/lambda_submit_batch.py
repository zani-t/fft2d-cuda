import boto3
import os

def lambda_handler(event, context):
    batch_client = boto3.client('batch')
    job_queue = '<job-queue-name>'
    job_definition = 'job-definition-name'
    
    # Extract bucket and key from the S3 event
    try:
        s3_bucket = event['Records'][0]['s3']['bucket']['name']
        s3_key = event['Records'][0]['s3']['object']['key']
    except (IndexError, KeyError) as e:
        print(f"Error parsing S3 event: {e}")
        return
    
    # Submit AWS Batch job with environment variables
    try:
        response = batch_client.submit_job(
            jobName='job-name',
            jobQueue=job_queue,
            jobDefinition=job_definition,
            containerOverrides={
                'environment': [
                    {'name': 'INPUT_BUCKET', 'value': s3_bucket},
                    {'name': 'INPUT_KEY', 'value': s3_key},
                ]
            }
        )
        print(f"Submitted Batch job: {response['jobId']}")
    except Exception as e:
        print(f"Error submitting Batch job: {e}")
