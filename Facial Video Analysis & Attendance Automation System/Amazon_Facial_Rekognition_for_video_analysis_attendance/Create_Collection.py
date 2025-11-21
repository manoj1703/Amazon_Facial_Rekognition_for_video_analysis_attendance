import pandas as pd
import boto3
from botocore.exceptions import ClientError

credential = pd.read_csv("facial_video_analysis_accessKeys.csv")
access_key_id = credential['Access key ID'][0]
secret_access_key = credential['Secret access key'][0]
region_name = 'us-east-2'  # Specify your AWS region here

client = boto3.client(
    'rekognition',
    aws_access_key_id=access_key_id,
    aws_secret_access_key=secret_access_key,
    region_name=region_name  # Add this line to specify the region
)

def create(COLLECTION_NAME):
    print('Creating collection: {}'.format(COLLECTION_NAME))
    try:
        response = client.create_collection(CollectionId=COLLECTION_NAME)
        print('Collection ARN: {}'.format(response['CollectionArn']))
        print('Status code: {}'.format(str(response['StatusCode'])))
        st1 = 'Collection: {} has been created.'.format(COLLECTION_NAME)
        return st1
    except client.exceptions.ResourceAlreadyExistsException:
        print('Collection: {} already exists.'.format(COLLECTION_NAME))
        st1 = 'Collection: {} already exists.'.format(COLLECTION_NAME)
        return st1
    except ClientError as e:
        st = "Cannot create / Don't give space for collection Name"
        return st

def delete(COLLECTION_NAME):
    print('Deleting collection: {}'.format(COLLECTION_NAME))
    try:
        response = client.delete_collection(CollectionId=COLLECTION_NAME)
        print('Deleting collection: {}'.format(COLLECTION_NAME))
        print('Status code: {}'.format(str(response['StatusCode'])))
        st1 = 'Collection: {} has been deleted.'.format(COLLECTION_NAME)
        return st1
    except client.exceptions.ResourceNotFoundException:
        print('No such Collection: {}'.format(COLLECTION_NAME))
        st2 = 'No such Collection: {}'.format(COLLECTION_NAME)
        return st2
    except ClientError as e:
        st = 'cannot delete / problem with client'
        return st

def list_collections():
    try:
        print('Displaying collections...')
        response = client.list_collections()
        collections = response['CollectionIds']
        print(len(collections), collections)
        return len(collections), collections
    except ClientError as e:
        return 0, "Problem in client"

# if __name__ == '__main__':
#     COLLECTION_NAME = 'Face_recognition_collection'
#     delete(COLLECTION_NAME)
#     # create(COLLECTION_NAME)
#     l, p = list_collections()
#     print(l)
#     print(p)
