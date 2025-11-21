import pandas as pd
import boto3
from botocore.exceptions import ClientError

# Load AWS credentials
credential = pd.read_csv("facial_video_analysis_accessKeys.csv")
access_key_id = credential['Access key ID'][0]
secret_access_key = credential['Secret access key'][0]
region_name = 'us-east-2'  # Your AWS region

# Initialize Rekognition client
client = boto3.client(
    'rekognition',
    aws_access_key_id=access_key_id,
    aws_secret_access_key=secret_access_key,
    region_name=region_name
)


def add_face_to_collection(source_img_bytes, image_name, COLLECTION_NAME):
    """
    Adds a face to a Rekognition collection.

    Parameters:
        source_img_bytes (bytes): Image bytes of the face to register
        image_name (str): Name/label of the person (ExternalImageId)
        COLLECTION_NAME (str): Rekognition collection name

    Returns:
        list: Messages about registration status
    """
    lst = []
    try:
        print(f'Adding face for {image_name} into {COLLECTION_NAME}...')

        request = {'Bytes': source_img_bytes}
        response = client.index_faces(
            CollectionId=COLLECTION_NAME,
            Image=request,
            ExternalImageId=image_name,
            QualityFilter='AUTO',
            DetectionAttributes=['ALL']
        )

        face_records = response.get('FaceRecords', [])

        if not face_records:
            lst.append(f"No faces found in {image_name}'s image")
            lst.append("❌ Registration not completed")
        else:
            face_id = face_records[0]['Face']['FaceId']
            lst.append(f"✅ Face indexed for {image_name}")
            lst.append(f"Face ID: {face_id}")
            lst.append(f"Person name: {face_records[0]['Face']['ExternalImageId']}")
            lst.append("✔ Successfully registered face")

        return lst

    except ClientError as e:
        print("Error:", e)
        lst.append("❌ Registration failed: Ensure name has no spaces and try again")
        return lst


# Debugging / Direct Run
if __name__ == '__main__':
    img = "photo2.jpg"  # sample test image
    with open(img, 'rb') as source_image:
        source_bytes = source_image.read()

    results = add_face_to_collection(source_bytes, "Kishore_Neelavara", "Face_recognition_collection")
    for line in results:
        print(line)
