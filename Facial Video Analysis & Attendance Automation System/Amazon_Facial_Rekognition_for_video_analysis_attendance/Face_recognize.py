import pandas as pd
import boto3
import io
from PIL import Image, ImageDraw, ImageFont

# Read AWS credentials from CSV file

credential = pd.read_csv("facial_video_analysis_accessKeys.csv")
access_key_id = credential['Access key ID'][0]
secret_access_key = credential['Secret access key'][0]
region_name = 'us-east-2'  # Specify your AWS region here

# Initialize AWS Rekognition client
AWS_REKOG = boto3.client(
    'rekognition',
    aws_access_key_id=access_key_id,
    aws_secret_access_key=secret_access_key,
    region_name=region_name  # Add this line to specify the region
)


def get_bounding_boxes(request):
    response = AWS_REKOG.detect_faces(Image=request, Attributes=['ALL'])
    bounding_boxes = [details['BoundingBox'] for details in response['FaceDetails']]
    return bounding_boxes


def face_exists(request):
    response = AWS_REKOG.detect_faces(Image=request, Attributes=['ALL'])
    return bool(response['FaceDetails'])


def get_face_name(face, image, COLLECTION_NAME):
    img_width, img_height = image.size
    width = img_width * face['Width']
    height = img_height * face['Height']
    left = img_width * face['Left']
    top = img_height * face['Top']
    area = (left, top, left + width, top + height)
    cropped_image = image.crop(area)
    bytes_array = io.BytesIO()
    cropped_image.save(bytes_array, format="PNG")
    request = {'Bytes': bytes_array.getvalue()}

    if face_exists(request):
        response = AWS_REKOG.search_faces_by_image(
            CollectionId=COLLECTION_NAME, Image=request, FaceMatchThreshold=70)
        if response['FaceMatches']:
            return response['FaceMatches'][0]['Face']['ExternalImageId']
        else:
            return 'Not recognized'
    return ''


def face_recognition_saving_image(image, COLLECTION_NAME):
    bytes_array = io.BytesIO()
    image.save(bytes_array, format="PNG")
    request = {'Bytes': bytes_array.getvalue()}
    bounding_boxes = get_bounding_boxes(request)
    img_width, img_height = image.size
    faces_name = [get_face_name(face, image, COLLECTION_NAME) for face in bounding_boxes]

    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", size=40)
    recognized_faces = []

    for i, face_name in enumerate(faces_name):
        if not face_name:
            continue
        width = img_width * bounding_boxes[i]['Width']
        height = img_height * bounding_boxes[i]['Height']
        left = img_width * bounding_boxes[i]['Left']
        top = img_height * bounding_boxes[i]['Top']
        points = ((left, top), (left + width, top), (left + width, top + height), (left, top + height), (left, top))
        draw.line(points, fill='#00d400', width=4)
        draw.text((left, top), face_name, font=font)
        recognized_faces.append(
            f'A face has been recognized. Name: {face_name}' if face_name != "Not recognized" else "Not recognized face")
        print(recognized_faces[-1])

    print('Faces recognition has finished.')
    return image, recognized_faces


# Example usage
if __name__ == '__main__':
    COLLECTION_NAME = 'Face_recognition_collection'
    img = "photo1.jpg"
    source_img = Image.open(img)
    result_img, recognized_faces = face_recognition_saving_image(source_img, COLLECTION_NAME)
    result_img.save("output.png")  # Save the image with recognized faces
