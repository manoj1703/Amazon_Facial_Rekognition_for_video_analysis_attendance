import cv2
import boto3
import io
import pandas as pd
import certifi
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ================= AWS Credentials =================
credential = pd.read_csv("facial_video_analysis_accessKeys.csv")
access_key_id = credential['Access key ID'][0]
secret_access_key = credential['Secret access key'][0]
region_name = 'us-east-2'

AWS_REKOG = boto3.client(
    "rekognition",
    aws_access_key_id=access_key_id,
    aws_secret_access_key=secret_access_key,
    region_name=region_name,
    verify=certifi.where()
)

# ================= Helper Functions =================
def get_bounding_boxes(request):
    try:
        response = AWS_REKOG.detect_faces(Image=request, Attributes=['ALL'])
        return [details['BoundingBox'] for details in response['FaceDetails']]
    except Exception as e:
        print(f"⚠ Rekognition error in get_bounding_boxes: {e}")
        return []


def face_exists(request):
    try:
        response = AWS_REKOG.detect_faces(Image=request, Attributes=['ALL'])
        return bool(response['FaceDetails'])
    except Exception as e:
        print(f"⚠ Rekognition error in face_exists: {e}")
        return False


def get_face_name(face_box, pil_image, collection_name):
    """Crop face, search in Rekognition collection, return recognized name"""
    try:
        img_width, img_height = pil_image.size
        width = img_width * face_box['Width']
        height = img_height * face_box['Height']
        left = img_width * face_box['Left']
        top = img_height * face_box['Top']

        cropped = pil_image.crop((left, top, left + width, top + height))
        bytes_array = io.BytesIO()
        cropped.save(bytes_array, format="PNG")
        request = {"Bytes": bytes_array.getvalue()}

        if face_exists(request):
            response = AWS_REKOG.search_faces_by_image(
                CollectionId=collection_name,
                Image=request,
                FaceMatchThreshold=70
            )
            if response.get("FaceMatches"):
                return response["FaceMatches"][0]["Face"]["ExternalImageId"]
            else:
                return "Not recognized"
        return ""
    except Exception as e:
        print(f"⚠ Rekognition error in get_face_name: {e}")
        return "Error"


def recognize_faces_in_frame(frame, collection_name):
    """Recognize faces in a single video frame"""
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_width, img_height = pil_image.size

    bytes_array = io.BytesIO()
    pil_image.save(bytes_array, format="PNG")
    request = {"Bytes": bytes_array.getvalue()}

    bounding_boxes = get_bounding_boxes(request)
    faces_name = [get_face_name(box, pil_image, collection_name) for box in bounding_boxes]

    draw = ImageDraw.Draw(pil_image)
    try:
        font = ImageFont.truetype("arial.ttf", size=20)
    except:
        font = ImageFont.load_default()

    recognized_faces = []
    for i, face_name in enumerate(faces_name):
        if not face_name:
            continue
        box = bounding_boxes[i]
        width = img_width * box['Width']
        height = img_height * box['Height']
        left = img_width * box['Left']
        top = img_height * box['Top']

        points = ((left, top), (left + width, top),
                  (left + width, top + height), (left, top + height), (left, top))
        draw.line(points, fill='#00d400', width=3)
        draw.text((left, max(0, top - 25)), face_name, font=font, fill="red")
        recognized_faces.append(face_name)

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR), recognized_faces


# ================= Video Feed for Flask =================
def rekognition_video_feed(collection_name):
    """Generator function that streams video frames with recognition"""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    frame_count = 0
    last_recognized = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Only send every 5th frame to AWS to reduce cost/latency
        if frame_count % 5 == 0:
            frame, last_recognized = recognize_faces_in_frame(frame, collection_name)
        else:
            # Just show frame without processing
            if last_recognized:
                cv2.putText(frame, f"Last recognized: {', '.join(last_recognized)}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            continue

        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    cap.release()
