import os
import io
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify, Response
from werkzeug.utils import secure_filename
from PIL import Image

# Import your existing modules
from Create_Collection import list_collections, create, delete
from Register_Faces import add_face_to_collection
from Face_recognize import face_recognition_saving_image

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

# Webcam
camera = cv2.VideoCapture(0)

# Global variable to hold recognized names
recognized_faces = []


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def start_page():
    return render_template('index.html')


# ================= COLLECTIONS ===================
@app.route('/collection_page')
def collection_page():
    count, lst = list_collections()
    return render_template('collection.html', count=count, lst=lst)


@app.route('/create_page', methods=['POST'])
def create_page():
    COLLECTION_NAME = str(request.form['collection-name']).strip()
    statement = create(COLLECTION_NAME)
    count, lst = list_collections()
    return render_template('collection.html', count=count, lst=lst, statement=statement)


@app.route('/delete_page')
def delete_page():
    COLLECTION_NAME = request.args.get('name')
    statement = delete(COLLECTION_NAME)
    count, lst = list_collections()
    return render_template('collection.html', count=count, lst=lst, statement=statement)


# ================= REGISTER ===================
@app.route('/register_page')
def register_page():
    count, lst = list_collections()
    return render_template('register.html', lst=lst)


@app.route('/register_faces', methods=['POST'])
def register_faces():
    required_files = ['file_front', 'file_left', 'file_right']
    missing_files = [f for f in required_files if f not in request.files or request.files[f].filename == '']

    if missing_files:
        statement = f'Missing files: {", ".join(missing_files)}'
        count, lst = list_collections()
        return render_template('register.html', lst=lst, statement=statement)

    name = str(request.form['person-name']).strip()
    COLLECTION_NAME = request.form['collection']

    registration_result = []
    saved_filenames = []

    for file_key in required_files:
        file = request.files[file_key]
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)

            # Open and convert to bytes
            Register_image = Image.open(save_path)
            bytes_array = io.BytesIO()
            Register_image.save(bytes_array, format="PNG")
            source_image_bytes = bytes_array.getvalue()

            # Register in AWS Rekognition
            res = add_face_to_collection(source_image_bytes, name, COLLECTION_NAME)
            registration_result.extend(res)
            saved_filenames.append(filename)
        else:
            registration_result.append(f"Invalid file type for {file_key}")

    count, lst = list_collections()
    return render_template('register.html', lst=lst, reg_lst=registration_result, filename=saved_filenames)


# ================= RECOGNIZE STATIC ===================
@app.route('/recognize_page')
def recognize_page():
    count, lst = list_collections()
    return render_template('recognize.html', lst=lst)


@app.route('/recognize_faces', methods=['POST'])
def recognize_faces():
    if 'file' not in request.files:
        statement = 'No file part'
        count, lst = list_collections()
        return render_template('recognize.html', lst=lst, statement=statement)

    file = request.files['file']
    if file.filename == '':
        statement = 'No image selected for uploading'
        count, lst = list_collections()
        return render_template('recognize.html', lst=lst, statement=statement)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        Register_image = Image.open('static/uploads/' + filename)

        COLLECTION_NAME = request.form['collection']
        count, lst = list_collections()

        path = "result/" + filename
        result_img, res_lst = face_recognition_saving_image(Register_image, COLLECTION_NAME)
        os.makedirs("static/result", exist_ok=True)
        result_img.save('static/' + path)

        return render_template('recognize.html', lst=lst, filename=path, res_lst=res_lst)
    else:
        statement = 'Allowed image types are -> png, jpg, jpeg, gif'
        count, lst = list_collections()
        return render_template('recognize.html', lst=lst, statement=statement)


# ================= LIVE RECOGNITION ===================
def gen_frames(collection_name):
    global recognized_faces
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Convert frame (OpenCV BGR -> PIL RGB)
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result_img, res_lst = face_recognition_saving_image(img_pil, collection_name)

            # Update recognized names
            recognized_faces = res_lst

            # Convert back to OpenCV BGR
            result_frame = cv2.cvtColor(np.array(result_img), cv2.COLOR_RGB2BGR)

            # Encode JPEG
            ret, buffer = cv2.imencode('.jpg', result_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/recognize_live')
def recognize_live():
    count, lst = list_collections()
    return render_template('recognize_live.html', lst=lst)


@app.route('/video_feed')
def video_feed():
    collection_name = request.args.get("collection")
    return Response(gen_frames(collection_name),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/recognized_names')
def recognized_names():
    global recognized_faces
    return jsonify({"names": recognized_faces})


# ================= CACHE DISABLE ===================
@app.after_request
def add_header(response):
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'
    return response


if __name__ == '__main__':
    app.run(port=5002, debug=True)
