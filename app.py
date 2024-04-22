from flask import Flask, render_template, request
from flask import render_template
from PIL import Image
from werkzeug.utils import send_from_directory
import cv2
import numpy as np
import os
from ultralytics import YOLO
from classifer import get_img_classification, check_whether_target_directory_exists

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template(
        "index.html"
    )

# this allows the upload of a single file
# nothing is done with it currently and it is not saved anywhere
@app.route('/', methods=['GET', 'POST'])
def predict_img():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            # save uploaded image into the uploads folder
            filepath = os.path.join(basepath, 'uploads', f.filename)
            f.save(filepath)

            # perform image prediction
            global imgpath
            predict_img.imgpath = f.filename
            img = cv2.imread(filepath)
            frame = cv2.imencode('.jpg', cv2.UMat(img))[1].toBytes()
            image = Image.open(io.BytesIO(frame))
            yolo = YOLO('yolo_filename.pt')
            detections = yolo.predict(image, save=True)
            return display(f.filename)

@app.route('/<path:filename>')
def display(filename):
    folder_path = 'runs/detect'
    files = os.listdir(folder_path)
    latest_file = files[0]
    environ = request.environ

    return send_from_directory(folder_path, latest_file, environ)

# """Interacting with the classifier module"""
# # First check whether the target directory already exists - this will save previous results being overwritten accidently. Consider having a popup confirm intention with user
# print(check_whether_target_directory_exists('/workspaces/animal-classifier-app/src'+'/'+'output'))
# # Second request a single image to be classified; this also works for multiple images if you provide a source path for a directory
# results = get_img_classification('src/models/best.pt', '/workspaces/animal-classifier-app/src/IMG_0290.jpg', '/workspaces/animal-classifier-app/src', 'output')
# print(results)
