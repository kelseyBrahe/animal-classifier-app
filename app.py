from flask import Flask, render_template, request, url_for
from flask import render_template
from PIL import Image
from werkzeug.utils import send_from_directory, secure_filename
# import cv2
# import tkinter as tk
# from tkinter import filedialog
# import numpy as np
import os
from ultralytics import YOLO
from classifer import get_img_classification, check_whether_target_directory_exists

UPLOAD_FOLDER = '/uploads'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def hello_world():
    return render_template(
        "index.html"
    )

#
# This function allows the user to upload multiple files that
# will save to the /uploads folder.
# From here we can loop through the images to process them.
# We will need to clean the uploads folder after processing the images.
@app.route('/upload', methods=["GET", "POST"])
def uploadFiles():
    if request.method == "POST":
        basepath = os.path.dirname(__file__)
        files = request.files.getlist('files')
        if 'files' in request.files:
            try:
                for f in files:
                    filename = secure_filename(f.filename)
                    filepath = os.path.join(basepath, 'uploads', filename)
                    f.save(filepath)
                return render_template("index.html")
            except FileNotFoundError:
                print('File not found.')
                return render_template("index.html")

#
# This function will get the value of the bounding box and call
# the methods for image prediction.
# If the bounding box is selected 'boundingBox' will equal "on", else None
# It will need to process all of the images in the uploads folder,
# save them to the /results folder, call a display function and then
# clean the uploads folder.
@app.route('/predict', methods=["GET", "POST"])     
def predict():
    if request.method == "POST":
        # this is the value of the bounding box
        boundingBox = request.form.get("bBox")
        print(boundingBox == "on")
    return render_template("index.html")


# """Interacting with the classifier module"""
# # First check whether the target directory already exists - this will save previous results being overwritten accidently. Consider having a popup confirm intention with user
# print(check_whether_target_directory_exists('/workspaces/animal-classifier-app/src'+'/'+'output'))
# # Second request a single image to be classified; this also works for multiple images if you provide a source path for a directory
# results = get_img_classification('src/models/best.pt', '/workspaces/animal-classifier-app/src/IMG_0290.jpg', '/workspaces/animal-classifier-app/src', 'output')
# print(results)
