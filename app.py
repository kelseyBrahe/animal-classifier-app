from flask import Flask, render_template, request
from flask import render_template
from pathlib import Path
from werkzeug.utils import secure_filename
import os
import requests
from classifer import get_img_classification

UPLOAD_FOLDER = '/uploads'
IMG_COUNTER = 0
SUBFOLDER = ''

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
IMAGES = os.listdir(os.path.join(app.static_folder, "results"))

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
#
@app.route('/upload', methods=["POST"])
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
                return render_template("index.html", ready=True)
            except FileNotFoundError:
                print('File not found.')
                return render_template("index.html", empty=True)

#
# This function will get the value of the bounding box and call
# the methods for image prediction.
# If the bounding box is selected 'boundingBox' will equal "on", else None
# It will need to process all of the images in the uploads folder,
# save them to the /results folder
#
@app.route('/predict', methods=["POST"])     
def predict():
    if request.method == "POST":
        # run image prediction and save to results
        source_directory = os.path.join(os.path.dirname(__file__), 'uploads')
        if(len(os.listdir(source_directory)) == 0):
            return render_template("index.html", empty=True)
        else:
            results = get_img_classification('src/models/best.pt', source_directory, app.static_folder, "results")
            # get statistics
            for result in results.get('boxes'):
                animal = str(result.get('animal'))
                confidence_score = str(result.get('confidence_score'))
                box_coordinates_normalised = str(result.get('box_coordinates_normalised'))
                image_filename = str(result.get('image_filename'))
                # get path to correct subfolder
                save_path = os.path.join(app.static_folder, "results" + "/" + animal)
                # save to a txt file
                filename = os.path.join(save_path, image_filename.split('.')[0] + ".txt")
                file = open(filename, "w")
                file.write(
                    animal + "\n" +
                    confidence_score + "\n" +
                    box_coordinates_normalised + "\n" +
                    image_filename + "\n"
                )
                file.close()
            
            # get list of all subdirectories in /results
            subfolders = os.listdir(os.path.join(app.static_folder, "results"))
            # clear the uploads folder
            [f.unlink() for f in Path(os.path.join(os.path.dirname(__file__), 'uploads')).glob("*") if f.is_file()]
            
            return render_template("index.html", subfolders=subfolders, complete=True)

#
# This image will display the subfolders created during image prediction
#
@app.route('/view_images', methods=["POST"])
def view():
    # get list of all subdirectories in /results
    subfolders = os.listdir(os.path.join(app.static_folder, "results"))
    return render_template("index.html", subfolders=subfolders)

#
# Display images in selected subfolder
#
@app.route('/display', methods=["POST"])
def display():
    subfolder = request.form['subfolder']
    # save information for displaying images
    global IMAGES 
    IMAGES = os.listdir(os.path.join(app.static_folder, "results/" + subfolder))
    global IMG_COUNTER
    IMG_COUNTER = 0
    global SUBFOLDER
    SUBFOLDER = subfolder
    current_img = "/static/results/" + SUBFOLDER + "/" + IMAGES[IMG_COUNTER]
    while(
        not current_img.lower().endswith(('.png', '.jpg', '.jpeg', 'tiff', '.bmp', '.gif'))
        and IMG_COUNTER != len(IMAGES) - 1
    ):
        IMG_COUNTER += 1
        current_img = "/static/results/" + SUBFOLDER + "/" + IMAGES[IMG_COUNTER]
    
    stats = display_stats(IMAGES[IMG_COUNTER].split('.')[0], SUBFOLDER)
    if stats is not None:
        return render_template("index.html", 
                            current_img=current_img,
                            animal=stats[0],
                            confidence_score=stats[1],
                            box_coordinates_normalised = stats[2],
                            image_filename = stats[3]
                            )
    else:
        return render_template("index.html", current_img=current_img)

#
# Navigate back to the list of subfolders
#
@app.route('/back', methods=["POST"])
def back():
    # get list of all subdirectories in /results
    subfolders = os.listdir(os.path.join(app.static_folder, "results"))
    return render_template("index.html", subfolders=subfolders)

#
# Navigates to the previous image, keeping track of place in results folder
#
@app.route('/previous_img', methods=["POST"])
def previous():
    global IMG_COUNTER
    global IMAGES
    if IMG_COUNTER != 0:
        IMG_COUNTER = IMG_COUNTER - 1
    
    current_img = "/static/results/" + SUBFOLDER + "/" + IMAGES[IMG_COUNTER]
    while(
        not current_img.lower().endswith(('.png', '.jpg', '.jpeg', 'tiff', '.bmp', '.gif'))
        and IMG_COUNTER != -1
    ):
        IMG_COUNTER -= 1
        current_img = "/static/results/" + SUBFOLDER + "/" + IMAGES[IMG_COUNTER]
    
    stats = display_stats(IMAGES[IMG_COUNTER].split('.')[0], SUBFOLDER)
    if stats is not None:
        return render_template("index.html", 
                            current_img=current_img,
                            animal=stats[0],
                            confidence_score=stats[1],
                            box_coordinates_normalised = stats[2],
                            image_filename = stats[3]
                            )
    else:
        return render_template("index.html", current_img=current_img)

#
# Navigates to the next image, keeping track of place in results folder
#
@app.route('/next_img', methods=["POST"])
def next():
    global IMG_COUNTER
    global IMAGES
    if IMG_COUNTER != len(IMAGES) - 1:
        IMG_COUNTER = IMG_COUNTER + 1
    
    current_img = "/static/results/" + SUBFOLDER + "/" + IMAGES[IMG_COUNTER]
    while(
        not current_img.lower().endswith(('.png', '.jpg', '.jpeg', 'tiff', '.bmp', '.gif'))
        and IMG_COUNTER != len(IMAGES) - 1
        and IMG_COUNTER + 1 != len(IMAGES) -1
    ):
        IMG_COUNTER += 1
        current_img = "/static/results/" + SUBFOLDER + "/" + IMAGES[IMG_COUNTER]
    
    
    stats = display_stats(IMAGES[IMG_COUNTER].split('.')[0], SUBFOLDER)
    if stats is not None:
        return render_template("index.html", 
                            current_img=current_img,
                            animal=stats[0],
                            confidence_score=stats[1],
                            box_coordinates_normalised = stats[2],
                            image_filename = stats[3]
                            )
    else:
        return render_template("index.html", current_img=current_img)

def display_stats(image_filename, subfolder):
    filename = os.path.join(app.static_folder, "results/" + subfolder + "/" + image_filename + ".txt")
    if (Path(filename).exists()):
        with open(filename) as file:
            lines = [line.rstrip() for line in file]
        return lines