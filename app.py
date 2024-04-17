from flask import Flask
from flask import render_template
from classifer import get_img_classification, check_whether_target_directory_exists

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template(
        "index.html"
    )


# """Interacting with the classifier module"""
# # First check whether the target directory already exists - this will save previous results being overwritten accidently. Consider having a popup confirm intention with user
# print(check_whether_target_directory_exists('/workspaces/animal-classifier-app/src'+'/'+'output'))
# # Second request a single image to be classified; this also works for multiple images if you provide a source path for a directory
# results = get_img_classification('src/models/best.pt', '/workspaces/animal-classifier-app/src/IMG_0290.jpg', '/workspaces/animal-classifier-app/src', 'output')
# print(results)
