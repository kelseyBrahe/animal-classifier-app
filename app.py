from flask import Flask
from classifer import get_img_classification

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

results = get_img_classification('src/models/best.pt', '/workspaces/animal-classifier-app/src/IMG_0290.jpg', '/workspaces/animal-classifier-app/src', 'output')
print(results)