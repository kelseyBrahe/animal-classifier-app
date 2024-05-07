#  COSC591 Group B Classifier app

This is the source code for the image classification app being developed by COSC591 Group B for T1 2024. The aim of the app is to classify single or multiple images of identified species. 

## Usage

### Run the app directly on your machine

1. Ensure all dependencies listed in requirements.txt are installed
```bash
pip install -r requirements.txt
```
2. Run
```bash
python app.py
```
**********
To run the flask app, 
1. run 
```bash
python -m flask run
```
2. Crtl + click on the URL provided in the terminal

### Run the app on a VSCode devcontainer
1. Ensure devcontainer plugin is installed in your VSCode and Docker is installed on your machine
2. Open folder with code and you should be prompted to reopen in container. If not, get the devcontainer options in your command palette and select reopen in container. When building the container for the first time expect to wait (13GB of dependencies on a minimal ubuntu OS)
3. Run
```bash
python app.py
```

## Contributing
Limited to members of COSC591 Group B only

## License
TBC