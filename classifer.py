from ultralytics import YOLO
import os
from flask import Flask
from PIL import Image


app = Flask(__name__)

def get_img_classification(vars_model, vars_source_image_path, vars_target_image_path, vars_target_directory):
    """
    Function to use YOLO model to classify a single or multiple images with a folder. This function also 
    controls a number of other classifcation related functions such as logging, writing metadata tags
    and postprocesses to move out images from the main output folder to subdirectories with the respective
    animals name.
    """
    vars_img_results = {} #initialise variable to return results
    model = YOLO(vars_model) #load model
    # results = model.predict(vars_source_image_path, conf=0.5, save=True, project=vars_target_image_path, name=vars_target_directory, exist_ok=True) #predict image
    results = model.predict(vars_source_image_path, conf=0.5, exist_ok=True)

    vars_img_results = {
                    'source_file_path': vars_source_image_path
                    , 'target_file_path': '{}/{}/{}'.format(vars_target_image_path, vars_target_directory, vars_source_image_path.split('/', -1)[0])
                    , 'model_variant': vars_model
                    } #write header data
    box_list=[]
    
    for r in results:
        try:     
            # Access the original image from the result
            image_array = r.plot()
                
            # Convert the image array to PIL Image
            image = Image.fromarray(image_array)
                
            # Generate a filename for the image
            filename = os.path.basename(r.path)

            # Check if there are any boxes detected
            if len(r.boxes.data) == 0:
                class_name = "unknown"
            else:
                class_id = int(r.boxes.data[0][-1])
                class_name = model.names[class_id]

            save_path = os.path.join(app.static_folder, "results", class_name)
            os.makedirs(save_path, exist_ok=True)
                
            # Save the image using PIL
            image.save(os.path.join(save_path, filename))
            
            var = r.boxes.cls.item()
            box_list_item = {
                            'animal': r.names[r.boxes.cls.item()],
                            'confidence_score': r.boxes.conf.item(),
                            'box_coordinates_normalised': r.boxes.xywhn.numpy().tolist()[0],
                            'image_path': r.path,
                            'image_filename': filename
                            } # write box payload data - can be multiple boxes/classifications
            box_list.append(box_list_item)
        except RuntimeError:
            continue

    vars_img_results['boxes'] = box_list 
    
    return vars_img_results

def check_whether_target_directory_exists(dir):
    """
    Function to check whether a directory exists
    """
    return os.path.exists(dir)

def check_whether_target_directory_already_has_log_file(dir):
    """
    Function to check whether the log file exists
    """
    return os.path.isfile(dir+'/classification.log')

def check_whether_target_directory_already_has_file(dir, file):
    """
    Function to check whether a file already exists in a directory
    """
    return os.path.isfile(dir+'/'+file)

def files_already_classified_in_previous_session(dir):
    """
    Function to get a list of all files already processed from the log file. This can be called from the main app
    in the event of a session dying and user wanting to resume session from where it left off. This could be handy 
    if a large number of images are being processed. In the main app function check_whether_target_directory_already_has_log_file()
    can be used to check if there is a log, this function can be used to get process files from the log then app can exclude 
    processed files from loop which call function get_img_classification()
    """
    with open(dir+'/classification.log') as file:
        filename_list = [line.rstrip().split('CLASSIFICATION COMPLETE: ')[-1] for line in file] 
    filename_list = list(set(filename_list))
    return filename_list

if __name__=="__main__":
    """FOR TESTING PURPOSES"""
    vars_model = 'src/models/best.pt'
    # vars_source_image_path = '/workspaces/animal-classifier-app/src/IMG_0290.jpg'
    vars_source_image_path = '/workspaces/animal-classifier-app/src/few_images'
    vars_target_image_path = '/workspaces/animal-classifier-app/src'
    vars_target_directory = 'output'
    results = get_img_classification(vars_model, vars_source_image_path, vars_target_image_path, vars_target_directory)
    print(results)
