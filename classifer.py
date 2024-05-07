from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename
from flask import Flask
from PIL import Image
import torch

app = Flask(__name__)

def get_img_classification(vars_model, vars_source_image_path, vars_target_image_path, vars_target_directory):
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
            print("IF CHECK")
            if len(r.boxes.data) == 0:
                class_name = "unknown"
                print("CLASS NAME")
                print(class_name)
            else:
                class_id = int(r.boxes.data[0][-1])
                class_name = model.names[class_id]
                print("CLASS NAME")
                print(class_name)

            print("CLASS NAME")
            print(class_name)
            
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
    return os.path.exists(dir)

if __name__=="__main__":
    vars_model = 'src/models/best.pt'
    # vars_source_image_path = '/workspaces/animal-classifier-app/src/IMG_0290.jpg'
    vars_source_image_path = '/workspaces/animal-classifier-app/src/few_images'
    vars_target_image_path = '/workspaces/animal-classifier-app/src'
    vars_target_directory = 'output'
    results = get_img_classification(vars_model, vars_source_image_path, vars_target_image_path, vars_target_directory)
    print(results)
    print(check_whether_target_directory_exists(vars_target_image_path+'/'+vars_target_directory))


