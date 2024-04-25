from ultralytics import YOLO
import os
import logging

def get_img_classification(vars_model, vars_source_image_path, vars_target_image_path, vars_target_directory):

    log_filename = 'classification.log'
    log_path = vars_target_image_path+'/'+vars_target_directory+'/'+log_filename
    logging.basicConfig(level=logging.INFO, filename=log_path)

    vars_img_results = {} #initialise variable to return results
    model = YOLO(vars_model) #load model
    results = model.predict(vars_source_image_path, conf=0.5, save=True, project=vars_target_image_path, name=vars_target_directory, exist_ok=True) #predict image
    vars_img_results = {
                    'source_file_path': vars_source_image_path
                    , 'target_file_path': '{}/{}/{}'.format(vars_target_image_path, vars_target_directory, vars_source_image_path.split('/', -1)[0])
                    , 'model_variant': vars_model
                    } #write header data
    box_list=[]
    for r in results:
        try:
            var = r.boxes.cls.item()
            box_list_item = {
                            'animal': r.names[r.boxes.cls.item()]
                            , 'confidence_score': r.boxes.conf.item()
                            , 'box_coordinates_normalised': r.boxes.xywhn.numpy().tolist()[0]
                            , 'image_path': r.path
                            , 'image_filename': r.path.split('/')[-1]
                            } # write box payload data - can be multiple boxes/classifications
            box_list.append(box_list_item)
            logging.info('CLASSIFICATION COMPLETE: '+r.path.split('/')[-1]) 
        except RuntimeError:
            continue

    vars_img_results['boxes'] = box_list 
    
    return vars_img_results

def check_whether_target_directory_exists(dir):
    return os.path.exists(dir)

def check_whether_target_directory_already_has_log_file(dir):
    return os.path.isfile(dir+'/classification.log')

def files_already_classified_in_previous_session(dir):
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
    print(check_whether_target_directory_already_has_log_file(vars_target_image_path+'/'+vars_target_directory))
    results = get_img_classification(vars_model, vars_source_image_path, vars_target_image_path, vars_target_directory)
    print(results)
    print(check_whether_target_directory_exists(vars_target_image_path+'/'+vars_target_directory))
    print(check_whether_target_directory_already_has_log_file(vars_target_image_path+'/'+vars_target_directory))
    print(files_already_classified_in_previous_session(vars_target_image_path+'/'+vars_target_directory))

