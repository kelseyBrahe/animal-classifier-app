from ultralytics import YOLO
import os
import logging
import piexif
import shutil

def get_img_classification(vars_model, vars_source_image_path, vars_target_image_path, vars_target_directory):
    """
    Function to use YOLO model to classify a single or multiple images with a folder. This function also 
    controls a number of other classifcation related functions such as logging, writing metadata tags
    and postprocesses to move out images from the main output folder to subdirectories with the respective
    animals name.
    """
    vars_img_results = {} #initialise variable to return results
    model = YOLO(vars_model) #load model
    results = model.predict(vars_source_image_path, conf=0.5, save=True, project=vars_target_image_path, name=vars_target_directory, exist_ok=True) #predict image
    
    log_filename = 'classification.log' #log file name hardcoded
    log_path = vars_target_image_path+'/'+vars_target_directory+'/'+log_filename # set location for log file
    logging.basicConfig(level=logging.INFO, filename=log_path) # configure info level logging. This will be used resume sessions that prematurely die
    
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
            logging.info('CLASSIFICATION COMPLETE: '+r.path.split('/')[-1]) # log each file or batch of files that are processed by a single function call
            add_metadata_tags_to_output_image_files(vars_target_image_path+'/'+vars_target_directory+'/'+r.path.split('/')[-1],r.names[r.boxes.cls.item()],r.boxes.conf.item())
                # add metadata per reqs
            write_file_to_subdirectory(r.path.split('/')[-1], r.names[r.boxes.cls.item()], vars_target_image_path, vars_target_directory)
                # copy otuput image from output main directory to animal specific subdirectory
        except RuntimeError:
            continue

    cleanup_image__output_directory(vars_target_image_path, vars_target_directory) # cleanup output directory

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

def add_metadata_tags_to_output_image_files(path, animal, confidence_score):
    """
    Function to add metadata to images. EXIF XPKeywords metadata field was chosen 
    as this appears underimage properties in Windows OS
    """
    exif_dict = piexif.load(path) #load any exif dictionary that may already exist for the image
    try:
        existing_xpkeywords = exif_dict["0th"][piexif.ImageIFD.XPKeywords] # fetch xpkeywords
        existing_xpkeywords = bytes(existing_xpkeywords)
        existing_xpkeywords = existing_xpkeywords[:-2].decode('utf-16-le') # decode xpkeywords bytes to string
        xpkeywords = existing_xpkeywords+','+ '{animal:'+animal+' confidence:'+str(confidence_score)+'}' #concat old tag with new tag
    except KeyError:
        xpkeywords = '{animal:'+animal+' confidence:'+str(confidence_score)+'}'
    exif_dict["0th"][piexif.ImageIFD.XPKeywords] = xpkeywords.encode("utf-16le")
    exif_bytes = piexif.dump(exif_dict) # convert stirng to bytes
    piexif.insert(exif_bytes, path) # add metadata

def write_file_to_subdirectory(file, animal, vars_target_image_path, vars_target_directory):
    """
    Copy output files with bounding boxes to animal specific subdirectories
    """
    if check_whether_target_directory_exists(vars_target_image_path+'/'+vars_target_directory+'/'+animal)==False:
        os.mkdir(vars_target_image_path+'/'+vars_target_directory+'/'+animal)
    if check_whether_target_directory_already_has_file(vars_target_image_path+'/'+vars_target_directory+'/'+animal, file)==False:
        shutil.copy(vars_target_image_path+'/'+vars_target_directory+'/'+file, vars_target_image_path+'/'+vars_target_directory+'/'+animal)

def cleanup_image__output_directory(vars_target_image_path, vars_target_directory):
    """
    Function to remove output images that have been copied to an animal specific subdirectory from the 
    main output directory
    """
    for filename in os.listdir(vars_target_image_path+'/'+vars_target_directory):
        if filename.split('.')[-1].lower() == 'jpg':
            try:
                os.remove(vars_target_image_path+'/'+vars_target_directory+'/'+filename)
            except FileNotFoundError:
                pass


if __name__=="__main__":
    """FOR TESTING PURPOSES"""
    vars_model = 'src/models/best.pt'
    # vars_source_image_path = '/workspaces/animal-classifier-app/src/IMG_0290.jpg'
    vars_source_image_path = '/workspaces/animal-classifier-app/src/few_images'
    vars_target_image_path = '/workspaces/animal-classifier-app/src'
    vars_target_directory = 'output'
    results = get_img_classification(vars_model, vars_source_image_path, vars_target_image_path, vars_target_directory)
    print(results)
