from ultralytics import YOLO

def get_img_classification(vars_model, vars_source_image_path, vars_target_image_path, vars_target_directory):
    vars_img_results = {} #initialise variable to return results
    model = YOLO(vars_model) #load model
    results = model.predict(vars_source_image_path, conf=0.5, save=True, project=vars_target_image_path, name=vars_target_directory, exist_ok=True) #predict image
    vars_img_results = {
                    'file_name': vars_source_image_path.split('/')[-1]
                    , 'source_file_path': vars_source_image_path
                    , 'target_file_path': '{}/{}/{}'.format(vars_target_image_path, vars_target_directory, vars_source_image_path.split('/', -1)[0])
                    , 'model_variant': vars_model
                    } #write header data
    for r in results:
        vars_img_results['box{}'.format(results.index(r))] = {
                                                            'animal': r.names[r.boxes.cls.item()]
                                                            , 'confidence_score': r.boxes.conf.item()
                                                            , 'box_coordinates_normalised': r.boxes.xywhn.numpy().tolist()[0]
                                                            } # write box payload data - can be multiple boxes/classifications
    return vars_img_results

if __name__=="__main__":
    vars_model = 'src/models/best.pt'
    vars_source_image_path = '/workspaces/animal-classifier-app/src/IMG_0290.jpg'
    vars_target_image_path = '/workspaces/animal-classifier-app/src'
    vars_target_directory = 'output'
    results = get_img_classification(vars_model, vars_source_image_path, vars_target_image_path, vars_target_directory)
    print(results)

