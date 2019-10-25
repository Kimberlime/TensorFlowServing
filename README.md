## README

1. Shell script
    * run the script with the following command ```./infer_with_saved_model.sh```
    * `infer_with_saved_model.sh` installs requirements, pulls a docker image, run the docker image, and does an inference. 

2. Implmentation
    * Implemented export function and inference function with tensorflow serving.
    * `export_frozen_to_saved_model.py` exports frozen graph to SavedModel format. 
    * `image_example.py` has function **infer_with_serving_client()** 
        * ```infer_with_serving_client(image_data, url, return_elements)``` makes an inference with tensorflow serving by http request.
The numpy array of image data shape should be (1, width, height, 3)

For the detailed implementation process, check <https://github.com/Kimberlime/TensorFlowServing>