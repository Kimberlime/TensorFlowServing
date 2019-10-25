pip install -r requirements.txt

sudo docker pull kk316497/object_detection_serving

python ./export_frozen_to_saved_model.py

sudo docker run --name object_detection_container -p 8501:8501 -t kk316497/object_detection_serving &

python ./image_example.py
