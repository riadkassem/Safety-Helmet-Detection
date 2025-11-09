from ultralytics import solutions

inf = solutions.Inference(
    model="/home/archlinux/Desktop/Projects/CV/Safety_Helmet_Detection/my_model/my_model.pt",  # you can use any model that Ultralytics support, i.e. YOLO11, YOLOv10
)

inf.inference()

# Make sure to run the file using command `streamlit run path/to/file.py`