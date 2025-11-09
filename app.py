from ultralytics import solutions

## Change path to where the model.pt is
inf = solutions.Inference(model="my_model/my_model.pt")
inf.inference()

# Make sure to run the file using command `streamlit run path/to/file.py`
