# Safety Helmet Detection

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Ultralytics YOLO](https://img.shields.io/badge/YOLO-Ultralytics-orange)
![License](https://img.shields.io/badge/license-MIT-green)

A real-time object detection project for detecting safety helmets using YOLO. This project supports inference on images, videos, and webcam streams, and also includes a Streamlit web interface for easy visualization.

---

## 📦 Requirements

Install the necessary Python packages:

```bash
pip3 install pillow pyyaml ultralytics
```

---

## 🗂 Dataset Preparation

Split your dataset into training, validation, and test sets:

```bash
python train_val_split.py --datapath="dataset" --train_pct=0.70 --val_pct=0.15 --test_pct=0.15
```

- `dataset/` should contain images and annotations.
- Adjust the percentages to control the split.

---

## 🚀 Training

Train the YOLO model on your dataset:

```bash
yolo detect train data=data.yaml model=yolo11s.pt epochs=30 imgsz=640
```

- `data.yaml` contains paths to your train/val/test datasets and class names.
- `epochs` and `imgsz` can be modified depending on your hardware.

---

## 🔍 Prediction / Inference

Run detection on test images:

```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=data/test/images save=True conf=0.25
```

View sample predictions in a notebook:

```python
import glob
from IPython.display import Image, display

for image_path in glob.glob('runs/detect/predict/*.jpg')[:10]:
    display(Image(filename=image_path, height=400))
    print('\n')
```

---

## ✅ Validation

Evaluate model performance:

```bash
yolo val model=runs/detect/train/weights/best.pt data=data.yaml
```

---

## 💾 Save Model

Store model weights and training results:

```bash
mkdir my_model
cp runs/detect/train/weights/best.pt my_model/my_model.pt
cp -r runs/detect/train my_model

cd my_model
zip my_model.zip my_model.pt
zip -r my_model.zip train
```

---

## 🎥 Inference Options

**Video file inference:**

```bash
python yolo_detect.py --model my_model.pt --source '../video.mp4' --resolution 1280x720
```

**Real-time webcam inference:**

```bash
python yolo_detect.py --model my_model.pt --source usb0 --resolution 1280x720
```

**Streamlit web app for interactive inference:**

```bash
streamlit run streamlit-inference.py
```

---

## 📂 Project Structure

```
CV/
├─ dataset/                # Raw dataset (images + annotations)
├─ train_val_split.py      # Dataset splitting script
├─ yolo_detect.py          # Detection script for images, videos, and webcam
├─ streamlit-inference.py  # Streamlit app for real-time detection
├─ my_model/               # Trained model & results
├─ data.yaml               # YOLO dataset configuration
├─ yolo11s.pt              # Pre-trained YOLO model
└─ yolo11n.pt              # Optional lightweight YOLO model
```

---

## ⚡ Notes

- Uses **Ultralytics YOLO v11** for object detection.
- Adjust `imgsz`, `epochs`, and `conf` thresholds for optimal results.
- Webcam device may vary (`usb0`, `0`, etc.) depending on your system.
- Streamlit app allows interactive, real-time detection in a browser.
- Model and training results are stored in `my_model/`.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🔗 References

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

