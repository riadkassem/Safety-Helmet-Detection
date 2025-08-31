from pathlib import Path
import random
import os
import sys
import shutil
import argparse
import xml.etree.ElementTree as ET
from PIL import Image

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--datapath', help='Path to data folder containing image and annotation files',
                    required=True)
parser.add_argument('--train_pct', type=float, default=0.7)
parser.add_argument('--val_pct', type=float, default=0.15)
parser.add_argument('--test_pct', type=float, default=0.15)
args = parser.parse_args()

data_path = args.datapath
train_percent, val_percent, test_percent = args.train_pct, args.val_pct, args.test_pct

# Check that splits sum to 1
if not os.path.isdir(data_path):
    print('Directory specified by --datapath not found.')
    sys.exit(0)
if abs((train_percent + val_percent + test_percent) - 1.0) > 1e-6:
    print('Train, val and test percentages must sum to 1.')
    sys.exit(0)

# Input dataset structure
input_image_path = os.path.join(data_path, 'images')
input_label_path = os.path.join(data_path, 'labels')

# Output paths
cwd = os.getcwd()
train_img_path = os.path.join(cwd, 'data/train/images')
train_lbl_path = os.path.join(cwd, 'data/train/labels')
val_img_path   = os.path.join(cwd, 'data/validation/images')
val_lbl_path   = os.path.join(cwd, 'data/validation/labels')
test_img_path  = os.path.join(cwd, 'data/test/images')
test_lbl_path  = os.path.join(cwd, 'data/test/labels')

for dir_path in [train_img_path, train_lbl_path,
                 val_img_path, val_lbl_path,
                 test_img_path, test_lbl_path]:
    os.makedirs(dir_path, exist_ok=True)

# VOC → YOLO converter
def convert_voc_to_yolo(xml_file, img_file, out_file, class_mapping):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    img = Image.open(img_file)
    img_w, img_h = img.size

    with open(out_file, "w") as f:
        for obj in root.findall("object"):
            cls = obj.find("name").text
            if cls not in class_mapping:
                continue
            cls_id = class_mapping[cls]

            xmlbox = obj.find("bndbox")
            xmin = int(xmlbox.find("xmin").text)
            ymin = int(xmlbox.find("ymin").text)
            xmax = int(xmlbox.find("xmax").text)
            ymax = int(xmlbox.find("ymax").text)

            x_center = ((xmin + xmax) / 2) / img_w
            y_center = ((ymin + ymax) / 2) / img_h
            w = (xmax - xmin) / img_w
            h = (ymax - ymin) / img_h

            f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

# Define your classes here (edit as needed!)
class_mapping = {
    "person": 0,
    "helmet": 1,
    "head": 2,
    # add all your classes
}

# Get all images
img_file_list = [path for path in Path(input_image_path).rglob('*') if path.suffix.lower() in [".jpg", ".png", ".jpeg"]]
file_num = len(img_file_list)
train_num = int(file_num * train_percent)
val_num = int(file_num * val_percent)
test_num = file_num - train_num - val_num

print(f"Total images: {file_num}")
print(f"Train: {train_num}, Val: {val_num}, Test: {test_num}")

# Split and convert
for i, set_num in enumerate([train_num, val_num, test_num]):
    for ii in range(set_num):
        img_path = random.choice(img_file_list)
        img_fn = img_path.name
        base_fn = img_path.stem
        xml_fn = base_fn + ".xml"
        xml_path = os.path.join(input_label_path, xml_fn)

        if i == 0:
            new_img_path, new_lbl_path = train_img_path, train_lbl_path
        elif i == 1:
            new_img_path, new_lbl_path = val_img_path, val_lbl_path
        else:
            new_img_path, new_lbl_path = test_img_path, test_lbl_path

        # copy image
        shutil.copy(img_path, os.path.join(new_img_path, img_fn))

        # convert XML to YOLO if it exists
        if os.path.exists(xml_path):
            yolo_label_out = os.path.join(new_lbl_path, base_fn + ".txt")
            convert_voc_to_yolo(xml_path, img_path, yolo_label_out, class_mapping)

        img_file_list.remove(img_path)

import yaml

data_yaml = {
    'train': os.path.abspath(train_img_path),
    'val': os.path.abspath(val_img_path),
    'test': os.path.abspath(test_img_path),
    'nc': len(class_mapping),
    'names': [name for name, _ in sorted(class_mapping.items(), key=lambda x: x[1])]
}

with open("data.yaml", "w") as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

print("✅ data.yaml created!")
