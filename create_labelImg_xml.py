import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import sys
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from shutil import rmtree

from process_data_coop import Process_data

Processor = Process_data()

Save_dir = 'detector_xml_1104'
if os.path.exists(Save_dir):
    rmtree(Save_dir)
os.makedirs(Save_dir)

Img_dir = '/media/HDD/SangTV/Zalo_AI_challenge/training_dataset/training_data/motobike_png/'
    

def create_xml_file(img_path, save_dir=Save_dir):
    boxes,_ = Processor.detect(img_path)
    #print(boxes)
    img = cv2.imread(img_path)
    h, w, d = img.shape
    filename = img_path.split('/')[-1]
    folder = img_path.replace(filename, '')
    prefix_name = filename.split('.')
    prefix_name = '_'.join(prefix_name[:-1])

    # write xml file
    annotation = ET.Element('annotation')
    folder_tag = ET.SubElement(annotation, 'folder')
    folder_tag.text = folder
    filename_tag = ET.SubElement(annotation, 'filename')
    filename_tag.text = filename
    path_tag = ET.SubElement(annotation, 'path')
    path_tag.text = img_path

    size = ET.SubElement(annotation, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(w)
    height = ET.SubElement(size, 'height')
    height.text = str(h)
    depth = ET.SubElement(size, 'depth')
    depth.text = str(d)

    for y1, x1, y2, x2 in boxes:
        object = ET.SubElement(annotation, 'object')
        name = ET.SubElement(object, 'name')
        name.text = 'motorbike'
        pose = ET.SubElement(object, 'pose')
        pose.text = 'Unspecified'
        truncated = ET.SubElement(object, 'truncated')
        truncated.text = '0'
        difficult = ET.SubElement(object, 'difficult')
        difficult.text = '0'
        bndbox = ET.SubElement(object, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(x1)
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(y1)
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(x2)
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(y2)

    detected_boxes = ET.tostring(annotation)
    with open(os.path.join(save_dir, '{}.xml'.format(prefix_name)), 'wb') as f:
        f.write(detected_boxes)

for idx, filename in enumerate(os.listdir(Img_dir)):
    print(idx + 1, end='\r')
    img_path = os.path.join(Img_dir, filename)
    create_xml_file(img_path)
