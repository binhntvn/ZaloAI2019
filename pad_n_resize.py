import glob
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET



def padding_n_resize(img, size=(128,128)):
    img_size = img.size
    long_dim = max(img_size)
    new_im = Image.new("RGB", (long_dim, long_dim), (255,255,255))
    
    new_im.paste(img, ((long_dim-img_size[0])//2,
                        (long_dim-img_size[1])//2))
    if long_dim < size[0]:
        # upscale
        new_im = new_im.resize(size, Image.BICUBIC)
    else:
        # downscale
        new_im.thumbnail(size, Image.LANCZOS)
    return new_im

folder_imgs_in = '/media/HDD/SangTV/Zalo_AI_challenge/training_dataset/training_data/motobike_png'
folder_xml_in = '/project/gan/detector_xml_1104_merged'
folder_img_out = '/media/HDD/SangTV/Zalo_AI_challenge/training_dataset/training_data/crop_png'

#paths = glob.glob('path_in'.format())
#yourImage.crop((0, 30, w, h-30))
path_xmls = glob.glob('{}/*'.format(folder_xml_in))
for path in path_xmls:
    root = ET.parse(path).getroot()
    filename = root.find('filename').text
    fn = filename.split('.')[0]
    img = Image.open('{}/{}'.format(folder_imgs_in,filename))
    for idx, obj in enumerate(root.findall('object')):
        x1 = int(obj.find('bndbox/xmin').text)
        y1 = int(obj.find('bndbox/ymin').text)
        x2 = int(obj.find('bndbox/xmax').text)
        y2 = int(obj.find('bndbox/ymax').text)
        cropped_img = img.crop((x1, y1, x2, y2))  # left, up, right, bottom
        cropped_img = padding_n_resize(cropped_img)
        cropped_img.save('{}/{}_{}.png'.format(folder_img_out, fn, idx))



