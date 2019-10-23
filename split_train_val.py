from shutil import rmtree, copy2
import os
import random

random.seed(123)
Train_ratio = 0.9 # 90% images will be in Training set

All_img_dir = '../training_dataset/training_data/motobike_png'
Trn_img_dir = '../training_dataset/trainingset/motobike_png'
Val_img_dir = '../training_dataset/validationset/motobike_png'

def reset_dir(dir_path):
    if os.path.exists(dir_path):
        rmtree(dir_path)

    os.makedirs(dir_path)

reset_dir(Trn_img_dir)
reset_dir(Val_img_dir)


all_files = os.listdir(All_img_dir)
random.shuffle(all_files)

NUM_TRAIN_IMGS = int(len(all_files) * Train_ratio)
train_list = all_files[:NUM_TRAIN_IMGS]
val_list = all_files[NUM_TRAIN_IMGS:]

for filename in train_list:
    copy2(os.path.join(All_img_dir, filename), os.path.join(Trn_img_dir, filename))

for filename in val_list:
    copy2(os.path.join(All_img_dir, filename), os.path.join(Val_img_dir, filename))
