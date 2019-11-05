import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used
import sys
import numpy as np
from client.motor_classifier import MotorbikeClassifier

CLASSIFIER_MODEL_PATH = '/media/HDD/SangTV/Zalo_AI_challenge/evaluation_script/client/motorbike_classification_inception_net_128_v4_e36.pb'


class GetClassifyScore():
    def __init__(self, model_path=CLASSIFIER_MODEL_PATH,
                 img_shape=(128, 128, 3),
                 output_shape=2048, 
                 gpu_id=-1,
                 mem_fraction=0.2):
        self.img_shape = img_shape
        self.img_shape = img_shape[0]
        self.output_shape = output_shape
        self.img_size = img_shape[0]
        self.model = MotorbikeClassifier(model_path, gpu_id, mem_fraction, self.img_size)


    def preprocessing(self, np_arr):
        '''Preprocessing input of motorbike classifier'''
        np_arr = np_arr.astype(np.float)
        np_arr /= 255.0
        return np_arr


    def get_scores(self, images, batch_size=16):
        n_images = images.shape[0]
        n_batches = np.ceil(n_images/batch_size).astype(int)
        feature_arr = np.empty((n_images, self.output_shape))
        score_arr = np.empty((n_images, 1))

        for i in range(n_batches):
            start = i*batch_size
            if start + batch_size < n_images:
                end = start + batch_size
            else:
                end = n_images
            batch = images[start:end]
            #batch = self.preprocessing(batch) # because we are working with Pytorch tensor
            scores, features = self.model.predict(batch)
            feature_arr[start:end] = features
            score_arr[start:end] = scores

        return score_arr, feature_arr


