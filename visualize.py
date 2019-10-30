import cv2
import os
import numpy as np
from sklearn.decomposition import PCA
pca = PCA(n_components=2)

SAVE_PATH = 'distribution.png'

IMG_SIZE = 4096         # Output image size is 2048x2048
THUMBNAIL_SIZE= 64 # Each image will be resized to this size before pasting to output image
HALF_SIZE = int(THUMBNAIL_SIZE / 2)
HALP_IMG_SIZE = int(IMG_SIZE / 2)

OUT_IMG = np.zeros((IMG_SIZE, IMG_SIZE, 3))

VALID_BOUND_VALUE = int((IMG_SIZE - THUMBNAIL_SIZE) / 2)

PATH_IMG_DIR = 'generated_10kimages'
PATH_TO_NOISE_VEC = 'noise.npy'

Listfile = os.listdir(PATH_IMG_DIR)
Listfile.sort()

Noise_vecs = np.load(PATH_TO_NOISE_VEC)

Pos_2d = pca.fit_transform(Noise_vecs)

Largest_dim = np.max(np.fabs(Pos_2d))

def convertToImgCoor(in_val):
    return int(in_val * VALID_BOUND_VALUE / Largest_dim) + HALP_IMG_SIZE

for i in range(Pos_2d.shape[0]):
    x = convertToImgCoor(Pos_2d[i][0])
    y = convertToImgCoor(Pos_2d[i][1])

    img = cv2.resize(cv2.imread(os.path.join(PATH_IMG_DIR, Listfile[i])),
                        (THUMBNAIL_SIZE, THUMBNAIL_SIZE))

    OUT_IMG[y - HALF_SIZE:y + HALF_SIZE, x - HALF_SIZE:x + HALF_SIZE, :] = img.copy()

cv2.imwrite(SAVE_PATH, OUT_IMG)
