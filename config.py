DIM = 64
IMAGE_SIZE = 128
OUTPUT_DIM = 128*128*3 # Number of pixels in each image

#RANDOM_SEED = 123
TRN_DATA_DIR = '../training_dataset/trainingset/'
VAL_DATA_DIR = '../training_dataset/validationset'

NOISE_DIM = 128
RESTORE_MODE = False # if True, it will load saved model from OUT_PATH and continue to train
START_ITER = 0 # starting iteration 
OUTPUT_PATH = 'checkpoints' # output path where result (.e.g drawing images, cost, chart) will be stored
CRITIC_ITERS = 5 # How many iterations to train the critic for
GENER_ITERS = 1
BATCH_SIZE = 4# Batch size. Must be a multiple of N_GPUS
END_ITER = 100000 # How many iterations to train for
LAMBDA = 10 # Gradient penalty lambda hyperparameter

LR = 1e-5