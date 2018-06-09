from keras import backend as K
import os

TEST_DIR= os.getcwd() + '/data/test/'
TRAIN_DIR= os.getcwd() + '/data/train5/'

ROWS = 256
COLS = 128
CHANNELS = 3

def root_mean_squared_error(y_true, y_pred):
    """
    RMSE loss function
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def get_images(fish):
    """Load files from train folder"""
    fish_dir = TRAIN_DIR+'{}'.format(fish)
    images = [fish+'/'+im for im in os.listdir(fish_dir)]
    return images

def read_image(src):
    import os
    from scipy import misc
    filepath=src
    im=misc.imread(filepath)
    #return im
    import scipy.misc  as mc
    
    return mc.imresize(im,(ROWS,COLS))

def center_normalize(x):
    return (x - K.mean(x)) / K.std(x)