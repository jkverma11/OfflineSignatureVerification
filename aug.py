import  keras
import os, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from matplotlib import ticker
from sklearn.model_selection import  train_test_split
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D, Dense, Activation
from keras.optimizers import SGD, Adagrad
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.optimizers import RMSprop, Adam
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot

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


TEST_DIR= os.getcwd() + '/data/test/'
TRAIN_DIR= os.getcwd() + '/data/train3/'

SIGNATURE_CLASSES = []
for x in range(1,10):
    SIGNATURE_CLASSES.append(str(x))

ROWS = 128
COLS = 64
CHANNELS = 3

files = []
y_all = []

for fish in SIGNATURE_CLASSES:
    fish_files = get_images(fish)
    files.extend(fish_files)
    
    y_fish = np.tile(fish, len(fish_files))
    y_all.extend(y_fish)
    print("{0} photos of {1}".format(len(fish_files), fish))
    
y_all = np.array(y_all)
print(len(files))
print(len(y_all))

X_all = np.ndarray((len(files), ROWS, COLS, CHANNELS), dtype=np.uint8)  

for i, im in enumerate(files): 
    X_all[i] = read_image(TRAIN_DIR+im)
    if i%1000 == 0: print('Processed {} of {}'.format(i, len(files)))

print(X_all.shape)
# One Hot Encoding Labels
y_all = LabelEncoder().fit_transform(y_all)
y_all = np_utils.to_categorical(y_all)

X_train, X_valid, y_train, y_valid = train_test_split(X_all, y_all, 
                                                    test_size=0.2, random_state=23, 
                                                    stratify=y_all)



X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,
                                         test_size=0.2, random_state=20, stratify=y_train)

# define data preparation
datagen = ImageDataGenerator(
    rotation_range=20,
    height_shift_range=0.2,
    width_shift_range=0.1)
# fit parameters from data


train_generator = datagen.flow_from_directory(
    os.getcwd() + '/data/train4/',
    target_size=(64, 128),
    batch_size=100,
    save_to_dir=os.getcwd() + '/data/train4/20/',
    class_mode='binary',
    save_prefix='aug',
    save_format='png')

#print (train_generator)

model = Sequential()

model.add(Activation(activation=center_normalize, input_shape=(ROWS, COLS, CHANNELS)))

# Layer 1
model.add(Conv2D(64, (3, 3), border_mode='same'))
model.add(Activation('relu'))

adam = Adam(lr=0.0001)
model.compile(optimizer=adam, loss=root_mean_squared_error, metrics=['accuracy'])

model.fit_generator(
        train_generator,
        steps_per_epoch=30,
        epochs=5)




