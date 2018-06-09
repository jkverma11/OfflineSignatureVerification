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
from keras.optimizers import SGD
from keras.regularizers import l2
from sklearn import metrics

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
    import scipy.misc  as mc
     
    return mc.imresize(im,(ROWS,COLS))


TEST_DIR= os.getcwd() + '/data/test/'
TRAIN_DIR= os.getcwd() + '/data/train/'

#SIGNATURE_CLASSES = ['A', 'B', 'C','D','E','F','G','H','T','U','Y','Z']

SIGNATURE_CLASSES = []
for x in range(1,138):
    SIGNATURE_CLASSES.append(str(x))

#print (SIGNATURE_CLASSES)

ROWS = 256
COLS = 128
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


train_x, valid_x, train_y, valid_y = train_test_split(X_all, y_all, 
                                                    test_size=0.2, random_state=23, 
                                                    stratify=y_all)

train_x, test_x, train_y, test_y = train_test_split(train_x, train_y,
                                         test_size=0.2, random_state=20, stratify=train_y)

# to run this code, you'll need to load the following data: 
# train_x, train_y
# valid_x, valid_y
# test_x, test_y
# see http://aqibsaeed.github.io/2016-09-24-urban-sound-classification-part-2/ for details

# data dimension parameters 
frames = 128
bands = 256
num_channels = 3
num_labels = test_y.shape[1]

print ("Num Lables:" + str(num_labels))

# this model implements the 5 layer CNN described in https://arxiv.org/pdf/1608.04363.pdf
# be aware, there are 2 main differences: 
# the input is 60x41 data frames with 2 channels => (60,41,2) tensors 
# the paper seems to report using 128x128 data frames (with no mention of channels)
# the paper also uses a receptive field size of 5x5 - as our input is smaller, I'm using 3x3

f_size = 5

model = Sequential()

# Layer 1 - 24 filters with a receptive field of (f,f), i.e. W has the shape (24,1,f,f). 
# This is followed by (4,2) max-pooling over the last two dimensions and a ReLU activation function.
model.add(Conv2D(96, f_size, f_size, border_mode='same', input_shape=(bands, frames, num_channels)))
model.add(MaxPooling2D(pool_size=(4, 2)))
model.add(Activation('relu'))

# Layer 2 - 48 filters with a receptive field of (f,f), i.e. W has the shape (48,24,f,f). 
# Like L1 this is followed by (4,2) max-pooling and a ReLU activation function.
model.add(Conv2D(192, f_size, f_size, border_mode='same'))
model.add(MaxPooling2D(pool_size=(4, 2)))
model.add(Activation('relu'))

# Layer 3 - 48 filters with a receptive field of (f,f), i.e. W has the shape (48, 48, f, f). 
# This is followed by a ReLU but no pooling.
model.add(Conv2D(192, f_size, f_size, border_mode='valid'))
model.add(Activation('relu'))

# flatten output into a single dimension, let Keras do shape inference
model.add(Flatten())

# Layer 4 - a fully connected NN layer of 64 hidden units, L2 penalty of 0.001
model.add(Dense(256, W_regularizer=l2(0.001)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Layer 5 - an output layer with one output unit per class, with L2 penalty, 
# followed by a softmax activation function
model.add(Dense(num_labels, W_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Activation('softmax'))


# create a SGD optimiser
sgd = SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)

# create adam optimiser
adam = Adam(lr=0.0001)

# a stopping function should the validation loss stop improving
earlystop = EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='auto')

# compile and fit model, reduce epochs if you want a result faster
# the validation set is used to identify parameter settings (epoch) that achieves 
# the highest classification accuracy
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=sgd)

#model.compile(loss=root_mean_squared_error, metrics=['accuracy'], optimizer=adam)

model.fit(train_x, train_y, validation_data=(valid_x, valid_y), callbacks=[earlystop], batch_size=96, nb_epoch=5)

# finally, evaluate the model using the withheld test dataset

# determine the ROC AUC score 
y_prob = model.predict_proba(test_x, verbose=0)
y_pred = np_utils.probas_to_classes(y_prob)
y_true = np.argmax(test_y, 1)
roc = metrics.roc_auc_score(test_y, y_prob)
print ("ROC:", round(roc,3))

# determine the classification accuracy
score, accuracy = model.evaluate(test_x, test_y, batch_size=96)
print("\nAccuracy = {:.2f}".format(accuracy))