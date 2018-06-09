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
from helper import root_mean_squared_error, get_images, read_image, center_normalize

TEST_DIR= os.getcwd() + '/data/test/'
TRAIN_DIR= os.getcwd() + '/data/train5/'

SIGNATURE_CLASSES = []
for x in range(1,21):
    SIGNATURE_CLASSES.append(str(x))

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

X_train, X_valid, y_train, y_valid = train_test_split(X_all, y_all, 
                                                    test_size=0.2, random_state=23, 
                                                    stratify=y_all)

optimizer = RMSprop(lr=1e-4)
objective = 'categorical_crossentropy'

model = Sequential()

model.add(Activation(activation=center_normalize, input_shape=(ROWS, COLS, CHANNELS)))

# Layer 1
model.add(Conv2D(64, (3, 3), border_mode='same'))
model.add(Activation('relu'))

#Layer 2
model.add(Conv2D(64, (3, 3), border_mode='valid'))
model.add(Activation('relu'))
model.add(ZeroPadding2D(padding=(1, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

#Layer 3
model.add(Conv2D(96, (3, 3), border_mode='same'))
model.add(Activation('relu'))

# Layer 4
model.add(Conv2D(96, (3, 3), border_mode='valid'))
model.add(Activation('relu'))
model.add(ZeroPadding2D(padding=(1, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

# Layer 5
model.add(Conv2D(128, (2, 2), border_mode='same'))
model.add(Activation('relu'))

# Layer 6
model.add(Conv2D(128, (2, 2), border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

# Layer 7
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Layer 8
model.add(Dense(len(SIGNATURE_CLASSES)))
model.add(Activation('sigmoid'))


adam = Adam(lr=0.0001)
model.compile(optimizer=adam, loss=root_mean_squared_error, metrics=['accuracy'])


early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto')

model.fit(X_train, y_train, batch_size=64, epochs=10, validation_split=0.1, verbose=1, shuffle=True, callbacks=[early_stopping])

preds = model.predict(X_valid, verbose=1)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=20, stratify=y_train)

print("Validation Log Loss: {}".format(log_loss(y_valid, preds)))
score, acc = model.evaluate(X_test, y_test, batch_size=64)
print('Test score:', score)
print('Test accuracy:', acc)

test_files = [im for im in os.listdir(TEST_DIR)]
test = np.ndarray((len(test_files), ROWS, COLS, CHANNELS), dtype=np.uint8)

for i, im in enumerate(test_files):
    test[i] = read_image(TEST_DIR+im)

test_preds = model.predict(test, verbose=1)
submission = pd.DataFrame(test_preds, columns=SIGNATURE_CLASSES)
submission.insert(0, 'image', test_files)
submission.head()

submission.to_csv(os.getcwd() + '/signatureResults.csv',index=False)