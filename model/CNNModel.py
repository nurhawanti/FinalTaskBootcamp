import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.data_utils import image_preloader
import dataset as ds
import numpy as np
import cv2
import os

width = 64
height = 128
channel = 3
num_class = 2
dataset_file = 'dataset_orang.npz'
dir_dataset = 'D:\Ragil\Bootcamp QTI\FInalTaskBootcamp\dataset'
epoch = 1000
normalized = True

print('Load Dataset...\n')
if os.path.isfile(dataset_file):
    dsw = np.load(dataset_file)
    X = dsw['X']
    Ybin = dsw['Ybin']
    Y = dsw['Y']
    imgns = dsw['imgns']
    cls = dsw['cls']
else:
    X, Y, Ybin, imgns, cls = ds.load_data(dir_dataset,(64,128), channel, normalized)
    np.savez(dataset_file, X=X, Y=Y, Ybin=Ybin, imgns=imgns, cls=cls)

# Convolutional network building
network = input_data(shape=[None, height, width, channel])

network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)

network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)

network = fully_connected(network, 128, activation='relu')
network = dropout(network, 0.8)

network = fully_connected(network, 256, activation='relu')
network = dropout(network, 0.8)

network = fully_connected(network, num_class, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.001,
                     loss='categorical_crossentropy', name='target')

holdout_idx = ds.get_holdout_idx(Y,0.8,True)
Xtrain = X[holdout_idx==1]
Ytrain = Ybin[holdout_idx==1]
Xtest = X[holdout_idx==2]
Ytest = Ybin[holdout_idx==2]
            
# Train using classifier
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit(Xtrain, Ytrain, n_epoch=epoch, shuffle=True, validation_set=(Xtest, Ytest),
            show_metric=True, batch_size=None, run_id='classification')

model.save("model.tflearn")
            
print("Done.")
