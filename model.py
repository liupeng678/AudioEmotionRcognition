import librosa
import librosa.display
import pandas as pd
import numpy as np
from glob import glob
import os
from tqdm import tqdm_notebook as tqdm
import traceback
import cv2
import sklearn
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as plot
import math
from imgaug import augmenters as iaa
from tensorflow.python.keras.layers import *
from tensorflow.python.keras import Sequential,Model
from tensorflow.keras import optimizers as opts
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.python.keras import regularizers
from keras.utils import np_utils, Sequence
from sklearn.metrics import confusion_matrix


import keras
from classification_models.keras import Classifiers
from efficientnet.keras import EfficientNetB4



shape1 = 128
shape2 = 563



base_model = EfficientNetB4(input_shape=(shape1,shape2,3), weights='imagenet', include_top=False)
x = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(7, activation='softmax')(x)
model = keras.models.Model(inputs=[base_model.input], outputs=[output])

model.summary()