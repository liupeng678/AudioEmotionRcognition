# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np
import pandas as pd
import os
import librosa
import librosa.display
import IPython
from IPython.display import Audio
from IPython.display import Image
import matplotlib.pyplot as plt

EMOTIONS = {1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 0:'surprise'} # surprise je promenjen sa 8 na 0
DATA_PATH = '/home/liupeng/Desktop/paperFour/data/'
SAMPLE_RATE = 48000

y = []

data = pd.DataFrame(columns=['Emotion', 'Emotion intensity', 'Gender','Path'])
for dirname, _, filenames in os.walk(DATA_PATH):
    for filename in filenames:
        file_path = os.path.join('/kaggle/input/',dirname, filename)
        identifiers = filename.split('.')[0].split('-')
        emotion = (int(identifiers[2]))
        if emotion == 8: # promeni surprise sa 8 na 0
            emotion = 0
        if int(identifiers[3]) == 1:
            emotion_intensity = 'normal' 
        else:
            emotion_intensity = 'strong'
        if int(identifiers[6])%2 == 0:
            gender = 'female'
        else:
            gender = 'male'
        y.append(emotion)
        data = data.append({"Emotion": emotion,
                            "Emotion intensity": emotion_intensity,
                            "Gender": gender,
                            "Path": file_path
                             },
                             ignore_index = True
                          )
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session




def getMELspectrogram(audio, sample_rate):
    mel_spec = librosa.feature.melspectrogram(y=audio,
                                              sr=sample_rate,
                                              n_fft=1024,
                                              win_length = 512,
                                              window='hamming',
                                              hop_length = 256,
                                              n_mels=128,
                                              fmax=sample_rate/2
                                             )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

# test function
audio, sample_rate = librosa.load(data.loc[0,'Path'], duration=3, offset=0.5,sr=SAMPLE_RATE)
signal = np.zeros((int(SAMPLE_RATE*3,)))
signal[:len(audio)] = audio
mel_spectrogram = getMELspectrogram(signal, SAMPLE_RATE)
librosa.display.specshow(mel_spectrogram, y_axis='mel', x_axis='time')
print('MEL spectrogram shape: ',mel_spectrogram.shape)



mel_spectrograms = []
signals = []
for i, file_path in enumerate(data.Path):
    audio, sample_rate = librosa.load(file_path, duration=3, offset=0.5, sr=SAMPLE_RATE)
    signal = np.zeros((int(SAMPLE_RATE*3,)))
    signal[:len(audio)] = audio
    signals.append(signal)
    mel_spectrogram = getMELspectrogram(signal, sample_rate=SAMPLE_RATE)
    mel_spectrograms.append(mel_spectrogram)
    print("\r Processed {}/{} files".format(i,len(data)),end='')



def addAWGN(signal, num_bits=16, augmented_num=1, snr_low=15, snr_high=30): 
    signal_len = len(signal)
    # Generate White Gaussian noise
    noise = np.random.normal(size=(augmented_num, signal_len))
    # Normalize signal and noise
    norm_constant = 2.0**(num_bits-1)
    signal_norm = signal / norm_constant
    noise_norm = noise / norm_constant
    # Compute signal and noise power
    s_power = np.sum(signal_norm ** 2) / signal_len
    n_power = np.sum(noise_norm ** 2, axis=1) / signal_len
    # Random SNR: Uniform [15, 30] in dB
    target_snr = np.random.randint(snr_low, snr_high)
    # Compute K (covariance matrix) for each noise 
    K = np.sqrt((s_power / n_power) * 10 ** (- target_snr / 10))
    K = np.ones((signal_len, augmented_num)) * K  
    # Generate noisy signal
    return signal + K.T * noise



for i,signal in enumerate(signals):
    augmented_signals = addAWGN(signal)
    for j in range(augmented_signals.shape[0]):
        mel_spectrogram = getMELspectrogram(augmented_signals[j,:], sample_rate=SAMPLE_RATE)
        mel_spectrograms.append(mel_spectrogram)
        data = data.append(data.iloc[i], ignore_index=True)
        #print(data.iloc[i])
        # print(y[i])
        y.append(y[i])
    print("\r Processed {}/{} files".format(i,len(signals)),end='')


X = np.stack(mel_spectrograms,axis=0)
X = np.expand_dims(X,1)
X = X.swapaxes(1,3)
X = X.swapaxes(1,2)



shape2 = 128
shape1 = 563

X = np.reshape(X,(X.shape[0],shape2,shape1))

from keras.utils import np_utils, Sequence
y = np.array(y)
y = np.expand_dims(y,1)
y = np_utils.to_categorical(y, 8)
print('Shape of data: ',X.shape)
print('Shape of data: ',y.shape)

np.savez("./data1.npz", X=X, y=y)
# np.save("filename.npy",a)
# b = np.load("filename.npy")
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
        X, y, test_size=0.2, random_state=2)







#print(Y_test)

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
from keras.optimizers import *
from keras.losses import categorical_crossentropy
from classification_models.keras import Classifiers
from efficientnet.keras import EfficientNetB4

from keras.layers import   *
from keras import backend as  K

emotion_len = 8
def backend_reshape(x):
    return K.reshape(x, ( 64,281, 128))

inputs = Input(shape= [shape2, shape1])

x  = Conv1D(32, 4, padding="same", activation='relu')(inputs)
x  = Conv1D(64, 4, padding="same", activation='relu')(x)
x  = Conv1D(128, 2, padding="same", activation='relu')(x)
x = MaxPooling1D(pool_size= (3))(x)

# print(x.shape)
# x = Flatten()(x)
# x = Reshape((-1,-1))(x)
x = LSTM(64,dropout=0.2,recurrent_dropout=0.2,return_sequences=False)(x)

x = Dense(128,activation='relu')(x)
x = Dense(emotion_len,activation='softmax')(x)



model = Model(inputs= inputs, outputs= x)


#model.summary()
model.compile(loss=categorical_crossentropy,
                   optimizer=RMSprop(), metrics=['accuracy'])

history = model.fit(X_train, Y_train, batch_size=32, epochs=50,
                        validation_data=(X_test, Y_test), verbose=1, shuffle=True)


loss, acc = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', acc)