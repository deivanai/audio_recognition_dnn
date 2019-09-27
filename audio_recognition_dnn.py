#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
from pathlib import Path
import IPython.display as ipd

import numpy as np 
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
get_ipython().run_line_magic('matplotlib', 'inline')
import re
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# In[3]:


# read each wav file convert into a spectogram and flatten it as x
# read and the corresponding label and convert into one hot encoding as Y
# Baseline : Start with DNN classifier with  two hidden layesr and drop out as regularization at each step
# output layer  10 neurons for classification at output.
# Once the wave forms are converted, its akin to image comparison 
# MLP Model can be like a baseline.
# we will have to compare against performance using  CNNs , and also AUTOENCODERS.

#TODO - Use confusion matrix and validate how confused are the classification.
#TODO - remove any outliers in data input. 

AUDIO_PATH = 'Documents/data_speech_commands_v0.02/'
LABELS = ['eight', 'five', 'four', 'one',  'seven', 'six', 'three', 'two', 'zero', 'nine']


# In[4]:


#Original sound wav - we cant use as its highly random wav and difficult to identify pattern
#Many options - convert to spectogram / or more advanced : Mel-frequency cepstral coefficients (MFCCs) 
#current implementation is for spectogram .

#convert into spectograms : look at how the actual wav form it is 
def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)

sample_rate1, samples1 = wavfile.read(str(AUDIO_PATH) + '/zero/0a2b400e_nohash_0.wav')
sample_rate2, samples2 = wavfile.read(str(AUDIO_PATH) + '/one/0a2b400e_nohash_0.wav')
sample_rate3, samples3 = wavfile.read(str(AUDIO_PATH) + '/two/0a2b400e_nohash_0.wav')
frequencies1, times1, spectogram1 = log_specgram(samples1, sample_rate1)
frequencies2, times2, spectogram2 = log_specgram(samples2, sample_rate2)
frequencies3, times3, spectogram3 = log_specgram(samples3, sample_rate3)


# In[5]:


fig = plt.figure(figsize=(14, 8))
ax1 = fig.add_subplot(211)
ax1.imshow(spectogram1.T, aspect='auto', origin='lower', 
           extent=[times1.min(), times1.max(), frequencies1.min(), frequencies1.max()])
ax1.set_yticks(frequencies1[::16])
ax1.set_xticks(times1[::16])
ax1.set_title('Spectrogram of zero')
ax1.set_ylabel('Freqs in Hz')
ax1.set_xlabel('Seconds')

ax2 = fig.add_subplot(212)
ax2.imshow(spectogram2.T, aspect='auto', origin='lower', 
           extent=[times2.min(), times2.max(), frequencies2.min(), frequencies2.max()])
ax2.set_yticks(frequencies2[::16])
ax2.set_xticks(times2[::16])
ax2.set_title('Spectrogram of one')
ax2.set_ylabel('Freqs in Hz')
ax2.set_xlabel('Seconds')

#OBSERVATION : 
#Can see that half the time series is with silence and the clippings could be clipped for dimension reduction
#


# In[6]:


# reading files with only (0 -9) for simplicity, 
# lets neglect other wav files .
# TODO  add more classification such as silence, unknown for other input waves.
# DATASET USED 
# https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data
#data set downloaded locally  under relative path /Documents/data_speech_commands_vo.02


# for each label , read the training and validation files and corresponding target label 
def create_list_of_validation_files():

    fp = open(AUDIO_PATH + "validation_list.txt", 'r')
    line = fp.readline()
    cnt = 1
    validation_file_list =[]
    while line:
       validation_file_list.append(line.strip())
       line = fp.readline()
    return validation_file_list


#CREATE INPUT TRAIN AND VALIDATION DATA SET
#TODO - DIMENSION REDUCTION BY CLIPPING SILENCE( can see more shorter clips)
#TODO - DIMENSION REDUCTION BY Resampling 
#TODO - DATA AUGMENTATION BY ADDING NOISE AND CREATE NEW SOUND CLIPS 

def create_input(validation_file_list):

    filecount = 0
    wavefiles = []
    #using lists instead of numpy array for faster reading while loop 
    X_train = []
    X_validation = []
    y_train= []
    y_validation = []
    for  label in LABELS:
        #read each file under train_label and convert it into histogram assign to first ,
        #assign label 
        wavefile = glob.glob(AUDIO_PATH  + label + "/*.wav")
        for wav in wavefile:
            sample_rate, samples = wavfile.read(wav)
            frequencies, times, spectogram = log_specgram(samples, sample_rate)
            filename = re.split(AUDIO_PATH , wav)
   
            if(len(times)==99): 
                #currently reading only similar length file
                #TODO IMPROVE BY PADDING IF FILE LENGTH IS LESS OR CLIP IF MORE 
                try:
                    if(filename[1] in validation_file_list): 
                        X_validation.append(np.array(spectogram))
                        y_validation.append(label)
                    else:
                        X_train.append(np.array(spectogram))
                        y_train.append(label)
                except:
                     print(len(times))
                     pass 
    return X_train, y_train,X_validation, y_validation




# In[7]:


validation_list = create_list_of_validation_files()
# get all the files that needs to be in validation test data set, so we can correctly split the wav forms


# In[8]:


X_train,y_train,X_validation, y_validation = create_input(validation_list)
print(len(X_train), len(y_train), len(X_validation), len(y_validation))


# In[9]:


X_Train = np.asarray(X_train)
y_Train = np.asarray(y_train)
X_Validation = np.asarray(X_validation)
y_Validation = np.asarray(y_validation)


# FIRST LABEL ENCODE AND ONE HOT ENCODE FOR CATEGORICAL OUTPUT TO FEED TO NEURAL NW

label_encoder = LabelEncoder()
label_encoder.fit(y_Train)
y_Train = label_encoder.transform(y_Train.reshape(-1,1))
y_Validation = label_encoder.transform(y_Validation.reshape(-1,1))
onehot_encoder = OneHotEncoder(sparse=False)
y_Train = onehot_encoder.fit_transform(y_Train.reshape(-1,1))
y_Validation = onehot_encoder.transform(y_Validation.reshape(-1,1))


# In[10]:


print(X_Train.shape, X_Validation.shape)


# In[11]:


#FLATTEN INPUT ARRAYS
INPUT_SHAPE = 99* 161
X_Train = X_Train.reshape(32290, INPUT_SHAPE)
X_Validation = X_Validation.reshape(3338,INPUT_SHAPE)


# In[12]:


#normalize inputs, we take log of values to have much reasonable scale between frequencies
# check for other custom built libraries which could do logarithmic transform to give more weightage to frequency range of interest

#X_Train = np.log(X_Train+ 1e-10) # add small error to avoid  infinity. 
#print(np.min(X_Train), np.max(X_Train))
print(np.min(X_Train), np.max(X_Train))
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_Train = scaler.fit_transform(X_Train)
X_Validation = scaler.transform(X_Validation)
print(np.min(X_Train), np.max(X_Train))
print(np.min(X_Validation), np.max(X_Validation))


# In[15]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

EPOCHS = 10 # ideally need to run for longer epoch 
BATCHSIZE = 32

model = Sequential()
model.add(Dense(8192,  kernel_initializer='he_normal',activation='relu', input_shape=(INPUT_SHAPE,)))
model.add(Dropout(0.2))
model.add(Dense(10, kernel_initializer='he_normal', activation='softmax')) 
#SINCE WE WANT OUTPUT TO BE PROBABILITY OF PREDICTION TO PARTICULAR CLASS, use softmax activation at output
model.summary()

model.compile(loss='categorical_crossentropy', # ITS CLASSIFICATION PROBLEM 
              optimizer='adam', # CAN VARY OPTIMIZER for baseline , adam is selected.  
              metrics=['accuracy']) # TODO check for confidence ( probablity of top three prediction class difference etc) 

history = model.fit(X_Train, y_Train,
                    batch_size=BATCHSIZE,
                    epochs=EPOCHS,
                    verbose=1,
                    shuffle=True,
                    validation_data=(X_Validation, y_Validation)) 


# In[16]:


plt.plot(history.history['acc'], label='Train Accuracy')
plt.plot(history.history['val_acc'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()


# In[17]:


plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


# In[18]:


y_predicted = model.predict(X_Validation)
print(y_predicted.shape)


# In[19]:


print(y_predicted[0])


# In[20]:


print(y_Validation[0])


# In[ ]:




