import numpy as np
from grabscreen import grab_screen
import cv2
import time
import os
import pandas as pd
from tqdm import tqdm
from collections import deque
from models import inception_v3 as googlenet
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Conv3D, MaxPooling3D, AveragePooling3D, Dropout, Dense, Concatenate
from random import shuffle
from keras.models import Model

FILE_I_END = 55

WIDTH = 480
HEIGHT = 270
LR = 1e-3
EPOCHS = 30

MODEL_NAME = 'a'
PREV_MODEL = 'a'

LOAD_MODEL = True

wl = 0
sl = 0
al = 0
dl = 0

wal = 0
wdl = 0
sal = 0
sdl = 0
nkl = 0

w = [1,0,0,0,0,0,0,0,0]
s = [0,1,0,0,0,0,0,0,0]
a = [0,0,1,0,0,0,0,0,0]
d = [0,0,0,1,0,0,0,0,0]
wa = [0,0,0,0,1,0,0,0,0]
wd = [0,0,0,0,0,1,0,0,0]
sa = [0,0,0,0,0,0,1,0,0]
sd = [0,0,0,0,0,0,0,1,0]
nk = [0,0,0,0,0,0,0,0,1]

base_model = googlenet(WIDTH, HEIGHT, 3, LR, output=9, model_name=MODEL_NAME)
# add a global spatial average pooling layer
x = base_model.output
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(9, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
if LOAD_MODEL:
    model.load_weights(PREV_MODEL)
    print('We have loaded a previous model!!!!')

model.compile(optimizer='Adam', loss='categorical_crossentropy')

# iterates through the training files


for e in range(EPOCHS):
    #data_order = [i for i in range(1,FILE_I_END+1)]
    data_order = [i for i in range(1,FILE_I_END+1)]
    shuffle(data_order)
    for count,i in enumerate(data_order):
        
        #try:
            file_name = 'training_data-{}.npy'.format(i)
            # full file info
            train_data = np.load(file_name)
            print('training_data-{}.npy'.format(i),len(train_data))

##            # [   [    [FRAMES], CHOICE   ]    ] 
##            train_data = []
##            current_frames = deque(maxlen=HM_FRAMES)
##            
##            for ds in data:
##                screen, choice = ds
##                gray_screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
##
##
##                current_frames.append(gray_screen)
##                if len(current_frames) == HM_FRAMES:
##                    train_data.append([list(current_frames),choice])


            # #
            # always validating unique data: 
            shuffle(train_data)
            train = train_data
            #test = train_data[-50:]

            X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,3)
            Y = np.array([i[1] for i in train])#.resize(1, 2048)

            #test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,3)
            #test_y = [i[1] for i in test]

            model.fit(X, Y, epochs=2, batch_size=3, validation_split=0.1)#, validation_data=({'input': test_x}, {'targets': test_y}))


            if count%10 == 0:
                print('SAVING MODEL!')
                model.save(MODEL_NAME)
                    
        #except Exception as e:
           # print(str(e))
            
    








#

#tensorboard --logdir=foo:J:/phase10-code/log

