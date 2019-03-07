import numpy as np
import cv2
import time
from desktopmagic.screengrab_win32 import getRectAsImage
from getkeys import key_check
import os
import mss
from ThreadWorker import ThreadWorker


w = [1,0,0,0,0,0,0,0,0]
s = [0,1,0,0,0,0,0,0,0]
a = [0,0,1,0,0,0,0,0,0]
d = [0,0,0,1,0,0,0,0,0]
wa = [0,0,0,0,1,0,0,0,0]
wd = [0,0,0,0,0,1,0,0,0]
sa = [0,0,0,0,0,0,1,0,0]
sd = [0,0,0,0,0,0,0,1,0]
nk = [0,0,0,0,0,0,0,0,1]




def captureImage(bbox = None):
    with mss.mss() as sct:
        return sct.grab(bbox)
    #return getRectAsImage(bbox)

def keys_to_output(keys):
    '''
    Convert keys to a ...multi-hot... array
     0  1  2  3  4   5   6   7    8
    [W, S, A, D, WA, WD, SA, SD, NOKEY] boolean values.
    '''
    output = [0,0,0,0,0,0,0,0,0]

    if 'W' in keys and 'A' in keys:
        output = wa
    elif 'W' in keys and 'D' in keys:
        output = wd
    elif 'S' in keys and 'A' in keys:
        output = sa
    elif 'S' in keys and 'D' in keys:
        output = sd
    elif 'W' in keys:
        output = w
    elif 'S' in keys:
        output = s
    elif 'A' in keys:
        output = a
    elif 'D' in keys:
        output = d
    else:
        output = nk
    return output



def convertOpenCV(rawImg):
    img_np = np.array(rawImg)
    #img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    #del img_np
    del rawImg
    return img_np


def main(file_name, starting_value):
    worker = ThreadWorker(starting_value, file_name)
    worker.startThread()
    training_data = []
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    last_time = time.time()
    paused = False
    print('STARTING!!!')
    cv2.namedWindow('Screen', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
    while(True):
        
        if not paused:
            screen = captureImage((0,0,1024,576))
            keys = key_check()
            screen = convertOpenCV(screen)
            
            # resize to something a bit more acceptable for a CNN
            screen = cv2.resize(screen, (480,270), interpolation=4)
            # run a color convert:
            screen = cv2.cvtColor(screen, cv2.COLOR_RGBA2RGB)
            output = keys_to_output(keys)
            if not (worker.appendData(screen, output)):
                break
            
            #print('loop took {} seconds'.format(time.time()-last_time))
            cv2.imshow('Screen',screen)
            cv2.waitKey(1)

                    
        keys = key_check()
        if 'R' in keys:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)

if __name__ == "__main__":
    starting_value = 1
    while True:
        file_name = 'training_data-{}.npy'.format(starting_value)

        if os.path.isfile(file_name):
            print('File exists, moving along',starting_value)
            starting_value += 1
        else:
            print('File does not exist, starting fresh!',starting_value)
            
            break
    main(file_name, starting_value)
