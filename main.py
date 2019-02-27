from grabscreen import grab_screen, captureImage
from multiprocessing import Lock, Process, Queue, Value
import numpy as np
from timeit import default_timer as timer
import cv2
from mss import mss
from desktopmagic.screengrab_win32 import (
getDisplayRects, saveScreenToBmp, saveRectToBmp, getScreenAsImage,
getRectAsImage, getDisplaysAsImages)

def convertOpenCV(rawImg):
    img_np = np.array(rawImg)
    img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    return img

class ThreadWorker:
    def __init__(self):
        self.queue = Queue(3)
        self.running = Value('d', True)
        self.windowName = 'Screen'
        self.thread = Process(target=self.worker)

    def startThread(self):
        self.thread.start()

    def join(self):
        self.thread.join()

    def worker(self):
        cv2.namedWindow('Screen', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
        while True:
            t = timer()		
            rawImg = self.queue.get(True, 10)
            img = convertOpenCV(rawImg)
            #img = detectEdges(img, cv2.cv2.COLOR_BGR2GRAY, cv2.cv2.COLOR_GRAY2RGB)
            cv2.imshow(self.windowName, img)

            ### Keyboard Input ###

            key = cv2.waitKey(1)
            windowProperty = cv2.getWindowProperty(self.windowName, 0)
            if key == 27 or windowProperty < 0:
                self.running.value = False
                break
            print('FPS: ', (1 / (timer() - t))) 

    def appendImage(self, img):
        if not self.running.value:
            return False
        try:
            self.queue.put(img, True, 10)
        except Exception:
            return False
        return True
	


if __name__ == '__main__':
 while True:
     tw = ThreadWorker()
     tw.startThread()
     while True:
        screen = getRectAsImage((0,0,800,600))
        val = tw.appendImage(screen)
        if not val:
            break
        mss().close()
     break
	#cv2.imshow('screen', convertOpenCV(screen))
	

	
