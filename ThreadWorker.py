from multiprocessing import Queue, Value, Process, JoinableQueue
import numpy as np
import gc
import sys
import os

class ThreadWorker:
    def __init__(self, starting_value, file_name):
        self.screenQueue = Queue(500)
        self.keyQueue = Queue(500)
        self.running = Value('d', True)
        self.thread = Process(target=self.worker)
        self.starting_value = starting_value
        self.file_name = file_name

    def startThread(self):
        self.thread.start()

    def join(self):
        self.thread.join()

    def worker(self):
        training_data = []
        while self.running.value:
            screen = self.screenQueue.get(True, 10)
            keys = self.keyQueue.get(True, 10)
            training_data.append([screen, keys])
            if len(training_data) % 100 == 0:
                print(len(training_data))
                if len(training_data) == 500:
                    np.save(self.file_name, training_data)
                    print('SAVED')
                    training_data = []
                    self.starting_value += 1
                    self.file_name = 'training_data-{}.npy'.format(self.starting_value)


    def appendData(self, screen, keys):
        if not self.running.value:
            return False
        try:
            self.screenQueue.put(screen, True, 10)
            self.keyQueue.put(keys, True, 10)
        except Exception:
            return False
        return True

    def stop(self):
        self.running.value = False
