'''
Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

#import tflearn
#from tflearn.layers.conv import conv_2d, max_pool_2d,avg_pool_2d, conv_3d, max_pool_3d, avg_pool_3d
#from tflearn.layers.core import input_data, dropout, fully_connected
#from tflearn.layers.estimator import regression
#from tflearn.layers.normalization import local_response_normalization
#from tflearn.layers.merge_ops import merge
from keras.applications import InceptionV3





def inception_v3(width, height, frame_count, lr, output=9, model_name = 'sentnet_color.model'):
    #network = input_data(shape=[None, width, height,3], name='input')
    return InceptionV3(False, input_shape=(width, height, 3), pooling = 'avg')


    
    #network = regression(loss, optimizer='momentum',
     #                    loss='categorical_crossentropy',
      #                   learning_rate=lr, name='targets')
    
    #model = tflearn.DNN(network,
     #                   max_checkpoints=0, tensorboard_verbose=0,tensorboard_dir='log')


    #return model



