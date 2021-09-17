# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 11:18:46 2021

@author: Malihe Javidi
"""
import numpy as np
from keras.optimizers import SGD
from keras import backend as K
from keras.models import load_model
from keras import models
from keras.preprocessing import image
from matplotlib import pyplot as plt
import pickle 
import keras 
# We only test DenseNet-121 in this script for demo purpose
from densenet169 import DenseNet #For HUST we eused dense121

layer_no           = 80
img_rows, img_cols = 224, 224
num_classes        = 2


#LOAD Data##############################################################################
path1        = '.\\Dataset\\COVID_CT\\'
all_images   = []
targets      = []
subjects     = os.listdir(path1)
num_classes  = len(subjects)     
print('Number of Classes: ', num_classes)

for number1 in range(0, num_classes):
    print(number1)
    path2 = (path1 + subjects[number1] + '\\')
    files = os.listdir(path2);
    numberOfsamples = len(files)
    for number2 in range(0,numberOfsamples):
        path3 = path2 + files[number2]
        img = cv2.imread(path3 , 0)
        img = cv2.resize(img,(img_rows,img_rows))
        img = img.reshape(img_rows, img_cols, 1)
        all_images.append(img)#        
        targets.append(int(number1))

inputs_data     = np.array(all_images)
inputs_data     = inputs_data.astype('float32')

mean_score = np.mean(inputs_data)
std_score  = np.std(inputs_data)
inputs_data    -= mean_score
inputs_data    /= std_score


targets_array = np.array(targets, dtype=np.uint8) 
targets       = np_utils.to_categorical(targets_array, num_classes,dtype=np.uint8)
print(inputs_data.shape)
#End LOAD Data######################################################

#LOAD Model##############################################################################
#First I load the saved model 

weights_path = '.\\DenseNetModels\\COVID_CT\\model-98-0.9195.h5' ########model-98-0.9195_KFOLD_COVIDCT_DENSE169_Adam.h5
model = DenseNet(img_rows=img_rows, img_cols=img_cols, reduction=0.5, classes=num_classes, weights_path=weights_path)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
#End LOAD Model######################################################

#Then use it to visualize
layer_outputs    = [layer.output for layer in model.layers[1:layer_no]] # Extracts the outputs of the top 12 layers
activation_model = models.Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
activation_model.summary()


#Extract DenseNet Features from all inputs_data
CovidNo = 349
maxSampleNo = inputs_data.shape[0]
for sampleNo in range(0,maxSampleNo):  
    sample = inputs_data[sampleNo,:,:,0]
    img_tensor = image.img_to_array(sample)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    activations = activation_model.predict(img_tensor)  
    
    number = 72                                  ##############################
    layer_activation = activations[number]
    out=layer_activation[0, :, :, :]
    out     = out.astype('float32')
    if sampleNo < CovidNo:
        f = open('.\\DenseNetFeatures\\COVID_CT\\COVID\\'+str(sampleNo)+'Out_lay72.pckl', 'wb')#######
    else:
        f = open('.\\DenseNetFeatures\\COVID_CT\\NonCOVID\\'+str(sampleNo)+'Out_lay72.pckl', 'wb')#######
    pickle.dump( out, f)
    f.close()
