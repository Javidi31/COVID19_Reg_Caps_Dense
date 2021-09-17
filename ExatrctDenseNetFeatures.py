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
#LOAD All Data as inputs
#inputs_data = inputs



# Test pretrained model
#weights_path = '.\\COVID19_models\\model-98-0.9195_KFOLD_COVIDCT_DENSE169_Adam.h5' #model-40-0.9998.h5'
#model = DenseNet(img_rows=img_rows, img_cols=img_cols, reduction=0.5, classes=num_classes, weights_path=weights_path)
##model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
#model.summary()

#sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
#model.summary()




#then use it to visualize
layer_outputs    = [layer.output for layer in model.layers[1:layer_no]] # Extracts the outputs of the top 12 layers
activation_model = models.Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
activation_model.summary()


maxSampleNo = inputs_data.shape[0]
for sampleNo in range(0,maxSampleNo):  
    sample = inputs_data[sampleNo,:,:,0]
    img_tensor = image.img_to_array(sample)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    #sample_feat = TrDB[1,:]
    #img_tensor_feat = np.expand_dims(sample_feat, axis=0)
    #activations = activation_model.predict([img_tensor,img_tensor_feat])
    activations = activation_model.predict(img_tensor)  
    
    #number = 15                                 ##############################
    number = 72                                  ##############################
    layer_activation = activations[number]
    #print(layer_activation.shape)
    out=layer_activation[0, :, :, :]
    #out     = out.astype('float16')
    out     = out.astype('float32')
#    out=layer_activation[0, :, :, 14]
#    out=out-out.min()
#    out=out*128
#    plt.matshow(out, cmap='viridis')
    f = open('.\\Features_COVIDCT_KFOLD_Lay72_Float32_Adam\\'+str(sampleNo)+'Out_lay72.pckl', 'wb')#######
    #f = open('.\\Features_Prot2_Lay17\\'+str(sampleNo)+'Out_lay17.pckl', 'wb')#######
    pickle.dump( out, f)
    f.close()
