# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 22:58:07 2021

@author: Malihe Javidi
"""



import cv2
import numpy as np
from keras.optimizers import SGD
import keras.backend as K
import keras
import os
import pickle
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.utils import np_utils
from sklearn.model_selection import KFold
from keras.callbacks import ModelCheckpoint
# We only test DenseNet-121 in this script for demo purpose
from densenet169 import DenseNet 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report



lr_reducer    = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=100)
#csv_logger    = CSVLogger('DenseNet169_COVID_KFOLD1.csv')


batch_size    = 8
nb_epoch      = 100
data_augmentation = False
img_rows, img_cols = 224, 224
img_channels  = 1
#validation_split = 0.05
num_folds = 5

#LOAD Train data##############################################################################
path1        = 'D:\\DeepLearning\\COVID19\\Dataset\\My_COVID_CT\p3\\'
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
    for number2 in range(0,numberOfsamples):#(0,train_no):
        path3 = path2 + files[number2]
        img = cv2.imread(path3 , 0)
        img = cv2.resize(img,(img_rows,img_rows))
        img = img.reshape(img_rows, img_cols, 1)
        all_images.append(img)#        
        targets.append(int(number1))

inputs     = np.array(all_images)
inputs     = inputs.astype('float32')
#mean_image = np.mean(inputs, axis=0)
#std_image  = np.std(inputs, axis=0)
#inputs    -= mean_image
#inputs    *= std_image
mean_score = np.mean(inputs)
std_score  = np.std(inputs)
inputs    -= mean_score
inputs    /= std_score


targets_array = np.array(targets, dtype=np.uint8) 
targets       = np_utils.to_categorical(targets_array, num_classes,dtype=np.uint8)
print(inputs.shape)
#End LOAD Train data######################################################


# Define per-fold score containers
acc_per_fold  = []
loss_per_fold = []


# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True,random_state=101)

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(inputs, targets):
    str_logger='DenseNet169_COVID_KFOLD'+str(fold_no)+'.csv'
    csv_logger    = CSVLogger(str_logger)
    
    #model = DenseNet(img_rows=224, img_cols=224, reduction=0.5, classes=num_classes, weights_path=weights_path)
    model = DenseNet(img_rows, img_cols,  reduction=0.5, classes=num_classes)    
    #MALIHE COMMENT THIS
    #sgd   = SGD(lr=1e-3, decay=1e-4, momentum=0.9, nesterov=True)#lr=1e-2 decay=1e-6 momentum=0.9, nesterov=True
    #model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    #End MALIHE COMMENT THIS
    opt = keras.optimizers.adam(lr=0.0001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    #model.summary()
        
    #Train model with our data
    MODEL_DIR = "D:\\DeepLearning\\COVID19\\PythonCode\\DenseNet-Keras-master\\logsFolder\\denseNet_CTCOVID_KFOLD1_ranSeed"                         # ********
    filepath = "saved-model-{epoch:02d}-{val_acc:.4f}.hdf5"

    checkpoint = ModelCheckpoint(filepath=os.path.join(MODEL_DIR,
                       "model-{epoch:02d}-{val_acc:.4f}.h5"),
                        monitor = "val_acc", save_weights_only=True)#save_best_only = True)
    
    print('Not using data augmentation.')
    model.fit(inputs[train], targets[train],
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              #validation_split=validation_split,
              validation_data = (inputs[test], targets[test]),
              shuffle=True,
              callbacks=[lr_reducer, early_stopper, csv_logger,checkpoint])
      
    #End Train model with our data
    
    # Generate generalization metrics
    scores = model.evaluate(inputs[test], targets[test], verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    
    y_pred     = model.predict(inputs[test])
    y_pred_max = y_pred.argmax(axis=1)
    y_pred_max = np.array(y_pred_max, dtype=np.uint8) 
    print('confusion_report------------------------------------------------')
    print(confusion_matrix(targets_array[test], y_pred_max))
    print('classification_report------------------------------------------------')
    print(classification_report(targets_array[test], y_pred_max,digits=4))

    # Increase fold number
    fold_no = fold_no + 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')


#f = open('InputTest.pckl', 'wb')
#pickle.dump( inputs[test], f)
#f.close()
#f = open('InputTrain.pckl', 'wb')
#pickle.dump( inputs[train], f)
#f.close()
#f = open('TargetTest.pckl', 'wb')
#pickle.dump( targets[test], f)
#f.close()
#f = open('TargetTrain.pckl', 'wb')
#pickle.dump( targets[train], f)
#f.close()
#
#f = open('IndTrain.pckl', 'wb')
#pickle.dump( train, f)
#f.close()
#f = open('IndTest.pckl', 'wb')
#pickle.dump( test, f)
#f.close()

#load variables
#f = open('TrLCedar.pckl', 'rb')
#y_train = pickle.load(f)
#f.close()