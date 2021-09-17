
"""REG_CS_CAPS_DENSE.ipynb

"""

from __future__ import division, print_function, unicode_literals

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 1.x
# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import pickle
import csv
import numpy as np
import tensorflow as tf
import pandas as pd
import httplib2
import os
import requests
from PIL import Image
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from google.colab import drive
drive.mount("/content/gdrive/")

tf.reset_default_graph()

np.random.seed(42)
tf.set_random_seed(42)

"""# Load Data

"""

#!unzip '/mydrive/My Drive/Dataset/COVID19/Features/Features_COVIDCT_KFOLD_Lay72_Float32.zip' -d '/mydrive/My Drive/Dataset/COVID19/Features/Features_COVIDCT_KFOLD_Lay72_Float32'
# !unzip /content/gdrive/MyDrive/Covid19/CovidCT/Features_COVIDCT_KFOLD_Lay15_Float32.zip -d '/content/gdrive/My Drive/Covid19/CovidCT/Features_COVIDCT_KFOLD_Lay15_Float32'

# count of data on each folder
#!ls '/mydrive/My Drive/Dataset/COVID19/Features'
# !ls  "gdrive/My Drive/Covid19/Features_Prot2_Lay15/Features_Prot2_Lay15" | wc -l

checkpoint_path = 'gdrive/My Drive/Covid19/codes/REG_CS_CAPS_DENSE/Prot1_CovidCT/'

image_size = 28
img_channel = 128
all_img_no = 349+397
# tr_img_no = 17716
# ts_img_no = 1969

pathMain = '/content/gdrive/MyDrive/Covid19/CovidCT/Features_COVIDCT_KFOLD_Lay72_Float32/'

from tqdm.notebook import tqdm
if False:
  cnt = 0
  All = np.zeros( shape = (all_img_no,image_size,image_size,img_channel) )
  All_label = np.zeros( shape = (all_img_no))
  for i in tqdm(range(349)):
    path = pathMain + 'COVID/'
    path = path +  str(i)+ 'Out_lay72.pckl'  
    with open(path, 'rb') as f:
      All[cnt,:,:,:] = pickle.load(f)
    All_label[cnt] = 0
    cnt +=1
  
  for i in tqdm(range(397)):
    path = pathMain + 'NonCOVID/'
    path = path +  str(i+349)+ 'Out_lay72.pckl'  
    with open(path, 'rb') as f:
      All[cnt,:,:,:] = pickle.load(f)
    All_label[cnt] = 1
    cnt +=1

  with open(pathMain + 'DS.pckl', 'wb') as f:
    pickle.dump([All, All_label], f)

else:
  with open(pathMain + 'DS.pckl', 'rb') as f:
      All, All_label = pickle.load(f)

num_class = len(set(All_label))

X = tf.placeholder(shape=[None, image_size, image_size, img_channel], dtype=tf.float32, name="X")

"""# Primary Capsules

The first layer will be composed of 32 maps of 6×6 capsules each, where each capsule will output an 8D activation vector:
"""

caps1_n_maps = 32
caps1_n_caps = caps1_n_maps * 6 * 6  # 1152 primary capsules 
# caps1_n_caps = caps1_n_maps * 11 * 11  # 41472 primary capsules    (FOR Cedar)
caps1_n_dims = 8

np.sqrt(11*11*256/(32*8))

conv1_params = {
    "filters": 256,
    "kernel_size": 9,#------------------------------
    "strides": 1,
    "padding": "valid",
    "activation": tf.nn.relu,
}

conv2_params = {
    "filters": caps1_n_maps * caps1_n_dims, # 256 convolutional filters
    "kernel_size": 9,#------------------------------
    "strides": 2,
    "padding": "valid",
    "activation": tf.nn.relu
}

conv1 = tf.layers.conv2d(X, name="conv1", **conv1_params)
conv2 = tf.layers.conv2d(conv1, name="conv2", **conv2_params)

caps1_raw = tf.reshape(conv2, [-1, caps1_n_caps, caps1_n_dims],
                       name="caps1_raw")

def squash(s, axis=-1, epsilon=1e-7, name=None):
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector

caps1_output = squash(caps1_raw, name="caps1_output")

"""# Digit Capsules

## Compute the Predicted Output Vectors
"""

caps2_n_caps = num_class
caps2_n_dims = 16

init_sigma = 0.1

W_init = tf.random_normal(
    shape=(1, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims),
    stddev=init_sigma, dtype=tf.float32, name="W_init")
W = tf.Variable(W_init, name="W")

batch_size = tf.shape(X)[0]
W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="W_tiled")

caps1_output_expanded = tf.expand_dims(caps1_output, -1,
                                       name="caps1_output_expanded")
caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2,
                                   name="caps1_output_tile")
caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, caps2_n_caps, 1, 1],
                             name="caps1_output_tiled")

caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled,
                            name="caps2_predicted")

"""## Routing by agreement"""

raw_weights = tf.zeros([batch_size, caps1_n_caps, caps2_n_caps, 1, 1],
                       dtype=np.float32, name="raw_weights")

"""### Round 1"""

routing_weights = tf.nn.softmax(raw_weights, dim=2, name="routing_weights")

weighted_predictions = tf.multiply(routing_weights, caps2_predicted,
                                   name="weighted_predictions")
weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True,
                             name="weighted_sum")

caps2_output_round_1 = squash(weighted_sum, axis=-2,
                              name="caps2_output_round_1")

"""### Round 2"""

caps2_output_round_1_tiled = tf.tile(
    caps2_output_round_1, [1, caps1_n_caps, 1, 1, 1],
    name="caps2_output_round_1_tiled")

agreement = tf.matmul(caps2_predicted, caps2_output_round_1_tiled,
                      transpose_a=True, name="agreement")

raw_weights_round_2 = tf.add(raw_weights, agreement,
                             name="raw_weights_round_2")

"""The rest of round 2 is the same as in round 1:"""

routing_weights_round_2 = tf.nn.softmax(raw_weights_round_2,
                                        dim=2,
                                        name="routing_weights_round_2")
weighted_predictions_round_2 = tf.multiply(routing_weights_round_2,
                                           caps2_predicted,
                                           name="weighted_predictions_round_2")
weighted_sum_round_2 = tf.reduce_sum(weighted_predictions_round_2,
                                     axis=1, keep_dims=True,
                                     name="weighted_sum_round_2")
caps2_output_round_2 = squash(weighted_sum_round_2,
                              axis=-2,
                              name="caps2_output_round_2")

caps2_output = caps2_output_round_2

"""# Estimated Class Probabilities (Length)"""

def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    with tf.name_scope(name, default_name="safe_norm"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=keep_dims)
        return tf.sqrt(squared_norm + epsilon)

y_proba = safe_norm(caps2_output, axis=-2, name="y_proba")

y_proba_argmax = tf.argmax(y_proba, axis=2, name="y_proba")

y_pred = tf.squeeze(y_proba_argmax, axis=[1,2], name="y_pred")

"""# Labels"""

y = tf.placeholder(shape=[None], dtype=tf.int64, name="y")

"""# Margin loss"""

m_plus = 0.9
m_minus = 0.1
lambda_ = 0.05

T = tf.one_hot(y, depth=caps2_n_caps, name="T")

caps2_output_norm = safe_norm(caps2_output, axis=-2, keep_dims=True,
                              name="caps2_output_norm")

present_error_raw = tf.square(tf.maximum(0., m_plus - caps2_output_norm),
                              name="present_error_raw")
present_error = tf.reshape(present_error_raw, shape=(-1, num_class),
                           name="present_error")

absent_error_raw = tf.square(tf.maximum(0., caps2_output_norm - m_minus),
                             name="absent_error_raw")
absent_error = tf.reshape(absent_error_raw, shape=(-1, num_class),
                          name="absent_error")

L = tf.add(T * present_error, lambda_ * (1.0 - T) * absent_error,
           name="L")

C = tf.placeholder(shape=[None,1], dtype=tf.float32)

L_CS = tf.multiply(L , C) 

margin_loss = tf.reduce_mean(tf.reduce_sum(L_CS, axis=1), name="margin_loss")

"""## Final Loss"""

alpha = 0.00001


regularizer = tf.nn.l2_loss(W_tiled)
beta = 0.001
##loss = tf.add(loss_, beta * regularizer, name="loss")
loss = tf.add(margin_loss, beta * regularizer, name="loss")

loss_for_plot = tf.add(margin_loss, beta * regularizer / tf.cast(batch_size, tf.float32), name="loss") #**

"""## Accuracy"""

correct = tf.equal(y, y_pred, name="correct")
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

"""## Training Operations"""

optimizer = tf.train.AdamOptimizer(learning_rate=0.0003)
gvs = optimizer.compute_gradients(loss)
capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs]
training_op = optimizer.apply_gradients(capped_gvs)


#training_op = optimizer.minimize(loss, name="training_op")

init = tf.global_variables_initializer()
saver = tf.train.Saver( max_to_keep =1 , filename = 'TestName')

def getNextBatchTrain(batch_size):
  N = np.size(Train,0)
  idx = np.random.randint(0,N,batch_size)
  batchLabel = Train_label[idx]
  return Train[idx,:]  , batchLabel.astype('uint8')

"""# Training"""

from sklearn.model_selection import KFold
kfold = KFold(n_splits=10, shuffle=True,random_state=101)

checkpoint_path_main = checkpoint_path

fold_no = 0
start_fold = 1
for train, test in kfold.split(All, All_label):
  fold_no += 1
  if start_fold > fold_no:
    continue

  best_acc_val = 0
  checkpoint_path = checkpoint_path_main + 'Fold' + str(fold_no)

  Train = All[train]
  Train_label = All_label[train]
  Test = All[test]
  Test_label = All_label[test]

  n_epochs = 400
  batch_size = 8
  restore_checkpoint = False

  n_iterations_per_epoch = len(Train_label) // batch_size
  n_iterations_validation = len(Test_label) #// batch_size
  best_loss_val = np.infty

  with tf.Session() as sess:
      if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
          saver.restore(sess, checkpoint_path)
          with open(checkpoint_path + 'OtherVars', 'rb') as f:
              acc_trains, loss_trains, acc_tests, loss_tests ,time_per_epochs, start_epoch , y_pred_all = pickle.load(f)
              start_epoch += 1
          print('\nStarting from epoch: %.0f\n' %(start_epoch + 1))
      else:
          print('\nCheck point not loaded\n')
          init.run()
          loss_trains = []
          acc_trains = [] 
          loss_tests = []
          acc_tests = []
          time_per_epochs = []
          start_epoch = 0
          y_pred_all = np.zeros((n_iterations_validation,n_epochs))

      for epoch in range(start_epoch,n_epochs):
          startTime = time.time()
          loss_train = []
          acc_train=[]
          for iteration in range(1, n_iterations_per_epoch + 1):
              X_batch, y_batch = getNextBatchTrain(batch_size)
              CC = np.zeros((batch_size,1))
              for i in range(batch_size):
                CC[i] = 1 - np.sum(Train_label == y_batch[i]) / len(Train_label)
              # Run the training operation and measure the loss:
              _, loss_train_batch,acc_train_batch = sess.run(
                  [training_op, loss_for_plot,accuracy],
                  feed_dict = {X: X_batch.reshape([-1, image_size, image_size, img_channel]),
                            y: y_batch,
                            C: CC})
              print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(
                        iteration, n_iterations_per_epoch,
                        iteration * 100 / n_iterations_per_epoch,
                        loss_train_batch),
                    end="")
              
              loss_train.append(loss_train_batch)
              acc_train.append(acc_train_batch)

          end_time = time.time()
          time_per_epochs.append(end_time - startTime)
          print('\nElapsed: %.1f' % (end_time - startTime))
          remainHour = (n_epochs-epoch) * (end_time - startTime)/3600
          print('Estimated remaining time: %.1f hours' % remainHour)

          acc_trains.append(np.mean(acc_train))
          loss_trains.append(np.mean(loss_train))

          #print("*****")
          #print("loss_train:",np.mean(loss_train)) #**        
          #print("acc_train",np.mean(acc_train)*100) #**Javidi
          #print("*****")

          # At the end of each epoch,
          # measure the validation loss and accuracy:
          loss_vals = []
          acc_vals = []
          
          for iteration in range(1, n_iterations_validation + 1):
              X_batch = Test[iteration-1:iteration]
              y_batch = Test_label[iteration-1:iteration].astype('uint8')
              CC = np.zeros((1,1))
              
              CC[0] = 1 - np.sum(Train_label == y_batch[0]) / len(Train_label)
              #X_batch, y_batch = getNextBatchTest(batch_size)
              loss_val, acc_val , y_pred_sample = sess.run(
                      [loss_for_plot, accuracy , y_pred],
                      feed_dict={X: X_batch.reshape([-1, image_size, image_size, img_channel]),
                                y: y_batch,
                                C: CC})

              loss_vals.append(loss_val)
              acc_vals.append(acc_val)
              y_pred_all[iteration-1 , epoch] =  y_pred_sample

              print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
                        iteration, n_iterations_validation,
                        iteration * 100 / n_iterations_validation),
                    end=" " * 10)
              
          loss_val = np.mean(loss_vals)
          acc_val = np.mean(acc_vals)

          print("\rEpoch: {}  Train accuracy: {:.4f}%  Loss_train: {:.6f}  Val accuracy: {:.4f}%  Loss_test: {:.6f}{}".format(
              epoch + 1, np.mean(acc_train)*100,np.mean(loss_train),acc_val * 100, loss_val,
              " (improved)" if loss_val < best_loss_val else ""))


          loss_tests.append(loss_val) #**
          acc_tests.append(acc_val) #**

          #print(np.mean(loss_trains)) #**        
          

          #**
          np.savetxt(checkpoint_path+"loss_tr.csv", loss_trains, delimiter=",")
          np.savetxt(checkpoint_path+"loss_te.csv", loss_tests, delimiter=",")
          np.savetxt(checkpoint_path+"acc_tr.csv", acc_trains, delimiter=",")
          np.savetxt(checkpoint_path+"acc_te.csv", acc_tests, delimiter=",")

          # Save model all the time
            save_path = saver.save(sess, checkpoint_path)
            best_acc_val = acc_val
            label_best = y_pred_all[: , epoch]
            print(classification_report(Test_label, label_best,digits=4))

          with open(checkpoint_path + 'OtherVars', 'wb') as f:
              start_epoch = epoch
              pickle.dump([acc_trains, loss_trains, acc_tests, loss_tests ,time_per_epochs, start_epoch,y_pred_all], f)

# print(confusion_matrix(Test_label, y_pred_all[:,epoch]))
# print(classification_report(Test_label, y_pred_all[:,epoch]))
y_pred_all[: , 2]