# COVID19_Reg_CS_Caps_Dense - Keras/Tensorflow


This is a [Keras](https://keras.io/) and [Tensorflow](https://www.tensorflow.org/) implementation of Regularized Cost-Sensitive CapsNet with DenseNet.

To know more about our proposed model, please refer to the [original paper](https://www.nature.com/articles/s41598-021-97901-4)

**************************************************************************************************************************************************
To run our "Regularized Cost-Sensitive CapsNet with DenseNet" follow the below steps:

1- Run "[mainDenseNet.py](https://github.com/Javidi31/COVID19_Reg_Caps_Dense/blob/master/mainDenseNet.py)" for the training model using DenseNet-18.
The best model will be saved in the folder "DenseNetModels"

2- Run "[ExtarctDenseNetFeatures.py](https://github.com/Javidi31/COVID19_Reg_Caps_Dense/blob/master/ExatrctDenseNetFeatures.py)" for extracting DenseNet features.
This code loads the best model from folder "DenseNetModels" (in step 1) 
and then extracts train and test features in a specific layer number. 
Features will be saved in the folder "DenseNetFeatures"

3- Run "[RegCSDenseCapsNet.py](https://github.com/Javidi31/RegCapsNet/blob/main/RegResCapsNet.py)" aiming signatures classification. This 
file used features of step 2 (which are saved in folder "DenseNetFeatures") 
as input data.

**************************************************************************************************************************************************
# Dataset

1- COVID_CT dataset is Available at (https://github.com/UCSD-AI4H/COVID-CT/tree/master/Images-processed) (a number of samples of original dataset are available in the Datasets folder).


**************************************************************************************************************************************************
you can clone code using the following command:

`git clone --single-branch --branch master https://github.com/Javidi31/COVID19_Reg_Caps_Dense.git`

Good luck
