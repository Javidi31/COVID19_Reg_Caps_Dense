# COVID19_Reg_CS_Caps_Dense - Keras/Tensorflow


This is a [Keras](https://keras.io/) and [Tensorflow](https://www.tensorflow.org/) implementation of Regularized Cost-Sensitive CapsNet with DenseNet.

To know more about our proposed model, please refer to the [original paper](https://www.nature.com/articles/s41598-021-97901-4)

**************************************************************************************************************************************************
To run our "Regularized Cost-Sensitive CapsNet with DenseNet" follow the below steps:

1- Run "[mainResNet.py](https://github.com/Javidi31/RegCapsNet/blob/main/mainResNet.py)" for the training model using ResNet-18.
The best model will be saved in the folder "ResNetModels"

2- Run "[ExatrctResNetFeatures.py](https://github.com/Javidi31/RegCapsNet/blob/main/ExtractResNetFeatures.py)" for extracting ResNet features.
This code loads the best model from folder "ResNetModels" (in step 1) 
and then extracts train and test features in a specific layer number. 
Features will be saved in the folder "ResNetFeatures"

3- Run "[RegResCapsNet.py](https://github.com/Javidi31/RegCapsNet/blob/main/RegResCapsNet.py)" aiming signatures classification. This 
file used features of step 2 (which are saved in folder "ResNetFeatures") 
as input data.

**************************************************************************************************************************************************
# Dataset

1- COVID_CT dataset is Available at (https://github.com/UCSD-AI4H/COVID-CT/tree/master/Images-processed) (a number of samples of original and pre-processed dataset are available in the Datasets folder).


**************************************************************************************************************************************************
For more information about loading dataset or setting the parameters, please refer to utilities folder.
**************************************************************************************************************************************************


Good luck
