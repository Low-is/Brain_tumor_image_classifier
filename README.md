# MRI Brain Tumor Image Classification
Classify brain MRI images as either "tumor" or "normal" using a Convolutional Neural Network

Model was adpated from an [Analytics Vidhya Tutorial](https://www.analyticsvidhya.com/blog/2020/10/create-image-classification-model-python-keras/) and trained on a Kaggle brain MRI dataset.


Neural netrowks (NNs) are often used for deep learning. For image classification, Convolutional Neural Networks (CNNs) are instead used rather than a regular neural network. NNs are often used with a 1D vector (as input data) and does not have spatial understanding (does not work with images). CNNs often use 2D or 3D grids (typically an image format) and are capable of maintaining spatial relationships (best for images, videos, and spatial data).

CNNs read images by applying a filter on the image (convolution layer) and identifies features (e.g., colors, edges, textures) all based on pixel values. To ensure powerful performance and accuracy when making predictons, the model will filter out any features considered "not important" (pooling layer) by reducing the spatial size (zooming in) and leaving the most important info (features present in that zoomed in image). Once all important features are kept, a couple of things happen at once:
1. The model performs linear combinations on the features, to ultimately decide which set of features best discriminate the images. To retain the most important features, weights are randomly assigned to each feature and the model runs and adjust those weights while adding bias to those features when their inital linear combination was not good enough to classify an image.
2. Applied an activation function (focuses on non-linearity) to ensure all important features are captured. 

After pooling, you can apply an optional dropout layer to the model, and it will randomly dropout a certain percentage of the training data. This can sometimes be helpful if you notice the model is overfitting, i.e., the model shows perfect performance on the training dataset but poor performance on the testing dataset.
