# Pallets-Image-In-Dark-Background-Improvement


there are 3 parts

darkenvmodel contains

traning/testing relative code and dataset

the code in the model is a Convolutional Neural Network (CNN) for image processing
there are 7 convolutional layers
starts by taking a color image with RGB channels and transforms it into feature channels
Each 'conv' layer takes the previous features and applies a 3x3 filter to them, 

gradually increasing the number of feature channels

 The 'ReLU' rectified linear unit activation function adds non-linearity,
 making the network learn complex patterns.

Max pooling reduces image size by simplifying information,
 and upsampling enlarges it while preserving details. 

after we decide the network structure we can apply
 The 'forward' function improve the image and returns control factors for adjustment.

 