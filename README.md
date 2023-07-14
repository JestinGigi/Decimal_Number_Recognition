# Handwritten Digit Recognition

This project aims to recognize handwritten digits using deep learning techniques. The implementation utilizes TensorFlow and OpenCV libraries for building and training a neural network model.

## Dataset

The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) provided by Keras is used in this project. It consists of a large number of labeled images of handwritten digits, which serve as the training and testing data for the model.

## Preprocessing

Before feeding the dataset to the model, the pixel values of the images are normalized to a range of 0 to 255. This step ensures that the data is on a consistent scale and can be effectively processed by the neural network.

## Model Architecture

The model architecture is built as a 4-layer sequential model. Each layer utilizes specific activation functions to introduce non-linearity into the network. The activation functions used are Rectified Linear Unit (ReLU) and Softmax.

## Model Compilation and Training

`model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])`

The model is trained iteratively presenting the training dataset to the model, allowing it to learn and adjust its internal parameters.


## Results

After training the model, it achieves an accuracy of 97% on the testing dataset. This indicates that the model is able to correctly identify handwritten digits with a high degree of accuracy.
