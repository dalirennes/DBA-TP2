#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Muhammad Ihtisham Alam Khan and Da Li
"""

import struct
import numpy as np
import time
import matplotlib.pyplot as plt


# provided function for reading idx files
def read_idx(filename):
    '''Reads an idx file and returns an ndarray'''
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


        
# Task 3: input pre-preprocessing
def input_preprocessing(x_train):
    """
    This function takes the input array and flattens each image sample to a 1D array.
    It also normalizes the values to be between 0 and 1.
    """  
    flat_x_train = x_train.reshape(x_train.shape[0], -1)
    flat_x_train = flat_x_train / 255 
    print(f"Shape of flattened x_train: {flat_x_train.shape}")
    return flat_x_train


# Task 4: output pre-processing
def output_preprocessing(y_train):
    """
    This function takes the input array and one-hot encodes it.
    """
    labels = np.zeros((y_train.size, 10), dtype=int)
    labels[np.arange(y_train.size), y_train] = 1
    return labels
        
# Task 5: creating and initializing matrices of weights
def layer_weights(m, n):
    return np.random.randn(m, n) * (1/ np.sqrt(n))
        
# Task 7: defining functions sigmoid, softmax, and sigmoid'
def sigmoid(x):
    """This function returns the sigmoid of x for each element in x"""
    return 1 / (1 + np.exp(-x))

def softmax(x):
    """This function returns the softmax of x for each element in x, a probability distribution for the classes"""
    return np.exp(x-np.max(x)) / np.sum(np.exp(x-np.max(x)) , axis=-1, keepdims=True)

def sigmoid_prime(x):
    """This function returns the derivative of the sigmoid of x for each element in x, it helps in backpropagation"""
    return np.exp(-x) / (1 + np.exp(-x))**2



# Task 8-9: forward pass
def forward_pass(input):
    """
    This function takes the input, performs the forward pass through the network, and returns the output.
    """
    #Reshape input to 2D array if it is 1D
    if input.ndim == 1:
        input = input.reshape(1, -1)

    #Hidden Layer 1
    z1 = np.dot(input, w1)
    a1 = sigmoid(z1)

    #Hidden Layer 2
    z2 = np.dot(a1, w2)
    a2 = sigmoid(z2)

    #Output Layer
    z3 = np.dot(a2, w3)
    a3 = softmax(z3)
    
    return [(z1,z2,z3), (a1,a2,a3)] 
        
# Task 10: backpropagation
def backpropagation(actual_input, activation, z , actual_output):
    """
    This function takes the actual_input, activations, z values, and actual_output from the forward pass and computes the weight updates.
    """
    # Error in the Output Layer
    e3 = activation[-1] - actual_output
    
    # Back Propagation Weight Update for w3, Output Layer to Hidden Layer 2
    dw3 = np.dot(activation[-2].T, e3)
    
    # Error in Hidden Layer 2
    e2 = np.dot(e3, w3.T) * sigmoid_prime(z[-2])
    
    # Weight Update for w2, Hidden Layer 2 to Hidden Layer 1
    dw2 = np.dot(activation[0].T, e2)
    
    # Error in Hidden Layer 1
    e1 = np.dot(e2, w2.T) * sigmoid_prime(z[0])
    
    # Weight Update for w1 Hidden Layer 1 to Input Layer
    dw1 = np.dot(actual_input.T, e1)
    
    return dw1, dw2, dw3



# Task 11: weight updates
def update_weight(w1, w2, w3, dw1, dw2, dw3, learning_rate):
    """
    This function updates the weights based on the learning rate and the weight changes computed during backpropagation.
    """
    w1 -= learning_rate * dw1
    w2 -= learning_rate * dw2
    w3 -= learning_rate * dw3
    return w1, w2, w3
        
# Task 12: computing error on test data
def compute_error(x, y):
    """
    This function computes the error rate on the input x and expected output y.
    """
    # Forward pass on the test data
    predictions = forward_pass(x)[1][-1]

    # Get predicted labels by finding the index of the max probability for each sample
    predicted_labels = np.argmax(predictions, axis=1)
    test_digits = np.argmax(y, axis=1)

    error_rate = np.sum(predicted_labels != test_digits) / len(test_digits)
    return error_rate
        

# Task 14-15: training
def train(x_train, y_train, num_epochs=30, learning_rate=0.001):
    """
    Train the neural network using the training dataset.
    
    Parameters:
        x_train: Training inputs, shape (num_samples, 784).
        y_train: Training labels, one-hot encoded, shape (num_samples, 10).
        num_epochs: The number of epochs (full dataset iterations) to train the network.
        learning_rate: The learning rate for weight updates.
    
    Returns:
        None
    """
    global w1, w2, w3  #updating the global weights

    for epoch in range(num_epochs):

        for i in range(x_train.shape[0]):

            sample_input = x_train[i].reshape(1, -1)
            sample_target = y_train[i].reshape(1, -1)

            fp_values = forward_pass(sample_input)

            # Perform backpropagation to calculate weight updates
            dw1, dw2, dw3 = backpropagation(sample_input, fp_values[1], fp_values[0],sample_target)

            # Update weights
            w1, w2, w3 = update_weight(w1, w2, w3, dw1, dw2, dw3, learning_rate)

        # Calculate and print the error rate for the epoch
        error_rate = compute_error(x_train, y_train)
        print(f"Epoch {epoch+1}/{num_epochs}, Error rate: {error_rate:f}")



# Task 16-18: batch training
def batch_training(x_train, y_train, num_epochs=30, learning_rate=0.001, batch_size=30):
    """
    Train the neural network using the training dataset with mini-batch training.
    
    Parameters:
        x_train: Training inputs, shape (num_samples, 784).
        y_train: Training labels, one-hot encoded, shape (num_samples, 10).
        num_epochs: The number of epochs (full dataset iterations) to train the network.
        learning_rate: The learning rate for weight updates.
        batch_size: The number of samples in each mini-batch.
    
    Returns:
        None
    """
    global w1, w2, w3  #updating the global weights

    for epoch in range(num_epochs):

        for i in range(0, x_train.shape[0], batch_size):

            batch_inputs = x_train[i:i+batch_size]
            batch_targets = y_train[i:i+batch_size]

            fp_values = forward_pass(batch_inputs)

            # Perform backpropagation to calculate weight updates
            dw1, dw2, dw3 = backpropagation(batch_inputs, fp_values[1], fp_values[0], batch_targets)

            # Update weights
            w1, w2, w3 = update_weight(w1, w2, w3, dw1, dw2, dw3, learning_rate)

        # Calculate and print the error rate for the epoch
        error_rate = compute_error(x_train, y_train)
        print(f"Epoch {epoch+1}/{num_epochs}, Error rate: {error_rate:f}")

def save_model(self, filename='model_weights.npz'):
        """ save model weights to local"""
        np.savez(filename, w1=self.weights['w1'], w2=self.weights['w2'], w3=self.weights['w3'])
        print(f"Model saved to {filename}")


#------MAIN-------
# Task 1: reading the MNIST files into Python ndarrays
x_train = read_idx('./mnist/train-images.idx3-ubyte')
y_train = read_idx('./mnist/train-labels.idx1-ubyte')   
x_test = read_idx('./mnist/t10k-images.idx3-ubyte')
y_test = read_idx('./mnist/t10k-labels.idx1-ubyte')

print(f"x_train shape:{x_train.shape}, {x_train.ndim}")
print(f"y_train shape:{y_train.shape} , {y_train.ndim}")
print(f"x_test shape:{x_test.shape}, {x_test.ndim}")
print(f"y_test shape:{y_test.shape}, {y_test.ndim}")

# Task 2: visualize a few bitmap images
# images = x_train[:10]
# for index,image in enumerate(images):
#     plt.subplot(1, 10, index+1)
#     plt.imshow(image)
# plt.show()


# Task 3: input pre-preprocessing
x_train = input_preprocessing(x_train)
x_test = input_preprocessing(x_test)
y_train = output_preprocessing(y_train)
y_test = output_preprocessing(y_test)

# Task 6: Initialize weight matrices for each layer connection
w1 = layer_weights(784, 128)  # Weights from input layer to hidden layer 1
w2 = layer_weights(128, 64)   # Weights from hidden layer 1 to hidden layer 2
w3 = layer_weights(64, 10)    # Weights from hidden layer 2 to output layer

# Task 14-15: training
# train(x_train, y_train, num_epochs=30, learning_rate=0.001)

# Task 16-18: training
batch_training(x_train, y_train, num_epochs=60, learning_rate=0.001, batch_size=30)
save_model()


# sample_input = x_train[0]

# output = forward_pass(sample_input)
# print("Output probabilities:", output)
# print("Shape of output:", output.shape) 
# print("Predicted digit:", np.argmax(output))  # Find the digit with the highest probability



# Task 13: error with initial weights
error_rate = compute_error(x_test, y_test)
print(f"Error rate with initial weights: {error_rate:f}")
"""ANSWER TO TASK 13: Error rate with initial weights: 0.9108 (91.08%)
The result does not confirm to my expectations since this error rate is too high but 
it is expected as the weights are randomly initialized and the network has not been trained yet."""




"""Question 1 What are the shapes of the four computed ndarrays? What is the meaning of the axes
 of each ndarray?
 
    Answer:
    x_train shape:(60000, 28, 28)
    x_train.ndim: 3
    This means that the x_train array has 60000 images, each image is 28x28 pixels.
    The dimension is 3 which means it is a 3D array and the first dimension is the number of images,
    the second and third dimensions are the height and width of the image.

    y_train shape:(60000,)
    y_train.ndim: 1
    This means that the y_train array has 60000 labels, one for each image in x_train.
    This is a 1D array. The length of the array is the same as the number of images in x_train.

    x_test shape:(10000, 28, 28)
    x_test.ndim: 3
    This means that the x_test array has 10000 images, each image is 28x28 pixels.
    This is a 3D array and the first dimension is the number of images,
    the second and third dimensions are the height and width of the image.

    y_test shape:(10000,)
    y_test.ndim: 1
    This means that the y_test array has 10000 labels, one for each image in x_test.
    This is a 1D array. The length of the array is the same as the number of images in x_test.

 """


"""Question 2: 
1:From Input Layer to Hidden Layer 1
    Equation: z1 = a0 . w1
    Activation: a1 = sigmoid(z1)

   -Shape of a0: The input a0 is a vector representing a flattened 28x28 image, so it has a shape of (784,).
   -Shape of w1: w1 is the weight matrix connecting the input layer (784 neurons) to the first hidden layer (128 neurons), so it has a shape of (784, 128).
   -Shape of z1: When we perform the dot product a0 . w1, the resulting z1 has a shape of (128,) , since 784 elements in a0 match the first dimension of w1, and we are left with 128 values.
   -Shape of a1: After applying the sigmoid function element-wise on z, a1 will also have a shape of (128,).

2:From Hidden Layer 1 to Hidden Layer 2:
    Equation: z2 = a1 . w2
    Activation: a2 = sigmoid(z2)
   -Shape of a1: The output from the first hidden layer a1 has a shape of (128,).
   -Shape of w2: w2 is the weight matrix connecting Hidden Layer 1 (128 neurons) to Hidden Layer 2 (64 neurons), so it has a shape of (128, 64).
   -Shape of z2: The dot product a1 . w2 results in z2 with a shape of (64,).
   -Shape of a2: After applying the sigmoid function on z2, a2 will also have a shape of (64,).

3:From Hidden Layer 2 to Output Layer:
    Equation: z3 = a2 . w3
    Activation: a3 = softmax(z3)
   -Shape of a2: The output from the second hidden layer a2 has a shape of (64,).
   -Shape of w3: w3 is the weight matrix connecting Hidden Layer 2 (64 neurons) to the Output Layer (10 neurons), so it has a shape of (64, 10).
   -Shape of z3: The dot product a2 . w3 results in z3 with a shape of (10,).
   -Shape of a3: After applying the softmax function on z3, a3 will also have a shape of (10,), representing the probabilities for each digit in the range of 0-9.

Each step of the feed-forward pass results in shapes that match correctly for the matrix operations. The shapes of intermediate results are as follows:
z1: (128,), a1: (128,)
z2: (64,), a2: (64,)
z3: (10,), a3: (10,)

The matrix multiplications and activation functions will work as expected, producing an output vector a3 of shape (10,), which is required for the classification of each input image."""


"""Question 3:
    1:Error in the Output Layer
    Equation: e3 = a3 - y
   -Shape of a3: The output activations a3 has a shape of (1, 10), representing the probabilities for each of the 10 output classes.
   -Shape of y: The one-hot encoded target vector for the input image has a shape of (1, 10).
   This means that they are eligible for subtraction, resulting in e3 with a shape of (1, 10).
   e3 represents the difference between the predicted output and the actual target values.

    2:Back Propagation Weight Update for w3 (Output Layer to Hidden Layer 2)
    Equation: δw3 = a2^T . e3 
   -Shape of a2: the activation from Hidden Layer 2 with a shape of (1, 64).
   -Shape of a2^T: The transpose of a2 will have a shape of (64, 1).
   -Shape of e3: (1, 10).
   -Shape of δw3: (64, 10). w3 has the same shape which allows for proper weight updates.

    3:Error in Hidden Layer 2
    Equation: e2 = (e3 . w3^T) * S'(z2)
   -Shape of e3: (1, 10).
   -Shape of w3^T: (10, 64).
   -Shape of e3 . w3^T: (1, 64).
   -Shape of S'(z2): The derivative of the sigmoid function, applied element-wise to z2, has a shape of (1, 64).
   -Shape of e2: The shape of (1, 64), representing the error for each neuron in Hidden Layer 2.

    4:Weight Update for w2 (Hidden Layer 2 to Hidden Layer 1)
    Equation: δw2 = a1^T . e2
    -Shape of a1: The activation from Hidden Layer 1 with a shape of (1, 128).
    -Shape of a1^T:The transpose of a1 with a shape of (128, 1).
    -Shape of e2: (1, 64).
    -Shape of δw2: (128, 64), matching the shape of w2.

    5:Error in Hidden Layer 1
    Equation: e1 = (e2 . w2^T) * S'(z1)
    -Shape of e2: (1, 64).
    -Shape of w2^T: The transpose of w2 has a shape of (64, 128).
    -Shape of e2 . w2^T: (1, 128).
    -Shape of S'(z1): The derivative of the sigmoid function, applied element-wise to z1, has a shape of (1, 128).
    -Shape of e1: The shape of (1, 128), representing the error for each neuron in Hidden Layer 1.

    6:Weight Update for w1 (Hidden Layer 1 to Input Layer)
    Equation: δw1 = a0^T . e1
    -Shape of a0: (1, 784).
    -Shape of a0^T: The transpose of a0 has a shape of (784, 1).
    -Shape of e1: (1, 128).
    -Shape of δw1: (784, 128), matching the shape of w1.

The shapes of the matrices and vectors align correctly for all operations in backpropagation:

The shapes of the matrices and vectors align correctly for all operations in backpropagation:
- e3: (1, 10)
- δw3: (64, 10)
- e2: (1, 64)
- δw2: (128, 64)
- e1: (1, 128)
- δw1: (784, 128)"""