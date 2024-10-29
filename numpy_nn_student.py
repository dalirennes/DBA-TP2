#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: XXX and YYY
"""

import struct
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from matplotlib.widgets import Button

#pip install scipy



class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.weights = {
            'w1': None,
            'w2': None,
            'w3': None
        }
    x_train, y_train, x_test, y_test = None, None, None, None

    # provided function for reading idx files
    def read_idx(self, filename):
        '''Reads an idx file and returns an ndarray'''
        with open(filename, 'rb') as f:
            zero, data_type, dims = struct.unpack('>HBB', f.read(4))
            shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

    # Task 1: reading the MNIST files into Python ndarrays
    def Task1_reading_the_MNIST_files(self) -> tuple:
        """ reading the MNIST files into Python ndarrays
        """
        self.x_train = self.read_idx('./mnist/train-images.idx3-ubyte')
        self.y_train = self.read_idx('./mnist/train-labels.idx1-ubyte')   
        self.x_test = self.read_idx('./mnist/t10k-images.idx3-ubyte')
        self.y_test = self.read_idx('./mnist/t10k-labels.idx1-ubyte')
        print(f"x_train shape:{self.x_train.shape} dimensions: {self.x_train.ndim}")
        print(f"x_train shape:{self.y_train.shape} dimensions: {self.y_train.ndim}")
        print(f"x_train shape:{self.x_test.shape} dimensions: {self.x_test.ndim}")
        print(f"x_train shape:{self.y_test.shape} dimensions: {self.y_test.ndim}")
        return self.x_train, self.y_train, self.x_test, self.y_test

    """
    Question 1 What are the shapes of the four computed ndarrays? What is the meaning of the axes
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

    # Task 2: visualize a few bitmap images
    def Task2_visualize_a_few_bitmap_images(self, x_train: np.ndarray) -> None:
        """ visualize a few bitmap images
        """
        # plot the first 5 images
        for i in range(5):
            plt.subplot(1, 5, i+1)
            plt.imshow(x_train[i])
        plt.show()
        # image = x_train[1]
        # plt.imshow(image)
        # # plt.show()


    # Task 3: input pre-preprocessing    

    def input_preprocessing(self, img: np.ndarray) -> np.ndarray:
        """ flattening to 1D and normalization
            Args:
                img: numpy array of images
            returns:
                normal: numpy array of normalized images
        """
        
        # flat the images to 1D
        dim_1 = img.reshape(img.shape[0], -1)
        # nomalize the images with 255
        normal = dim_1 / 255.0
        return normal

    def Task3_input_preprocessing(self, x_train: np.ndarray, x_test: np.ndarray) -> tuple:
        #process the input data
        self.x_train = self.input_preprocessing(x_train)
        self.x_test = self.input_preprocessing(x_test)

        print(f"x_train_flat shape: {self.x_train.shape}")
        print(f"x_test_flat shape: {self.x_test.shape}")

    # Task 4: output pre-processing

    def output_processing(self, y: np.ndarray) -> np.ndarray:
        """ converting categorical labels to  one-hot vectors
        """
        out_hot_vector = np.zeros((len(y), 10))

        # fancy indexing process
        out_hot_vector[np.arange(len(y)), y] = 1
        return out_hot_vector

    def Task4_output_processing(self, y_train: np.ndarray, y_test: np.ndarray) -> tuple:
        self.y_train = self.output_processing(y_train)
        self.y_test = self.output_processing(y_test)

        print(f"y_train shape: {self.y_train.shape}")
        print(f"y_test shape: {self.y_test.shape}")
        print(f"y_train[1]: {self.y_train[1]}")
        print(f"y_test[1]: {self.y_test[1]}")

    # Task 5-6: creating and initializing matrices of weights

    def layer_weights(self, m, n):
        """ creating and initializing matrices of weights
            the sum of the variance of the weights should be 1
        ·   Function: w = 1 / sqrt(n) * random(m, n)
            Args:
                m: number of rows
                n: number of columns
            returns:
                weights: numpy array of random weights
        """
        return np.random.randn(m, n) * 1 / np.sqrt(n)

    # define matrix weights for each layer
    def Task5_6_creating_and_initializing_matrices_of_weights(self) -> dict:
        """ creating and initializing matrices of weights
        """
        weights = {
            'w1': self.layer_weights(784, 128),
            'w2': self.layer_weights(128, 64),
            'w3': self.layer_weights(64, 10)
        }
        self.weights = weights
        print(f"w1 shape: {weights['w1'].shape}")
        print(f"w2 shape: {weights['w2'].shape}")
        print(f"w3 shape: {weights['w3'].shape}")
        # print(f"w3: {weights['w3']}")

    # Task 7: defining functions sigmoid, softmax, and sigmoid'

    def sigmod(self, x: np.ndarray) -> np.ndarray:
        """ sigmoid function
            Function: f(x) = 1 / (1 + e^(-x))
            Use this funciton as the activation function  for the hidden layers
                put values between 0 and 1
            Args:
                x: numpy array
            returns:
                numpy array
        """
        try:
            exponent = np.exp(-x)
        except OverflowError:
            print(f"[!!!] Sigmod OverflowError: {x}")
            exponent = np.clip(x, -500, 500)
        return 1 / (1 + exponent)

    def softmax(self, x: np.ndarray) -> np.ndarray:
        """ softmax function
            Function: f(x) = e^(x) / sum(e^(x))
            Use this function as the activation function for the output layer
                put network output values to a probability distribution 
        """
        #substract the maximum value from the input data to avoid overflow
        max_value = np.max(x, axis=1, keepdims=True)
        exp_value = np.exp(x - max_value)

        # calculate the probability keeping the sum of the probabilities to 1
        probability = exp_value / np.sum(exp_value, axis=1, keepdims=True)
        return probability

    def sigmoid_prime(self, x: np.ndarray) -> np.ndarray:
        """ sigmoid derivative
            Function: f'(x) = f(x) * (1 - f(x))
            Use for calculating the gradients of the hidden layers in backpropagation
            Throught the chain rule optimize the weights
        """
        s = self.sigmod(x)
        # then calculate the derivative of the sigmoid
        return s * (1 - s)



    """
    Question 2 Analyze the equations in terms of array shapes in order to verify that they are well formed.

    Answer:
        In forward pass, we use the dot product to calculate the weighted sum 
        z1 = x · w1    a1 = S(z1)
        z2 = a1 · w2    a2 = S(z2)
        z3 = a2 · w3    a3 = O(z3)

        z is the weighted sum of the inputs and weights
        a is the activation result of the weighted sum
        w is thee weight matrix

        1. Calculation 1: from the input layer to the first hidden layer
            z1 = x · w1
            x: (60000, 784) # 60000 is images, 784 is the pixels of the image
            w1: (784, 128)  # From 784 input neurons to 128 hidden neurons
            z1: (60000, 128) # 60000 images, 128 hidden neurons
            a1: (60000, 128) # Same shape as upper

        2. Calculation 2: from the first hidden layer to the second hidden layer 
            z2 = a1 · w2
            a1: (60000, 128)
            w2: (128, 64) # From 128 hidden neurons to 64 hidden neurons
            z2: (60000, 64) # 60000 images, 64 hidden neurons
            a2: (60000, 64) # Same shape as upper

        3. Calculation 3: from the second hidden layer to the output layer
            z3 = a2 · w3
            a2: (60000, 64)
            w3: (64, 10) # From 64 hidden neurons to 10 output neurons
            z3: (60000, 10) # 60000 images, 10 output neurons
            a3: (60000, 10) # Same shape as upper

        Conclusion: The shapes of the arrays are well formed.
    """

    # Task 8-9: forward pass

    def forward_pass(self, x: np.ndarray) -> tuple:
        """ forward pass
            Function: 
                z1 = x * w1  a1 = sigmoid(z1)
                z2 = a1 * w2  a2 = sigmoid(z2)
                z3 = a2 * w3  a3 = softmax(z3)
            args:  
                x: (60000, 784) # 60000 images, 784 pixels
        """

        #From (60000, 784) to (60000, 128)
        z1 = np.dot(x, self.weights['w1']) 
        a1 = self.sigmod(z1)

        #From (60000, 128) to (60000, 64)
        z2 = np.dot(a1, self.weights['w2'])
        a2 = self.sigmod(z2)

        #From (60000, 64) to (60000, 10)
        z3 = np.dot(a2, self.weights['w3'])
        a3 = self.softmax(z3)

        return (a1, a2, a3)

    def predict(self, x):
        """ For handling the prediction 
            Args:
                x: numpy array of images
                """
        # input_data = x.reshape(1, 784) / 255.0  # 确保归一化到 [0, 1] 范围
        a1, a2, a3 = self.forward_pass(x)
        
        print(f"output: {a3}")
        print(f"output shape: {a3.shape} ndim: {a3.ndim}")

        return np.argmax(a3)
    """
    Question 3 Analyze the equations in terms of array shapes in order to verify that they are wellformed.

    Answer:
        This is reverse of the forward pass. 
        We calculate the error of output layer and propagate it back to the input layer.
        
        1. From layer-3 to layer-2
            Equation: error3 = a3 - y
            calculate the err of the output layer
            a3: the output of the network
            y: the true image label 
            so the error3 is: a3 - y

            Then calculate the gradient of the weights
            Equation: delta3_weight = a2.T · error3 
            a2.T: shape(64, 60000) 
            error3: shape(60000, 10) 
            a2.T is the transpose of a2, so we could multiply the two matrices
            Then the result is (64, 10)
        
        2. From layer-2 to layer-1
            Equation: error2 = error3 · w3.T * sigmoid_prime(a2)
            w3.T: shape(10, 64) is the transpose of w3
            sigmoid_prime(a2): shape(60000, 64)
            So error2 is (60000, 64)

            Then calculate the gradient of the weights
            Equation: delta2_weight = a1.T · error2
            a1.T: shape(128, 60000)
            error2: shape(60000, 64)
            Then the result is (128, 64)

        3. From layer-1 to input layer
            the same as the upper
            Equation: error1 = error2 · w2.T * sigmoid_prime(a1)
            w2.T: shape(64, 128)
            sigmoid_prime(a1): shape(60000, 128)
            error1 is (60000, 128)

            Equation: delta1_weight = x.T · error1
            x.T: shape(784, 60000)
            error1: shape(60000, 128)
            Then the result is (784, 128)

        Conclusion: The shapes of upper arrays are well formed.
    """
    # Task 10: backpropagation

    def backpropagation(self, 
                        x: np.ndarray, 
                        y: np.ndarray, 
                        a1: np.ndarray, 
                        a2: np.ndarray, 
                        a3: np.ndarray) -> tuple:
        
        error3 = a3 - y  # (60000, 10)
        delta3_weight = np.dot(a2.T, error3)  # (64, 10)

        error2 = np.dot(error3, self.weights['w3'].T) * self.sigmoid_prime(a2) # (60000, 64)
        delta2_weight = np.dot(a1.T, error2) #  (128, 64)

        error1 = np.dot(error2, self.weights['w2'].T) * self.sigmoid_prime(a1) # (60000, 128)
        delta1_weight = np.dot(x.T, error1) # (784, 128)
        
        return (delta1_weight, delta2_weight, delta3_weight)


    # Task 11: weight updates

    def weight_updates(self,
                    delta1_weight: np.ndarray,
                    delta2_weight: np.ndarray,
                    delta3_weight: np.ndarray,
                    learning_rate = 0.001,) -> None:
        """ weight updates
            using the gradient descent to update the weights
            w = w - learning_rate * delta
            learning_rate default is 0.001
        """
        self.weights['w1'] -= learning_rate * delta1_weight # (784, 128)
        self.weights['w2'] -= learning_rate * delta2_weight # (128, 64)
        self.weights['w3'] -= learning_rate * delta3_weight # (64, 10)

    # Task 12: computing error on test data

    def compute_error(self, x: np.ndarray, y: np.ndarray) -> float:
        """ computing error on test data 
            x: text data
            y: test label 
        """
        
        # a3(60000, 10) is the output of the network  
        a1, a2, a3 = self.forward_pass(x)

        # index set of the maximum value of the array
        x_test_max = np.argmax(a3, axis=1)
        y_test_max = np.argmax(y, axis=1)

        error_rate = np.mean(x_test_max != y_test_max)
        return error_rate


    # Task 13: error with initial weights
    
    def Task13_error_with_initial_weights(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
        """ error with initial weights
        """
        error_rate = self.compute_error(x_test, y_test)
        """ randonly test error rate
            error rate: 0.8865
            error rate: 0.9108
            error rate: 0.9118
            error rate: 0.8988
        """
        print(f"error rate test: {error_rate}")


    # Task 14-15: training

    def training(self, 
                x: np.ndarray,
                y: np.ndarray,
                test_input: np.ndarray,
                test_label: np.ndarray,
                epochs = 30,           # default epochs: 30
                learning_rate = 0.001  # default learning rate: 0.001
                ) -> None:
        """ training the network
        """
        start = time.time()
        for epoch in range(epochs):
            for x_batch, y_batch in zip(x, y):
                # reshape the input data to 2D array
                x_batch = x_batch.reshape(1, -1)
                y_batch = y_batch.reshape(1, -1)
                a1, a2, a3 = self.forward_pass(x_batch)
                delta1_weight, delta2_weight, delta3_weight = self.backpropagation(x_batch, y_batch, a1, a2, a3)
                self.weight_updates(delta1_weight, delta2_weight, delta3_weight, learning_rate)
            
            # compute the error rate
            error_rate = self.compute_error(test_input, test_label)

            print(f"epoch: {epoch} error rate: {error_rate}")
        print(f"Training time: {time.time() - start} For {epochs} epochs, traning size: {len(x)}, batch size: fix=1")


    # Task 16-18: batch training
    def batch_training(self, 
                    x: np.ndarray,
                    y: np.ndarray,
                    x_test: np.ndarray,
                    y_test: np.ndarray,
                    epochs = 30,
                    batch_size = 64,   # default batch size: 64. Try different values with 32, 128, 256
                    learning_rate = 0.001) -> bool:
        """ batch training
        """
        start = time.time()
        for epoch in range(epochs):
            for i in range(0, len(x), batch_size):
                x_batch = x[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                a1, a2, a3 = self.forward_pass(x_batch)
                delta1_weight, delta2_weight, delta3_weight = self.backpropagation(x_batch, y_batch, a1, a2, a3)
                self.weight_updates(delta1_weight, delta2_weight, delta3_weight, learning_rate)
            
            # compute the error rate
            error_rate = self.compute_error(x_test, y_test)
            print(f"epoch: {epoch} error rate: {error_rate}")

        print(f"Training time: {time.time() - start} For {epochs} epochs, traning size: {len(x)}, batch size: {batch_size}")

    def save_model(self, filename='model_weights.npz'):
        """ save model weights to local"""

        np.savez(filename, w1=self.weights['w1'], w2=self.weights['w2'], w3=self.weights['w3'])
        print(f"Model saved to {filename}")


# end of the NueralNetwork 

class HandWritingInputView:
    def __init__(self, model_path='./model_weights.npz'):
        # 加载已训练的模型
        print(f"********* HandWritingInputView *********")
        print(f"Model path: {model_path}")
        nn = NeuralNetwork(784, 128, 10, 0.001)
        nn.Task5_6_creating_and_initializing_matrices_of_weights()
        self.model = nn
        self.load_model(model_path)  # 直接加载到 self.model.weights 中

        self.draw_view()

    def load_model(self, filename):
        weights: np.ndarray = np.load(filename, allow_pickle=True)
    
        print(f" weights: {type(weights) },  lens = {len(weights)}")
        print(f"{weights['w1'].shape}")
        self.model.weights['w1'] = weights['w1']
        self.model.weights['w2'] = weights['w2']
        self.model.weights['w3'] = weights['w3']

    def draw_view(self):
        self.drawing = False
        self.image = np.zeros((200, 200))  # 创建一个200x200的空白画布
        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(self.image, cmap='gray',  vmin=0, vmax=1)
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.last_x, self.last_y = None, None  # 用于保存上一次的坐标

        # 创建按钮
        ax_clear = plt.axes([0.8, 0.05, 0.1, 0.075])  # 按钮位置，[left, bottom, width, height]
        self.btn_clear = Button(ax_clear, '清除')
        self.btn_clear.on_clicked(self.clear_canvas)

    def on_mouse_press(self, event):
        """鼠标按下事件，开始绘图"""
        if event.inaxes == self.ax:
            self.drawing = True
            self.last_x, self.last_y = int(event.xdata), int(event.ydata)
            self.draw(self.last_x, self.last_y)

    def on_mouse_release(self, event):
        """鼠标释放事件，结束绘图"""
        self.drawing = False
        self.last_x, self.last_y = None, None
        self.predict_digit() 

    def on_mouse_move(self, event):
        """鼠标移动事件，用于绘制连续的笔画"""
        if self.drawing and event.inaxes == self.ax:
            x, y = int(event.xdata), int(event.ydata)
            self.draw_line(self.last_x, self.last_y, x, y)
            self.last_x, self.last_y = x, y

    def draw(self, x, y):
        """在指定坐标绘制点"""
        self.image[y-2:y+3, x-2:x+3] = 1  # 使用5x5的方块
        self.im.set_data(self.image)
        self.fig.canvas.draw_idle()

    def draw_line(self, x0, y0, x1, y1):
        """在两个点之间插值绘制直线，保证平滑效果"""
        num_points = max(abs(x1 - x0), abs(y1 - y0))  # 计算插值点的数量
        x_points = np.linspace(x0, x1, num_points)
        y_points = np.linspace(y0, y1, num_points)
        
        for x, y in zip(x_points, y_points):
            self.draw(int(x), int(y))

    def clear_canvas(self, event):
        """清除画布"""
        self.image.fill(0)  # 将画布重置为全黑
        self.im.set_data(self.image)
        self.fig.canvas.draw_idle()  # 更新图像显示

    def start_drawing(self):
        plt.show()

    def predict_digit(self):
        """预测手绘数字"""
        # 将画布图像缩放到28x28
        resized_image = zoom(self.image, (28 / 200, 28 / 200))
        input_data = resized_image.reshape(1, 784)  # 展平成模型的输入格式
        prediction = self.model.predict(input_data)  # 调用 NeuralNetwork 的 predict 方法
        print(f":::::: =>>>>>  {prediction}")

#end of the HandWritingInputView class
# ________________________________________________________________________

local_model_path = './model_weights.npz'


# Step 1: Generate the model
if  1 == 0:
    """ Task 0 - 19"""
    nn = NeuralNetwork(784, 128, 10, 0.001)
    nn.Task1_reading_the_MNIST_files()
    nn.Task2_visualize_a_few_bitmap_images(nn.x_train)
    nn.Task3_input_preprocessing(nn.x_train, nn.x_test)
    nn.Task4_output_processing(nn.y_train, nn.y_test)
    nn.Task5_6_creating_and_initializing_matrices_of_weights()
    nn.Task13_error_with_initial_weights(nn.x_test, nn.y_test)

    # """ Task 14-15"""
    # trim = 100 
    # x_train = nn.x_train[:trim]
    # y_train = nn.y_train[:trim]
    # x_test  = nn.x_test[:trim]
    # y_test  = nn.y_test[:trim]
    # nn.training(x_train, y_train, 
    #             x_test, y_test,
    #             epochs=30, learning_rate=0.001)

    """ Task 16-18"""
    nn.batch_training(nn.x_train, nn.y_train, nn.x_test, nn.y_test, epochs=60, batch_size=64, learning_rate=0.001)
    nn.save_model(local_model_path)

# Step 2: Test the model with handerwritting input
if 1 == 1:
    """ Test the model with handerwritting input 
    1. Draw a digit on the canvas
    2. Press the button to predict the digit
    3. The prediction will be printed
    """
    drawer = HandWritingInputView(model_path=local_model_path)
    drawer.start_drawing()


