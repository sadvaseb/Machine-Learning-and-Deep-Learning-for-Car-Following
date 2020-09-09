"""
Created on Thu Mar  8 02:56:46 2018
this file is a single layer RNN for Car following, could be used for leading vehicle or all surrounding vehicles.
This code runs the RNN several times and print MSE for each run. 
This code attempts to minimize % headway error
"""
import tensorflow as tf
from tensorflow.contrib import rnn 
import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import random
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn.python.learn import learn_runner
import tensorflow.contrib.metrics as metrics
import tensorflow.contrib.rnn as rnn
import sklearn.metrics as sm
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time


#generate seed for training, testing, and validation selection

#seeds for test data selection
seed1 = [2861, 2723, 4436, 9064, 2428, 4748, 5665, 8987, 8088, 3465, 5998,
       4636, 7457, 5431, 5494, 9009,   34, 5037, 7155, 1323, 2156, 6835,
       2796, 8652, 9877, 3561, 3292, 4503, 9124, 4397] 
#seeds for validation data selection
seed2 = [9284, 7802, 3369,  692, 2221, 6496, 2389, 6004, 3076, 3214, 9699,
       9189, 7811, 9127, 6113, 2676, 9032, 7712, 3312, 5200, 7831, 9576,
         75, 5590, 4562, 8011,  387, 3999, 9402, 4887]
tic = time.clock()

num_run = 2    # number of run   
state_size = 50 # number of cells in the hidden layer 
outputs = 1            #number of outputs
num_classes = outputs   # number of outputs
batch_size = 118  #In this code refers to length of back propagation
input_size = 6 
test_sample_size = .2 # percenetage testing data
validation_sample_size = 0.1 #percentage validation data
sample__stop_training = .1 # percentage of whole data which is selected from training dataset to compare with validation perfromance
num_batches = int((1-test_sample_size-validation_sample_size)*231752/(batch_size))-1 ###############238360
keep_rate = 0.8 #keeping rate in drop-out algorithm

inputs = input_size          #number of inputs
learning_rate = 0.0001 #Optimizer's learning rate
stop_training_error_time = 1 #this parameter shows after how many not improving trainings the training will stop 

def generateData():
    """ Generate Training, testing, and validation data """
    input1 = []
    with open('C:/Users/Sertab Gamma/Dropbox/Saeed Vasebi/2019/2019.04.30- IDM calibration/IDM_calibration_all_lane_data1.csv', 'r') as csv_f:        
        data = csv.reader (csv_f) 
        for row in data:
            input1.append (row [0:6+num_classes])
    csv_f.close()
    input1 = np.array(input1)
    """Select which parameters should be considered"""
    """Number of inputs in line 26 should be checked"""
    extractedData = input1[:,[0, 1, 2, 3, 4, 6, 5]]
    # String to float all data and remove columns' titles from the data
    input11 = []
    for i in range(1, len(extractedData)):   
        input11.append([])
        for j in range(0, inputs+num_classes):
            input11[i-1].append(float(extractedData[i][j]))
    input2 = np.array(input11)
    
    # Create Batches
    x_data = input2[:(len(input2)-(len(input2) % (batch_size)))]
    x1 = x_data.reshape((-1, batch_size, inputs+num_classes))    
    #TrainData = TrainData.transpose(1,0,2) 
    
    TrainData, TestData =train_test_split(x1, test_size=test_sample_size, random_state = seed1[item])
    TrainData, ValidationData =train_test_split(TrainData, test_size=validation_sample_size/(1-test_sample_size), random_state = seed2[item])
    
    
    # Find min and max values at each column for training dataset
    TrainData1 = TrainData.reshape((1, -1, inputs+num_classes))  
    col_min = TrainData1.min(axis=(1), keepdims=True)
    col_max = TrainData1.max(axis=(1), keepdims=True)
    """We will activate this weight adjuctment later """
    col_min = col_min - (col_max - col_min)*0.1
    col_max = col_max + (col_max - col_min)*0.1
    
    # Normalize training dataset using min-max approach
    TrainDataN = (TrainData1 - col_min +.000000000001)/ (col_max - col_min +.000000000002)
    
    # Normalize validation dataset with training's min and max values 
    ValidationData1 = ValidationData.reshape((1, -1, inputs+num_classes))
    ValidationDataN = (ValidationData1 - col_min +.000000000001)/ (col_max - col_min +.000000000002)    
    
    TestData1 = TestData.reshape((1, -1, inputs+num_classes))
    TestDataN = (TestData1 - col_min +.000000000001)/ (col_max - col_min +.000000000002) 
    
    # Assign input and output data
    TrainData_x = TrainDataN[:,:,0:inputs]
    TrainData_y = TrainDataN[:,:,inputs:inputs+num_classes]
    
    ValidationData_x = ValidationDataN[:,:,0:inputs]
    ValidationData_y = ValidationDataN[:,:,inputs:inputs+num_classes]
    
    TestData_x = TestDataN[:,:,0:inputs]
    TestData_y = TestDataN[:,:,inputs:inputs+num_classes] 
    
    return (TrainData_x, TrainData_y,ValidationData_x, ValidationData_y, TestData_x, TestData_y, col_min, col_max)

"""Plot training results"""
def plot(loss_list, _predictions_series, batchY):
    fig = plt.figure(figsize=(12,6))
    fig.add_subplot(1, 2, 1)
    # plot training progress
    plt.cla()
    plt.plot(loss_list)
    # Plot a batch of training data
    ttt = []
    for batch_series_idx in range(batch_size):
        one_hot_output_series = np.array(_predictions_series)[:, batch_series_idx, :]
        single_output_series = np.array([(out[0]) for out in one_hot_output_series])
        ttt.extend(single_output_series)
    ttt = np.array(ttt)
    batchY = batchY.reshape((batch_size))

    fig.add_subplot(1, 2, 2)
    plt.cla()
    #plt.axis([0, batch_size, 0, 1])
    left_offset = range(batch_size)
    plt.scatter(left_offset, (batchY), s=30, c="red", marker="s", alpha=0.3)
    plt.scatter(left_offset, (ttt), s=30, c="green", marker="s", alpha=0.3)
    plt.draw()
    plt.pause(0.0001)

"""Plot testing or validation results"""    
def plottest(_predictions_series, batchYT):
    ttt = []
    fig = plt.figure(figsize=(5,4))
    for batch_series_idx in range(batch_size):
        one_hot_output_series1 = np.array(_predictions_series)[:, batch_series_idx]
        single_output_series1 = np.array([(out[0]) for out in one_hot_output_series1])
        ttt.extend(single_output_series1)
    ttt = np.array(ttt)
    batchYT = batchYT.reshape((batch_size))
        
    fig.add_subplot(1, 1, 1)
    plt.cla()
    #plt.axis([0, truncated_backprop_length, 0, 1])
    left_offset1 = range(batch_size)
    plt.scatter(left_offset1, (batchYT), s=30, c="red", marker="s", alpha=0.3)
    plt.scatter(left_offset1, (ttt), s=30, c="green", marker="s", alpha=0.3)

    plt.draw()
    plt.pause(0.0001)

lost_train = []
lost_test = []
lost_validate = []
lost_test_line = []    

# Run the RNN 6 times and create 6 models
for item in range(num_run):
    # Configure RNN network
    tf.reset_default_graph()   # Reset all previous graphs
    
    #create variables' place holder
    batchX_placeholder = tf.placeholder(tf.float32, [None, batch_size, inputs])   #create RNN cells with softsign activation function
    batchY_placeholder = tf.placeholder(tf.float32, [None, batch_size, outputs])   #Create the graph with RNN cells
    keep_prob = tf.placeholder(tf.float32)
    time_step = tf.placeholder(tf.float32)
    col_min_holder = tf.placeholder(tf.float32, [1, 1, inputs+outputs])  # keeps columns min for denormalization
    col_max_holder = tf.placeholder(tf.float32, [1, 1, inputs+outputs])  # keeps columns max for denormalization
    
    
    init_state = tf.placeholder(tf.float32, [ 2, None, state_size])
    #init_state = tf.Variable(np.zeros((1,2, 1, state_size)), dtype=tf.float32)
    #l = tf.unstack(init_state, axis=0)

    
    #init_state = (tf.zeros([2, 1,state_size]),)
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(state_size, state_is_tuple=True)
    # Initial state of the LSTM memory.
    W2 = tf.Variable(np.zeros((state_size, num_classes)),dtype=tf.float32)
    b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)

    rnn_tuple_state = tf.nn.rnn_cell.LSTMStateTuple(init_state[0],init_state[1])
    
    # 'state' is a tensor of shape [batch_size, cell_state_size]
    output1, current_state = tf.nn.dynamic_rnn(lstm_cell, batchX_placeholder, initial_state=rnn_tuple_state)
            
    output1 = tf.reshape(output1, [-1, state_size])
    # generate predition
    logits = tf.matmul(output1, W2) + b2
    predictions_series = tf.sigmoid(logits)  
        
    outputRNN = tf.reshape(predictions_series, [-1, batch_size, outputs])         #change shape of stacked_outputs

    acel = tf.transpose(tf.nn.embedding_lookup(tf.transpose(batchX_placeholder),[2])) # extract accleration rate
    acel_min = tf.nn.embedding_lookup(tf.transpose(col_min_holder),[2])
    acel_max = tf.nn.embedding_lookup(tf.transpose(col_max_holder),[2])
    #acel1 = tf.reshape(tf.add(tf.multiply(acel, tf.subtract(acel_max,acel_min)), acel_min), [-1, batch_size, outputs]) # denormalize accleration
    acel1 = tf.add(tf.multiply(acel, tf.subtract(acel_max,acel_min)), acel_min)
    
    location = tf.transpose(tf.nn.embedding_lookup(tf.transpose(batchX_placeholder),[0])) # extract subject vehicle's location
    loc_min = tf.nn.embedding_lookup(tf.transpose(col_min_holder),[0])
    loc_max = tf.nn.embedding_lookup(tf.transpose(col_max_holder),[0])
    #location1 = tf.reshape(tf.add(tf.multiply(location, tf.subtract(loc_max,loc_min)), loc_min), [-1, batch_size, outputs]) # denormalize subject vehicle's location
    location1 = tf.add(tf.multiply(location, tf.subtract(loc_max,loc_min)), loc_min) # denormalize subject vehicle's location
    
    velo = tf.transpose(tf.nn.embedding_lookup(tf.transpose(batchX_placeholder),[1])) # extract subject vehicle's velocity
    velo_min = tf.nn.embedding_lookup(tf.transpose(col_min_holder),[1])
    velo_max = tf.nn.embedding_lookup(tf.transpose(col_max_holder),[1])
    #velo1 = tf.reshape(tf.add(tf.multiply(velo, tf.subtract(velo_max,velo_min)), velo_min), [-1, batch_size, outputs]) # denormalize subject vehicle's velocity
    velo1 = tf.add(tf.multiply(velo, tf.subtract(velo_max,velo_min)), velo_min)# denormalize subject vehicle's velocity
    
    leadH = tf.transpose(tf.nn.embedding_lookup(tf.transpose(batchX_placeholder),[5])) # extract leading vehicle's location at next time step with headway
    leadH_min = tf.nn.embedding_lookup(tf.transpose(col_min_holder),[5])
    leadH_max = tf.nn.embedding_lookup(tf.transpose(col_max_holder),[5])
    #leadH1 = tf.reshape(tf.add(tf.multiply(leadH, tf.subtract(leadH_max,leadH_min)), leadH_min), [-1, batch_size, outputs]) # denormalize leading vehicle's location at next time step with headway
    leadH1 = tf.add(tf.multiply(leadH, tf.subtract(leadH_max,leadH_min)), leadH_min)
    
    target = tf.transpose(tf.nn.embedding_lookup(tf.transpose(batchY_placeholder),[0])) # extract leading vehicle's headway at next time step 
    target_min = tf.nn.embedding_lookup(tf.transpose(col_min_holder),[6])
    target_max = tf.nn.embedding_lookup(tf.transpose(col_max_holder),[6])
    #target1 = tf.reshape(tf.add(tf.multiply(target, tf.subtract(target_max,target_min)), target_min), [-1, batch_size, outputs]) # denormalize leading vehicle's headway at next time step 
    target1 = tf.add(tf.multiply(target, tf.subtract(target_max,target_min)), target_min)
    
    #predict_accl = tf.reshape(tf.add(tf.multiply(outputRNN, tf.subtract(acel_max,acel_min)), acel_min), [-1, batch_size, outputs]) # denormalize predicted accleration at next time step 
    predict_accl = tf.add(tf.multiply(outputRNN, tf.subtract(acel_max,acel_min)), acel_min)
    #predict_head = tf.reshape(tf.subtract (leadH1, tf.add(location1, tf.add(tf.multiply(velo1,time_step),tf.multiply(predict_accl,tf.multiply(time_step, time_step))))) , [-1, batch_size, outputs]) # H1 = lead_Y1 - (Subject_Y0 + V0*t + a1*t^2)
    predict_head = tf.subtract (leadH1, tf.add(location1, tf.add(tf.multiply(velo1,time_step),tf.multiply(predict_accl,tf.multiply(time_step, time_step)))))
    
    predict_head_N = tf.divide (tf.subtract( predict_head,target_min), tf.subtract( target_max,target_min))
    
    #loss = tf.losses.mean_squared_error(outputRNN, batchY_placeholder) 
    loss = tf.reduce_mean(tf.divide(tf.abs(tf.subtract(predict_head_N,batchY_placeholder)),(batchY_placeholder)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)          #Gradient descent optimization by Adam optimizer
    training_op = optimizer.minimize(loss)          #Minimize MSE by the optimization function   

    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())    #initialize all the random variables
        stop_training = 0
        pervious_validation = 1
        x,y, xv, yv, xt, yt, col_min, col_max = generateData()
    
        while stop_training < stop_training_error_time:
            """Start training RNN"""
            loss_list = []
            _current_state = np.zeros((2, 1, state_size))
            sum_loss= 0
            for batch_idx in range(num_batches):
                
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
    
                batchX = x[:, start_idx:end_idx,:]
                batchY = y[:, start_idx:end_idx]
    
                _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                [loss, training_op, current_state, predict_head_N],
                feed_dict={
                                batchX_placeholder: batchX,
                                batchY_placeholder: batchY,
                                init_state: _current_state,
                                keep_prob: keep_rate,
                                time_step: 0.1,
                                col_min_holder: col_min,
                                col_max_holder: col_max
                                })
        
                loss_list.append(_total_loss)
                sum_loss = sum_loss + _total_loss   
                              
                #Plot training result
                if (batch_idx == num_batches-1):
                    plot(loss_list, _predictions_series, batchY)                    
                    print('This run s %headway error =', "%.7f" % (_total_loss*100))
                    ave_loss= sum_loss/num_batches
                    print ('Average training %headway error', "%.7f" % (ave_loss*100))
            """ Calculate accuracy of the model for validation dataset"""
            Ave_loss_validation= 0
            loss_listV = []
            for batch_idx in range(int(len(xv[0])/batch_size)):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
        
                batchXV = xv[:,start_idx:end_idx,:]
                batchYV = yv[:,start_idx:end_idx]
                _current_state, __predictions_series = sess.run(
                        [current_state, predict_head_N],
                        feed_dict={
                        batchX_placeholder: batchXV,
                        init_state: _current_state,
                        keep_prob: keep_rate,
                        time_step: 0.1,
                        col_min_holder: col_min,
                        col_max_holder: col_max
                        })
                if (batch_idx == int(len(xv[0])/batch_size)-2):
                    plottest(__predictions_series, batchYV)
                # Calculate MSE for validation data   
                ttt = []
                for batch_series_idx in range(batch_size):
                    one_hot_output_series = np.array(__predictions_series)[:,batch_series_idx, :]
                    single_output_series = np.array([(out[0]) for out in one_hot_output_series])
                    ttt.extend(single_output_series)
                ttt = np.array(ttt)
                batchYV = batchYV.reshape((batch_size))
                
                #validation_loss = mean_squared_error(batchYV,ttt)   
                validation_loss = np.mean(np.abs((batchYV - ttt) / batchYV))
                loss_listV.append(validation_loss)
                Ave_loss_validation = Ave_loss_validation + validation_loss
            validation_loss = Ave_loss_validation/(int(len(xv[0])/batch_size))
            
            print('Validation average %headway error is', "%.7f" % (validation_loss*100))
                        
            """Is the model overfitted?"""
            # Check if the model trained well enough
            if (validation_loss < 0.001):
                # First stop condition
                if ((pervious_validation - validation_loss)< 0.00001):
                    stop_training = stop_training + 1
                    print ('1. Reason for stop is validation does not improve')
            pervious_validation = validation_loss
    
            lost_train.append(loss_list)
            lost_validate.append(loss_listV)
        """Start test the model"""
        Ave_loss_test= 0
        loss_listT = []
        test_acceleration = []
        for batch_idx in range(int(len(xt[0])/batch_size)):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
    
            batchXT = xt[:,start_idx:end_idx,:]
            batchYT = yt[:,start_idx:end_idx]
            _current_state, _predictions_series = sess.run(
                    [current_state, predict_head_N],
                    feed_dict={
                                    batchX_placeholder: batchXT,
                                    init_state: _current_state,
                                    keep_prob: keep_rate,
                                    time_step: 0.1,
                                    col_min_holder: col_min,
                                    col_max_holder: col_max
                                    })
            """ Calculate % Error of the model for test dataset"""
            ttt = []
            for batch_series_idx in range(batch_size):
                one_hot_output_series = np.array(_predictions_series)[:, batch_series_idx, :]
                single_output_series = np.array([(out[0]) for out in one_hot_output_series])
                ttt.extend(single_output_series)
            ttt = np.array(ttt)
            batchYT1 = batchYT.reshape((batch_size))
            #test_loss = mean_squared_error(batchYT1,ttt) 
            test_loss = np.mean(np.abs((batchYT1 - ttt) / batchYT1))
            loss_listT.append(test_loss)
            Ave_loss_test = Ave_loss_test + test_loss
            test_acceleration.append(ttt)
            #Plot test data once in 20 batches
            if (batch_idx%50 == 1):
                plottest(_predictions_series,batchYT1)
                print('This batch s %headway error = ', "%.7f" % (test_loss*100))
        test_loss = Ave_loss_test/(int(len(xt[0])/batch_size))
        print ('********************************')
        print('Test average %headway error = ', "%.7f" % (test_loss*100))
        print ('')
        lost_test.append(loss_listT)
    plt.ioff()
    plt.show()

lost_train = np.array(lost_train)
lost_validate = np.array(lost_validate)
lost_test = np.array(lost_test)
lost_test_line = lost_test.reshape((1,-1))
test_acceleration = np.array(test_acceleration)
test_acceleration = test_acceleration.reshape((1,-1))
test_acceleration = test_acceleration * (col_max[0,0,2]-col_max[0,0,2]) + col_max[0,0,2]
print('=================================')
print('=================================')
print("All runs testing average %headway error", "%.7f" % (np.mean(lost_test)*100))
print("run time", "%.0f" %  (time.clock() - tic)) 
print ('********************************')

plot_prediction = _predictions_series*(col_max[:,:,input_size]-col_min[:,:,input_size])+ col_min[:,:,input_size]
plot_actual = batchYT1*(col_max[:,:,input_size]-col_min[:,:,input_size])+ col_min[:,:,input_size]