# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 17:06:55 2018
""" 

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import train_test_split 
import time

#Generate seed for training, testing, and validation selection
#seeds for test data selection
seed1 = [2861, 2723, 4436, 9064, 2428, 4748, 5665, 8987, 8088, 3465, 5998,
       4636, 7457, 5431, 5494, 9009,   34, 5037, 7155, 1323, 2156, 6835,
       2796, 8652, 9877, 3561, 3292, 4503, 9124, 4397]
#seeds for validation data selection
seed2 = [9284, 7802, 3369,  692, 2221, 6496, 2389, 6004, 3076, 3214, 9699,
       9189, 7811, 9127, 6113, 2676, 9032, 7712, 3312, 5200, 7831, 9576,
         75, 5590, 4562, 8011,  387, 3999, 9402, 4887]
tic = time.clock()

"""Define parameters"""
num_run = 1    # number of run 
truncated_backprop_length = 118    #number of previous values in the time series  
state_size = 40 # number of cells in the hidden layer 
num_classes = 1   # number of outputs
batch_size = 1
input_size = 6
num_layers = 4  #number of hidden layers
test_sample_size = .2 # percenetage testing data
validation_sample_size = 0.1 #percentage validation data
sample__stop_training = .1 # percentage of whole data which is selected from training dataset to compare with validation perfromance
num_batches = int((1-test_sample_size-validation_sample_size)*231752/(truncated_backprop_length*batch_size))-1
keep_rate = 0.9 #keeping rate in drop-out algorithm
inputs = input_size   #number of inputs
learning_rate = 0.001 #Optimizer's learning rate
stop_training_error_time = 1 #this parameter shows after how many not improving trainings the training will stop 

def generateData():
    """ Generate Training, testing, and validation data """
    input1 = []
    with open('IDM_calibration_all_lane_data1.csv', 'r') as csv_f:        
        data = csv.reader (csv_f) 
        for row in data:
            input1.append (row [0:6+num_classes])
    csv_f.close()
    input1 = np.array(input1)
    extractedData = input1[:,[0, 1, 2, 3, 4, 6, 5]]
    # String to float all data and remove columns' titles from the data
    input11 = []
    for i in range(1, len(extractedData)):   
        input11.append([])
        for j in range(0, inputs+num_classes):
            input11[i-1].append(float(extractedData[i][j]))
    input2 = np.array(input11)
    
    # Create Batches
    x_data = input2[:(len(input2)-(len(input2) % (truncated_backprop_length)))]
    x1 = x_data.reshape((batch_size, -1, truncated_backprop_length, inputs+num_classes))    
    x1 = x1.transpose(1,0,2,3) 
    
    TrainData, TestData =train_test_split(x1, test_size=test_sample_size, random_state = seed1[item])
    TrainData, ValidationData =train_test_split(TrainData, test_size=validation_sample_size/(1-test_sample_size), random_state = seed2[item])
    
    TrainData = TrainData.transpose(1,0,2,3)
    TestData = TestData.transpose(1,0,2,3)
    ValidationData = ValidationData.transpose(1,0,2,3)
    
    # Find min and max values at each column for training dataset
    TrainData1 = TrainData.reshape((1, -1, inputs+num_classes))  
    col_min = TrainData1.min(axis=(1), keepdims=True)
    col_max = TrainData1.max(axis=(1), keepdims=True)
    
    col_min = col_min - (col_max - col_min)*0.1
    col_max = col_max + (col_max - col_min)*0.1
    
    # Normalize training dataset using min-max approach
    TrainDataN = (TrainData1 - col_min +.00000000001)/ (col_max - col_min +.00000000002)
    
    # Normalize validation dataset with training's min and max values 
    ValidationData1 = ValidationData.reshape((1, -1, inputs+num_classes))
    ValidationDataN = (ValidationData1 - col_min +.00000000001)/ (col_max - col_min +.00000000002)    
    
    TestData1 = TestData.reshape((1, -1, inputs+num_classes))
    TestDataN = (TestData1 - col_min +.00000000001)/ (col_max - col_min +.00000000002) 
    
    # Assign input and output data
    TrainData_x = TrainDataN[:,:,0:inputs]
    TrainData_y = TrainDataN[:,:,inputs:inputs+num_classes]
    TrainData_y = TrainData_y.reshape((batch_size, -1))
    
    ValidationData_x = ValidationDataN[:,:,0:inputs]
    ValidationData_y = ValidationDataN[:,:,inputs:inputs+num_classes]
    ValidationData_y = ValidationData_y.reshape((batch_size, -1))
    
    TestData_x = TestDataN[:,:,0:inputs]
    TestData_y = TestDataN[:,:,inputs:inputs+num_classes] 
    TestData_y = TestData_y.reshape((batch_size, -1))
    
    return (TrainData_x, TrainData_y,ValidationData_x, ValidationData_y, TestData_x, TestData_y, col_min, col_max)

"""Plot training results"""
def plot(loss_list, _predictions_series, batchY):
    fig = plt.figure(figsize=(12,6))
    fig.add_subplot(1, 2, 1)
    # plot training progress
    plt.cla()
    plt.plot(loss_list)
    # Plot a batch of training data
    for batch_series_idx in range(batch_size):
        one_hot_output_series = np.array(_predictions_series)[:, batch_series_idx, :]
        single_output_series = np.array([(out[0]) for out in one_hot_output_series])

        fig.add_subplot(1, 2, batch_series_idx + 2)
        plt.cla()
        #plt.axis([0, truncated_backprop_length, 0, 1])
        batchY = batchY.reshape((truncated_backprop_length))
        left_offset = range(truncated_backprop_length)
        plt.scatter(left_offset, (batchY), s=30, c="red", marker="s", alpha=0.5)
        plt.scatter(left_offset, (single_output_series), s=30, c="green", marker="s", alpha=0.3)
    plt.draw()
    plt.pause(0.01)

"""Plot testing or validation results"""    
def plottest(_predictions_series, batchYT):
    
    fig = plt.figure(figsize=(5,4))
    for batch_series_idx in range(batch_size):
        one_hot_output_series1 = np.array(_predictions_series)[:, batch_series_idx]
        single_output_series1 = np.array([(out[0]) for out in one_hot_output_series1])
        
        fig.add_subplot(1, 1, batch_series_idx+1)
        plt.cla()
        #plt.axis([0, truncated_backprop_length, 0, 1])
        batchYT = batchYT.reshape((truncated_backprop_length))
        left_offset1 = range(truncated_backprop_length)
        plt.scatter(left_offset1, (batchYT), s=30, c="red", marker="s", alpha=0.5)
        plt.scatter(left_offset1, (single_output_series1), s=30, c="green", marker="s", alpha=0.3)

    plt.draw()
    plt.pause(0.0001)
    
def deep_learning_model():
    tf.reset_default_graph()  # rest all graphs
    
    batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length, input_size])
    batchY_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
    
    keep_prob = tf.placeholder(tf.float32)
    time_step = tf.placeholder(tf.float32)
    col_min_holder = tf.placeholder(tf.float32, [None, 1, inputs+num_classes])  # keeps columns min for denormalization
    col_max_holder = tf.placeholder(tf.float32, [None, 1, inputs+num_classes])  # keeps columns max for denormalization
    
    # Create cells' status
    init_state = tf.placeholder(tf.float32, [num_layers, 2, batch_size, state_size])  
    state_per_layer_list = tf.unstack(init_state, axis=0)
    rnn_tuple_state = tuple(
          [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0,:,:], state_per_layer_list[idx][1,:,:])
          for idx in range(num_layers)])
    # Weights and biases  
    W2 = tf.Variable(np.zeros((state_size, num_classes)),dtype=tf.float32)
    b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)
    
    # define cells' structure
    stacked_rnn = []
    for _ in range(num_layers):
        stacked_rnn.append(tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True))
        cell = tf.nn.rnn_cell.MultiRNNCell(stacked_rnn, state_is_tuple=True)
    states_series, current_state = tf.nn.dynamic_rnn(cell, batchX_placeholder, initial_state=rnn_tuple_state)
    states_series = tf.reshape(states_series, [-1, state_size])
    # generate predition
    logits = tf.matmul(states_series, W2) + b2
    predictions_series = tf.sigmoid(logits)  
    
    
    acel = tf.transpose(tf.nn.embedding_lookup(tf.transpose(batchX_placeholder),[2])) # extract accleration rate
    acel_min = tf.nn.embedding_lookup(tf.transpose(col_min_holder),[2])
    acel_max = tf.nn.embedding_lookup(tf.transpose(col_max_holder),[2])
    acel1 = tf.reshape(tf.add(tf.multiply(acel, tf.subtract(acel_max,acel_min)), acel_min), [batch_size, truncated_backprop_length, -1]) # denormalize accleration  
    
    location = tf.transpose(tf.nn.embedding_lookup(tf.transpose(batchX_placeholder),[0])) # extract subject vehicle's location
    loc_min = tf.nn.embedding_lookup(tf.transpose(col_min_holder),[0])
    loc_max = tf.nn.embedding_lookup(tf.transpose(col_max_holder),[0])
    location1 = tf.reshape(tf.add(tf.multiply(location, tf.subtract(loc_max,loc_min)), loc_min), [batch_size, truncated_backprop_length, -1]) # denormalize subject vehicle's location

    velo = tf.transpose(tf.nn.embedding_lookup(tf.transpose(batchX_placeholder),[1])) # extract subject vehicle's velocity
    velo_min = tf.nn.embedding_lookup(tf.transpose(col_min_holder),[1])
    velo_max = tf.nn.embedding_lookup(tf.transpose(col_max_holder),[1])
    velo1 = tf.reshape(tf.add(tf.multiply(velo, tf.subtract(velo_max,velo_min)), velo_min), [batch_size, truncated_backprop_length, -1]) # denormalize subject vehicle's velocity
    
    leadH = tf.transpose(tf.nn.embedding_lookup(tf.transpose(batchX_placeholder),[5])) # extract leading vehicle's location at next time step with headway
    leadH_min = tf.nn.embedding_lookup(tf.transpose(col_min_holder),[5])
    leadH_max = tf.nn.embedding_lookup(tf.transpose(col_max_holder),[5])
    leadH1 = tf.reshape(tf.add(tf.multiply(leadH, tf.subtract(leadH_max,leadH_min)), leadH_min), [batch_size, truncated_backprop_length, -1]) # denormalize leading vehicle's location at next time step with headway
    
    target = tf.reshape (batchY_placeholder, [batch_size, truncated_backprop_length, -1]) # extract leading vehicle's headway at next time step 
    target_min = tf.nn.embedding_lookup(tf.transpose(col_min_holder),[6])
    target_max = tf.nn.embedding_lookup(tf.transpose(col_max_holder),[6])
    target1 = tf.add(tf.multiply(target, tf.subtract(target_max,target_min)), target_min)
    
    predict_accl = tf.add(tf.multiply(predictions_series, tf.subtract(acel_max,acel_min)), acel_min)
    predict_head = tf.subtract (leadH1, tf.add(location1, tf.add(tf.multiply(velo1,time_step),tf.multiply(predict_accl,tf.multiply(time_step, time_step)))))
    
    predict_head_N = tf.divide (tf.subtract( predict_head,target_min), tf.subtract( target_max,target_min))
    predict_head_N = tf.reshape(predict_head_N, [truncated_backprop_length*batch_size, num_classes])
    
    # Reshape training output
    labels = tf.reshape(batchY_placeholder,[batch_size*truncated_backprop_length, num_classes])
    # Estimate error and run optimizer (gradient descent)

    loss = tf.reduce_mean(tf.divide(tf.abs(tf.subtract(predict_head_N,labels)),(labels)))
    predictions_series = tf.unstack(tf.reshape(predict_head_N, [batch_size, truncated_backprop_length, num_classes]), axis=1) 
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    """Start session and feed the network with data"""
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        plt.ion()
        plt.figure()
        plt.show()
        ave_loss = 1
        stop_training = 0
        pervious_validation = 1
        previous_sum_loss_10 = 1
    
        while stop_training < stop_training_error_time:
            loss_list = []
            _current_state = np.zeros((num_layers, 2, batch_size, state_size))
            sum_loss= 0
            sum_loss_10 = 0
            for batch_idx in range(num_batches):
                start_idx = batch_idx * truncated_backprop_length
                end_idx = start_idx + truncated_backprop_length
    
                batchX = x[:,start_idx:end_idx,:]
                batchY = y[:,start_idx:end_idx]
    
                _total_loss, _train_step, _current_state, _predictions_series, ttdoyou = sess.run(
                [loss, train_step, current_state, predictions_series, predict_head_N],
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
                    print('This run s %headway error = %', "%.7f" % (_total_loss*100) )
                    ave_loss= sum_loss/num_batches
                    print ('Average training %headway error %', "%.7f" % (ave_loss*100))
                    """ Calculate accuracy of the model for validation dataset"""
                    Ave_loss_validation= 0
                    loss_listV = []
                    for batch_idx in range(int(len(xv[0])/truncated_backprop_length)):
                        start_idx = batch_idx * truncated_backprop_length
                        end_idx = start_idx + truncated_backprop_length
                
                        batchXV = xv[:,start_idx:end_idx,:]
                        batchYV = yv[:,start_idx:end_idx]
                        __current_state, __predictions_series = sess.run(
                                [current_state, predictions_series],
                                feed_dict={
                                batchX_placeholder: batchXV,
                                init_state: _current_state,
                                time_step: 0.1,
                                col_min_holder: col_min,
                                col_max_holder: col_max
                                })
    
                        if (batch_idx == int(len(xv[0])/truncated_backprop_length)-2):
                            plottest(__predictions_series, batchYV)
                        # Calculate MSE for validation data
                        ttt = []
                        for batch_series_idx in range(batch_size):
                            one_hot_output_series = np.array(__predictions_series)[:, batch_series_idx, :]
                            single_output_series = np.array([(out[0]) for out in one_hot_output_series])
                            ttt.extend(single_output_series)
                        ttt = np.array(ttt)
                        batchYV = batchYV.reshape((truncated_backprop_length*batch_size))
                        validation_loss = np.mean(np.abs((batchYV- ttt) / batchYV))
                        loss_listV.append(validation_loss)
                        Ave_loss_validation = Ave_loss_validation + validation_loss
                    validation_loss1 = Ave_loss_validation/(int(len(xv[0])/truncated_backprop_length))
                    
                    print('Validation average %headway error is %', "%.7f" % (validation_loss*100) )
                    
                    # Check if the model trained well enough
                    if (validation_loss1 < 0.001):
                        # First stop condition
                        if ((pervious_validation - validation_loss1)< 0.000001):
                            stop_training = stop_training + 1
                    pervious_validation = validation_loss1

        lost_train.append(loss_list)
        lost_validate.append(loss_listV)
        """Test the model"""
        Ave_loss_test= 0
        loss_listT = []
        test_headway = []
        for batch_idx in range(int(len(xt[0])/truncated_backprop_length)):
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length
    
            batchXT = xt[:,start_idx:end_idx,:]
            batchYT = yt[:,start_idx:end_idx]
            _current_state, _predictions_series = sess.run(
                    [current_state, predictions_series],
                    feed_dict={
                                    batchX_placeholder: batchXT,
                                    init_state: _current_state,
                                    time_step: 0.1,
                                    col_min_holder: col_min,
                                    col_max_holder: col_max
                                    })
            
            """ Calculate MSE of the model for test dataset"""
            ttt = []
            for batch_series_idx in range(batch_size):
                one_hot_output_series = np.array(_predictions_series)[:, batch_series_idx, :]
                single_output_series = np.array([(out[0]) for out in one_hot_output_series])
                ttt.extend(single_output_series)
            ttt = np.array(ttt)
            batchYT1 = batchYT.reshape((truncated_backprop_length*batch_size))
            test_loss = np.mean(np.abs((batchYT1 - ttt) / batchYT1))
            loss_listT.append(test_loss)
            Ave_loss_test = Ave_loss_test + test_loss
            test_headway.append(ttt)
            #Plot test data once in 20 batches
            if (batch_idx%50 == 1):
                plottest(_predictions_series, batchYT)
                print('This batch s %headway error = ', "%.7f" % (test_loss*100)) 
        test_loss = Ave_loss_test/(int(len(xt[0])/truncated_backprop_length))
        print('Test average %headway error = %', "%.7f" % (test_loss*100))
        lost_test.append(loss_listT)
    plt.ioff()
    plt.show()
    return(test_headway, lost_train, lost_validate, lost_test)
    
def prepare_results


"Rest data gathering lists"""
lost_train = []
lost_test = []
lost_validate = []

"""Run the model"""
for item in range(num_run):
    print("Run Number", item+1)
    x,y, xv, yv, xt, yt, col_min, col_max = generateData()
    test_headway, lost_train, lost_validate, lost_test = deep_learning_model()
    
prepare_results 
print("All runs testing average %headway error %", "%.7f" % (np.mean(lost_test)*100))
print("run time", "%.0f" %  (time.clock() - tic)) 
