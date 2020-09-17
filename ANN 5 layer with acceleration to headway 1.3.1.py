# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import rnn 
import numpy as np
import matplotlib.pyplot as plt
import random
import csv
import time
from sklearn.model_selection import train_test_split

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



n_nodes_hl1 = 50
n_nodes_hl2 = 100
n_nodes_hl3 = 150   # num_hidden
n_nodes_hl4 = 80
n_nodes_hl5 = 40


num_run = 5                # number of run   
state_size = n_nodes_hl1   # number of cells in the hidden layer 
outputs = 1                # number of outputs
num_classes = outputs      # number of outputs
batch_size = 118           # In this code refers to length of back propagation
input_size = 6 
test_sample_size = .2        # percenetage testing data
validation_sample_size = 0.1 # percentage validation data
sample__stop_training = .1   # percentage of whole data which is selected from training dataset to compare with validation perfromance
num_batches = int((1-test_sample_size-validation_sample_size)*231752/(batch_size))-1 
keep_rate = 0.9    #   keeping rate in drop-out algorithm

inputs = input_size          # number of inputs
learning_rate = 0.0001       # Optimizer's learning rate
stop_training_error_time = 1 # this parameter shows after how many not improving trainings the training will stop 

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
    x_data = input2[:(len(input2)-(len(input2) % (batch_size)))]
    x1 = x_data.reshape((-1, batch_size, inputs+num_classes))    
    
    TrainData, TestData =train_test_split(x1, test_size=test_sample_size, random_state = seed1[item])
    TrainData, ValidationData =train_test_split(TrainData, test_size=validation_sample_size/(1-test_sample_size), random_state = seed2[item])
    
    
    # Find min and max values at each column for training dataset
    TrainData1 = TrainData.reshape((1, -1, inputs+num_classes))  
    col_min = TrainData1.min(axis=(1), keepdims=True)
    col_max = TrainData1.max(axis=(1), keepdims=True)
    
    col_min = col_min - (col_max - col_min)*0.1
    col_max = col_max + (col_max - col_min)*0.1
    
    # Normalize training dataset using min-max approach
    TrainDataN = (TrainData1 - col_min +.000000001)/ (col_max - col_min +.000000002)
    
    # Normalize validation dataset with training's min and max values 
    ValidationData1 = ValidationData.reshape((1, -1, inputs+num_classes))
    ValidationDataN = (ValidationData1 - col_min +.000000001)/ (col_max - col_min +.000000002)    
    
    TestData1 = TestData.reshape((1, -1, inputs+num_classes))
    TestDataN = (TestData1 - col_min +.000000001)/ (col_max - col_min +.000000002) 
    
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
    plt.scatter(left_offset, (batchY), s=30, c="red", marker="s", alpha=0.5)
    plt.scatter(left_offset, (ttt), s=30, c="green", marker="s", alpha=0.3)
    plt.draw()
    plt.pause(0.0001)

"""Plot testing or validation results"""    
def plottest(_predictions_series, batchYT):
    ttt = []
    fig = plt.figure(figsize=(5,4))
    for batch_series_idx in range(batch_size):
        one_hot_output_series1 = _predictions_series[:, batch_series_idx,:]
        single_output_series1 = np.array([(out[0]) for out in one_hot_output_series1])
        ttt.extend(single_output_series1)
    ttt = np.array(ttt)
    batchYT = batchYT.reshape((batch_size))
        
    fig.add_subplot(1, 1, 1)
    plt.cla()
    #plt.axis([0, truncated_backprop_length, 0, 1])
    left_offset1 = range(batch_size)
    plt.scatter(left_offset1, (batchYT), s=30, c="red", marker="s", alpha=0.5)
    plt.scatter(left_offset1, (ttt), s=30, c="green", marker="s", alpha=0.3)

    plt.draw()
    plt.pause(0.0001)

def ANN_model():
    # Configure ANN network
    tf.reset_default_graph()   #this resets the graphs

    batchX_placeholder = tf.placeholder(tf.float32, [None, batch_size, inputs])    #create ANN cells with softsign activation function
    batchY_placeholder = tf.placeholder(tf.float32, [None, batch_size, outputs])   #Create the graph with RNN cells
    keep_prob = tf.placeholder(tf.float32)
    time_step = tf.placeholder(tf.float32)
    col_min_holder = tf.placeholder(tf.float32, [None, 1, inputs+outputs])  # keeps columns min for denormalization
    col_max_holder = tf.placeholder(tf.float32, [None, 1, inputs+outputs])  # keeps columns max for denormalization
    
    
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([1, inputs, n_nodes_hl1])), 
                      'biases':tf.Variable(tf.random_normal([1, n_nodes_hl1]))}  

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([1, n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([1, n_nodes_hl2]))} 

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([1, n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([1, n_nodes_hl3]))}
    
    hidden_4_layer = {'weights':tf.Variable(tf.random_normal([1, n_nodes_hl3, n_nodes_hl4])),
                      'biases':tf.Variable(tf.random_normal([1, n_nodes_hl4]))}
    
    hidden_5_layer = {'weights':tf.Variable(tf.random_normal([1, n_nodes_hl4, n_nodes_hl5])),
                      'biases':tf.Variable(tf.random_normal([1, n_nodes_hl5]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([1, n_nodes_hl5, outputs])),
                    'biases':tf.Variable(tf.random_normal([1, outputs])),}

    l1 = tf.add(tf.matmul(batchX_placeholder,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.sigmoid(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.sigmoid(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.sigmoid(l3)
    
    l4 = tf.add(tf.matmul(l3,hidden_4_layer['weights']), hidden_4_layer['biases'])
    l4 = tf.sigmoid(l4)
    
    l5 = tf.add(tf.matmul(l4,hidden_5_layer['weights']), hidden_5_layer['biases'])
    l5 = tf.nn.sigmoid(l5)

    outputAN = tf.matmul(l5,output_layer['weights']) + output_layer['biases']
    outputANN = tf.reshape(outputAN, [-1, batch_size, outputs])
    
    acel = tf.transpose(tf.nn.embedding_lookup(tf.transpose(batchX_placeholder),[2]))  # extract accleration rate
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
    predict_accl = tf.add(tf.multiply(outputANN, tf.subtract(acel_max,acel_min)), acel_min)
    #predict_head = tf.reshape(tf.subtract (leadH1, tf.add(location1, tf.add(tf.multiply(velo1,time_step),tf.multiply(predict_accl,tf.multiply(time_step, time_step))))) , [-1, batch_size, outputs]) # H1 = lead_Y1 - (Subject_Y0 + V0*t + a1*t^2)
    predict_head = tf.subtract (leadH1, tf.add(location1, tf.add(tf.multiply(velo1,time_step),tf.multiply(predict_accl,tf.multiply(time_step, time_step)))))
    
    predict_head_N = tf.divide (tf.subtract( predict_head,target_min), tf.subtract( target_max,target_min))
    
    loss = tf.reduce_mean(tf.divide(tf.abs(tf.subtract(predict_head_N,batchY_placeholder)),(batchY_placeholder)))
    optimizer = tf.train.AdamOptimizer (learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)     

    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())    #initialize all the random variables
        stop_training = 0
        pervious_validation = 1
        x,y, xv, yv, xt, yt, col_min, col_max = generateData()
    
        while stop_training < stop_training_error_time:

            loss_list = []
            sum_loss= 0
            for batch_idx in range(num_batches):
                
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
    
                batchX = x[:, start_idx:end_idx,:]
                batchY = y[:, start_idx:end_idx]
    
                _total_loss, _train_step, _predictions_series = sess.run(
                [loss, training_op, predict_head_N],
                feed_dict={
                                batchX_placeholder: batchX,
                                batchY_placeholder: batchY,
                                time_step: 0.1,
                                col_max_holder: col_max,
                                col_min_holder: col_min,
                                keep_prob: keep_rate
                                })
        
                loss_list.append(_total_loss)
                sum_loss = sum_loss + _total_loss 
                if (batch_idx == num_batches-1):
                    plot(loss_list, _predictions_series, batchY)                    
                    print('This run s MSE =', "%.5f" % _total_loss )
                    ave_loss= sum_loss/num_batches
                    print ('Average training MSE', "%.5f" % (ave_loss*100))
    
            """ Calculate accuracy of the model for validation dataset"""
            Ave_loss_validation= 0
            loss_listV = []
            for batch_idx in range(int(len(xv[0])/batch_size)):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
        
                batchXV = xv[:,start_idx:end_idx,:]
                batchYV = yv[:,start_idx:end_idx]
                __predictions_series = sess.run(
                        [predict_head_N],
                        feed_dict={
                        batchX_placeholder: batchXV,
                        time_step: 0.1,
                        col_max_holder: col_max,
                        col_min_holder: col_min
                        })
                # Calculate MSE for validation data   
                ttt = []
                __predictions_series = np.array(__predictions_series)
                __predictions_series = __predictions_series.reshape(-1, batch_size, outputs)
                for batch_series_idx in range(batch_size):
                    one_hot_output_series = __predictions_series[:,batch_series_idx, :]
                    single_output_series = np.array([(out[0]) for out in one_hot_output_series])
                    ttt.extend(single_output_series)
                ttt = np.array(ttt)
                batchYV = batchYV.reshape((batch_size))
                if (batch_idx == int(len(xv[0])/batch_size)-2):
                    plottest(__predictions_series, batchYV)
                #validation_loss = mean_squared_error(batchYV,ttt)   
                validation_loss = np.mean(np.abs((batchYV - ttt) / batchYV))
                loss_listV.append(validation_loss)
                Ave_loss_validation = Ave_loss_validation + validation_loss
            validation_loss1 = Ave_loss_validation/(int(len(xv[0])/batch_size))
            
            print('Validation average %headway is', "%.5f" % (validation_loss1*100))
                        
            """check if the model overfitted"""
            # Check if the model trained well enough
            if (validation_loss1 < 0.001):
                # First stop condition
                if ((pervious_validation - validation_loss1)< 0.000001):
                    stop_training = 1
                    print ('1. Reason for stop is validation does not improve')
            pervious_validation = validation_loss1
    
        lost_train.append(loss_list)
        lost_validate.append(loss_listV)
        """Start test the model"""
        Ave_loss_test= 0
        loss_listT = []
        test_headway = []
        for batch_idx in range(int(len(xt[0])/batch_size)):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
    
            batchXT = xt[:,start_idx:end_idx,:]
            batchYT = yt[:,start_idx:end_idx]
            _predictions_series = sess.run(
                    [predict_head_N ],
                    feed_dict={
                                    batchX_placeholder: batchXT,
                                    time_step: 0.1,
                                    col_min_holder: col_min,
                                    col_max_holder: col_max
                                    })
            """ Calculate % Error of the model for test dataset"""
            ttt = []
            _predictions_series = np.array(_predictions_series)
            _predictions_series = _predictions_series.reshape(-1, batch_size, outputs)
            for batch_series_idx in range(batch_size):
                one_hot_output_series = _predictions_series[:,batch_series_idx, :]
                single_output_series = np.array([(out[0]) for out in one_hot_output_series])
                ttt.extend(single_output_series)
            ttt = np.array(ttt)
            batchYT1 = batchYT.reshape((batch_size))
            #test_loss = mean_squared_error(batchYT1,ttt) 
            test_loss = np.mean(np.abs((batchYT1 - ttt) / batchYT1))
            loss_listT.append(test_loss)
            Ave_loss_test = Ave_loss_test + test_loss
            test_headway.append(ttt)
            #Plot test data once in 20 batches
            if (batch_idx%50 == 1):
                plottest(_predictions_series,batchYT1)
                print('This batch s %headway  = ', "%.7f" % (test_loss*100)) 
        test_loss1 = Ave_loss_test/(int(len(xt[0])/batch_size))
        print ('********************************')
        print('Test average %headway = ', "%.7f" % (test_loss1*100))
        print ('')
        lost_test.append(loss_listT)
    plt.ioff()
    plt.show()
    return()

lost_train = []
lost_test = []
lost_validate = []
lost_test_line = []    

# Run the ANN n times and create n models
for item in range(num_run):

    ANN_model()
    
    
lost_train = np.array(lost_train)
lost_validate = np.array(lost_validate)
lost_test = np.array(lost_test)
lost_test_line = lost_test.reshape((-1,1))
test_headway = np.array(test_headway)
test_headway = test_headway.reshape((-1,))
test_headway = test_headway * (col_max[0,0,6]-col_min[0,0,6]) + col_min[0,0,6]
test_Location = (xt[0,:,5]* (col_max[0,0,5]-col_min[0,0,5]) + col_min[0,0,5]) - test_headway
test_velocity = (test_Location - (xt[0,:,0]* (col_max[0,0,0]-col_min[0,0,0]) + col_min[0,0,0]))/.1
test_acceleration = (test_velocity - (xt[0,:,1]* (col_max[0,0,1]-col_min[0,0,1]) + col_min[0,0,1]))/.1

print("All runs testing average %headway error", "%.7f" % (np.mean(lost_test)*100))
print("run time", "%.0f" %  (time.clock() - tic)) 
    
"""Denormalize final test results"""
plot_prediction = _predictions_series*(col_max[:,:,input_size]-col_min[:,:,input_size])+ col_min[:,:,input_size]
plot_actual = batchYT1*(col_max[:,:,input_size]-col_min[:,:,input_size])+ col_min[:,:,input_size]
plot_prediction = plot_prediction.reshape(-1,1)
