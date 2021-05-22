# -*- coding: utf-8 -*-
"""
Created on Mon May 20 00:14:22 2019 

@author: Sertab Gamma
This code is develped to calibrate IDM car following model with GA algorithm. It uses NGSIM dataset and deap library.
"""
"""
From  Calibrating Car-Following Models using Trajectory Data
V0 = 16.4
T = 1.39
s0 = 1.04
a = 1.52
b = .614
"""
import random
import numpy as np
from deap import creator, base, tools 
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import time
from sklearn.model_selection import train_test_split

#seeds for test data selection
seed1 = [2861, 2723, 4436, 9064, 2428, 4748, 5665, 8987, 8088, 3465, 5998,
       4636, 7457, 5431, 5494, 9009,   34, 5037, 7155, 1323, 2156, 6835,
       2796, 8652, 9877, 3561, 3292, 4503, 9124, 4397]
#seeds for validation data selection
seed2 = [9284, 7802, 3369,  692, 2221, 6496, 2389, 6004, 3076, 3214, 9699,
       9189, 7811, 9127, 6113, 2676, 9032, 7712, 3312, 5200, 7831, 9576,
         75, 5590, 4562, 8011,  387, 3999, 9402, 4887]

tic = time.clock()
num_run = 5    # number of run   
batch_size = 118  #In this code refers to length of back propagation
outputs = 1            #number of outputs
num_classes = outputs   # number of outputs
inputs = 6 
test_sample_size = .2 # percenetage testing dat
IterationToStop = 20 # number of not-improving iterations to stop GA

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
    extractedDa = input1[:,[0, 1, 2, 3, 4, 6, 5]]
    # String to float all data and remove columns' titles from the data
    input11 = []
    for i in range(1, len(extractedDa)):   
        input11.append([])
        for j in range(0, inputs+num_classes):
            input11[i-1].append(float(extractedDa[i][j]))
    input2 = np.array(input11)
    
    # Create Batches
    x_data = input2[:(len(input2)-(len(input2) % (batch_size)))]
    x1 = x_data.reshape((-1, batch_size, inputs+num_classes))    
    #TrainData = TrainData.transpose(1,0,2) 
    
    TrainData, TestData =train_test_split(x1, test_size=test_sample_size, random_state = seed1[item])
     
    # Find min and max values at each column for training dataset
    TrainData1 = TrainData.reshape((1, -1, inputs+num_classes))  
    col_min = TrainData1.min(axis=(1), keepdims=True)
    col_max = TrainData1.max(axis=(1), keepdims=True)
    
    col_min = col_min - (col_max - col_min)*0.1
    col_max = col_max + (col_max - col_min)*0.1
    TrainDataN = TrainData1.reshape((-1, inputs+num_classes))

    TestDataN = TestData.reshape((-1, inputs+num_classes))

    
    return (TrainDataN, TestDataN, col_min, col_max)
    
"""Calculate goodness of fit for a gene """    
def evalIDM(individual):
        if individual[1] < 1 or individual[1] > 500 or individual[2] < 1 or individual[2] > 800 or individual[3] < 1 or individual[3] > 600 or individual[4] < 1 or individual[4] > 600 or individual[0] <= 0 or individual[0] > 1537: 
           #    must be                  0.1 <T< 5  s                         0.1 <s0< 8  m                                            0.1 <a< 6  m/s2                     0.1 <b< 6  m/s2       0 <V0 < 70 m/s         
            err = 1
        else: 
            individual1 = [round (x) / 100 for x in individual] 
            s = (individual1[2]+extractedData[:,1]*individual1[1]+ (extractedData[:,1]*extractedData[:,4])/(2*(np.sqrt(individual1[3]*individual1[4]))+.00000000001))
            acel2 = individual1[3]*(1- ((extractedData[:,1]/(individual1[0]+.00000000001))**4) - ((s /(extractedData[:,3]+.0000000001))**2))   #IDM equation
            err = np.mean(abs((extractedData[:,6] - (extractedData[:,5] - (extractedData[:,0] + 0.1 * extractedData[:,1] + .01 * acel2[:]))))/extractedData[:,6])  # goodness of fit function based on % headway accuricy  = abs(actual headway- predicted headway)/ actual headway
            #err = np.sqrt(((np.sum((extractedData[:,6] - (extractedData[:,5] - (extractedData[:,0] + 0.1 * extractedData[:,1] + .005 * acel2[:])))**2))/(len(extractedData)))/ (((np.sum(extractedData[:,6]))/(len(extractedData)))**2))   # goodness of fit function based on % headway accuricy  = abs(actual headway- predicted headway)/ actual headway
        
        return err,

"""Calculate goodness of fit for a generation """   
def set_fitness(population): 
    fitnesses = [(individual, toolbox.evaluate(individual)) for individual in population]

    for individual, fitness in fitnesses:
        individual.fitness.values = fitness
    return(fitnesses)
""" Plot progress of GA"""        
def pull_stats(population, iteration=1):  
    fitnesses = [individual.fitness.values[0] for individual in population ] 
    return {
        'i': iteration,
        'mu': np.mean(fitnesses),
        'std': np.std(fitnesses),
        'max': np.max(fitnesses),
        'min': np.min(fitnesses)
    }




lost_train = []
train_best_calibrations = []
lost_test = []
lost_test_line = []    

# Run the GA n times and create n models
for item in range(num_run):
    stop_training = 0
    pervious_validation = 1
    extractedData, testData, col_min, col_max = generateData()
    
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) 
    creator.create("individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    """All attributes will have 2 decimals so the integars are multiplied at 100 """
    toolbox.register("attr0", random.randint, 0,7000) #V0 
    toolbox.register("attr1", random.randint, 1,500)  #T
    toolbox.register("attr2", random.randint, 1,800)  #s0
    toolbox.register("attr3", random.randint, 1,600)  #a
    toolbox.register("attr4", random.randint, 1,600)  #b
    
    toolbox.register("individual", tools.initCycle, creator.individual, (toolbox.attr0, toolbox.attr1, toolbox.attr2, toolbox.attr3, toolbox.attr4))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evalIDM)
    toolbox.register("mate", tools.cxBlend, alpha=0.3)  # it wwas 0.5
    #toolbox.register("mutate", tools.mutUniformInt, low= (0,1,1,1,1), up=(7000,500,800,600,600), indpb=0.4)
    #toolbox.register("mutate", tools.mutGaussian, mu= (0,0,0,0,0) , sigma= (10, .5, .5, .5, .5), indpb=0.4)
    toolbox.register("mutate", tools.mutPolynomialBounded, eta= 15, low= (0,1,1,1,1), up=(7000,500,800,600,600), indpb=0.4)                
    toolbox.register("select", tools.selTournament, tournsize=3)  
    
    population = toolbox.population(n=1000) 
    ## set fitness,
    fitnesses2 = set_fitness(population)

    ## globals,
    stats = []
    iteration = 1
    improvment = 0
    MinError = 1
    loss_listT = []
    while stop_training < 1:
        print('******************************** iteration =', iteration)
        current_population = list(map(toolbox.clone, population))
        
        offspring = []
        for _ in range(500): 
            i1, i2 = np.random.choice(range(len(population)), size=2, replace=False)
    
            offspring1, offspring2 = toolbox.mate(population[i1], population[i2])
    
            offspring.append(toolbox.mutate(offspring1)[0])
            offspring.append(toolbox.mutate(offspring2)[0])  
        
        for child in offspring:
            current_population.append(child)
    
        ## reset fitness,
        fitnesses2 = set_fitness(current_population)
    
        population[:] = toolbox.select(current_population, len(population))
        
        ## set fitness on individuals in the population,
        stats.append(pull_stats(population, iteration))
        fitnesses = [individual.fitness.values[0] for individual in population]
        ind = fitnesses.index (stats[iteration-1]['min'])
        tt = [individual for individual in population ]
        print ('The lowest error is ', stats[iteration-1]['min'], 'and it is achieved at ', [int(round (x))/ 100 for x in tt[ind]] )
        
        
        """Stop training condition"""
        if stats[iteration-1]['min'] == MinError:   # if error does not reducing
            improvment += 1
            if improvment == IterationToStop:   # If error has not been decreased for 20 generations
                stop_training = 1   # stop training and strat testing
        else:
            improvment = 0     # set number of not-improving generations to zero
            MinError = stats[iteration-1]['min']  # set minimum error to current error
        iteration += 1
    
    sns.set()
    _ = plt.scatter(range(1, len(stats)+1), [ s['mu'] for s in stats ], marker='.')
    _ = plt.title('average fitness per iteration')
    _ = plt.xlabel('iterations')
    _ = plt.ylabel('fitness')
    
    plt.show()
    plt.pause(0.0001)
    
    _ = plt.scatter(range(1, len(stats)+1), [ s['min'] for s in stats ], marker='*')
    _ = plt.axis([0, len(stats)+1, 0.8*stats[len(stats)-1]['min'], 1.2*stats[0]['min']])
    _ = plt.title('Min fitness per iteration')
    _ = plt.xlabel('iterations')
    _ = plt.ylabel('fitness')
    plt.pause(0.0001)
    
    fitnesses = [individual.fitness.values[0] for individual in population]
    ind = fitnesses.index (stats[len(stats)-1]['min'])
    tt = [individual for individual in population ]
    print ('The lowest error is ', stats[len(stats)-1]['min'])
    print ('It is achieved at ', [int(round (x))  / 100 for x in tt[ind]] )
    
    individual1 = [int(round (x))  / 100 for x in tt[ind]] 
    s1 = (individual1[2]+extractedData[:,1]*individual1[1]+ (extractedData[:,1]*extractedData[:,4])/(2*(np.sqrt(individual1[3]*individual1[4]))+.00000000001))
    acel3 = individual1[3]*(1- ((extractedData[:,1]/(individual1[0]+.00000000001))**4) - ((s1 /(extractedData[:,3]+.0000000001))**2))   #IDM equation
    err_per = np.mean(abs((extractedData[:,6] - (extractedData[:,5] - (extractedData[:,0] + 0.1 * extractedData[:,1] + .01 * acel3[:]))))/(extractedData[:,6]))   # goodness of fit function based on % headway accuricy  = abs(actual headway- predicted headway)/ actual headway
    err_per = err_per * 100  # percent error
    print ('********************************')
    print ('Training percent error for the best calibration values are %', "%.7f" % err_per)
    lost_train.append(err_per/100)
    train_best_calibrations.append(individual1) 
    
    """Calculate accuricy for Test data set"""
    test_acceleration = []
    test_velocity = []
    test_headway = []
    gen = np.array([int(x)/ 100 for x in tt[ind]]) 
    sT = (individual1[2]+testData[:,1]*individual1[1]+ (testData[:,1]*testData[:,4])/(2*(np.sqrt(individual1[3]*individual1[4]))+.00000000001))
    acelT = individual1[3]*(1- ((testData[:,1]/(individual1[0]+.00000000001))**4) - ((sT /(testData[:,3]+.0000000001))**2))   #IDM equation
    err_test = np.mean(abs((testData[:,6] - (testData[:,5] - (testData[:,0] + 0.1 * testData[:,1] + .01 * acelT[:])))))/(np.sum( testData[:,6]))   # goodness of fit function based on % headway accuricy  = abs(actual headway- predicted headway)/ actual headway
    err_test = err_test * 100  # percent error
    print ('Test percent error for the best calibration values are %', "%.7f" % err_test)
    for batch_idx in range(int(len(testData)/(batch_size))):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        batchYT = testData[start_idx:end_idx,inputs]
        batchXT = testData[start_idx:end_idx,0:inputs]
        s2 = (individual1[2]+batchXT[:,1]*individual1[1]+ (batchXT[:,1]*batchXT[:,4])/(2*(np.sqrt(individual1[3]*individual1[4]))+.00000000001))
        acel4 = individual1[3]*(1- ((batchXT[:,1]/(individual1[0]+.00000000001))**4) - ((s2 /(batchXT[:,3]+.0000000001))**2))   #IDM equation
        _predictions_series = (batchXT[:,5] - (batchXT[:,0] + 0.1 * batchXT[:,1] + .01 * acel4[:]))  # goodness of fit function based on % headway accuricy  = abs(actual headway- predicted headway)/ actual headway
        per_error_test = np.mean(abs (_predictions_series - batchYT)/(batchYT))
        loss_listT.append(per_error_test)
        
        test_acceleration.append(acel4)
        velo4 = acel4[:]*.1 + batchXT[:,1]   # calculate velocity for test data V1 = a1*t + V0
        test_velocity.append(velo4)
        test_headway.append(_predictions_series)
        if (batch_idx%50 == 1):
            fig = plt.figure(figsize=(7,6))                
            fig.add_subplot(1, 1, 1)
            plt.cla()
            left_offset1 = range(batch_size)
            plt.scatter(left_offset1, (batchYT), s=30, c="red", marker="s", alpha=0.5)
            plt.scatter(left_offset1, (_predictions_series), s=30, c="green", marker="s", alpha=0.3)
            plt.title('Actual headway vs prediction')
            plt.xlabel('step')
            plt.ylabel('headway(m)')
            plt.draw()
            plt.pause(0.001)
            print('This batch s %headway accurcy = %', "%.7f" % ((per_error_test)*100))  
    lost_test.append(loss_listT)       

lost_train = np.array(lost_train)
lost_test = np.array(lost_test)
lost_test_line = lost_test.reshape((1,-1))
test_acceleration = np.array(test_acceleration)
test_acceleration = test_acceleration.reshape((-1,1))
test_velocity = np.array(test_velocity)
test_velocity = test_velocity.reshape((-1,1))
test_headway = np.array(test_headway)
test_headway = test_headway.reshape((-1,1))
print('=================================') 
print('=================================')
print("All runs testing average %headway error", "%.7f" % (np.mean(lost_test)*100))
print("run time", "%.0f" %  (time.clock() - tic)) 
print ('********************************')

plot_prediction = _predictions_series
plot_actual = batchYT

