# Machine Learning and Deep Learning for Car-Following
 
## Background

This is a project which models human driving behavior based on parametric, machine learning, and deep learning algorithms. The goal of this project is to compare the performance of these algorithms in single-step predictions and multi-step predictions. 

This project investigates human car-following behavior. Car-Following models explain drivers' longitudinal behaviors. In other words, they predict when a driver accelerates or decelerates.  

## Modeling

This study conducts a cross-category comparison between one parametric model (intelligent driver model (IDM)), two new machine-learning CF models based on feedforward neural network (FNN) and recurrent neural network (RNN), and one novel deep-learning CF model (Deep-RNN) with long short-term memory (LSTM). 
The models are developed in TensorFlow and compared at local (single-step) and global (multi-step) levels.

The below figure shows the performance of the CF models in multi-step predictions. The result shows that the Deep Learning model has the best performance in multi-step predictions (i.e, 100 steps predictions). The Deep Learning CF model shows a higher variation in performance than the IDM model, which has constant performance in predicting CF behavior.

![alt text](https://github.com/sadvaseb/Machine-Learning-and-Deep-Learning-for-Car-Following/blob/master/Fig11-All-togather.jpg "Pic 1")



The Federal Highway Administration (FHWA) gathered the Next Generation Simulation dataset (NGSIM) for Interstate 80 freeway (I-80) in Emeryville, California. NGSIM trajectory data has been collected by FHWA in 2005 by a set of synchronized cameras on the freeway. The original dataset is available from [here](https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm).

the original dataset contains several noises. So, the database was cleaned and soften by the Multitude team. They created the Reconstructed NGSIM dataset, which is available from [here](http://www.multitude-project.eu/reconstructed-ngsim.html) However, these datasets only contains the subject vehicle's attributes (e.g., ID, location, and speed) and the following's and preceding's IDs. 

## This Dataset
We extended the Reconstructed NGSIM dataset to include surrounding vehicles' (i.e., following, subject, preceding, second preceding, and putative leaders on the left and right lanes) attributes. The surrounding vehicles are shown in the below picture. The attributes are the vehicles' IDs, locations, speeds, accelerations, velocity difference (i.e. velocity difference between the subject vehicle and its surrounding vehicles), headway (i.e. the bumper to bumper distance between the subject vehicle and its surrounding vehicle). Since no dataset with this information exists, NGSIM is restructured by an R code to generate the information. 

Best of luck!

## Cite
Please cite this dataset/code as:

Vasebi S., Hayeri Y.M., Jin J. (2020) Human Car-Following Behavior: Parametric, Machine-Learning, and Deep-Learning Perspectives. In: Stanton N. (eds) Advances in Human Aspects of Transportation. AHFE 2020. Advances in Intelligent Systems and Computing, vol 1212. Springer, Cham. https://doi.org/10.1007/978-3-030-50943-9_6
 

