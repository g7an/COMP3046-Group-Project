# COMP3046-Group-Project

## Description

### Background

This is a course project for HKBU COMP3046 Advanced Programming. The project objective is to make a high proficiency Artificial Neural Network (ANN). The dataset used contains vector data indicating the pixels of images, which is numbers between 0 to 9. The training outcome of the ANN is to predict the number on the image accurately. It also has a optimized version using CUDA on a dedicated branch to speed up the training process.

## Setup and Execute the program

* Download the repo

```
git clone https://github.com/g7an/COMP3046-Group-Project.git
```


* Customize your parameters

	* If boolean `train` in **run.cpp** file is set to true, the program will load training data from data folder; otherwise, the model will load data from 
	   a file and use these pre-trained weight to do predition directly 
	   
	* Adjust the number of hidden layers and neurons in each hidden layer by input number in array `layerSize` ,which is in **run.cpp**



* Run the program

	```
	$ g++ -std=c++11 MyANN.cpp MyMatrix.cpp run.cpp globalFunctions.cpp -o run
	$ ./run
	```

* Output of the program

	The trained weight and bias data will be stored in two seperate file **matrix_data_v2.txt** and **bias.txt**. If you want the ANN model to use the trained 	   weight and bias, please change their names to **input_matrix_data.txt** and **input_bias.txt** respectively. 

## Side Notes

### Design of The ANN class :

1. class member

	* float r : learning rate;
	* int epochs: number of epoches;
	* int decayEpoch : number of epoches between two learing rate reduction.
	* float decay : the rate of learning rate reduction.
	* int num_hidLayer : the number of hidden layers
	* int* total_neurons : number of neurons in each hidden layer
	* int batch_size : the size of mini-batch size;
	* std::vector<MyMatrix<float> *> hidWeight : a vector of matrixes to store the weights of the ANN model
	* std::vector<MyMatrix<float> *> delta_w : a vector of matrixes to store the change of  weights in each mini-batch
	* MyMatrix<float> *input : a matrix pointer to store the input data
	* MyMatrix<float> *target: a matrix pointer to store the target data
	* std::vector<MyMatrix<float> *> outH: to store the output value in every hidden layer and output layer.
	* std::vector<float *> bias : store the value of bias for each hidden layer and output layer.
	* std::vector<float *> bias : store the value of change of bias 
	* MyMatrix<float> *partError : store part of the backpropagation result.
	* MyMatrix<float> *netH: store part of the feed forward result.

1. class functions
 
	* MyANN(float, int, int,int*,int,int,float);
		input:learning rate,epochs, mini-batch size, an array contains the number of neurons in each layer,
			the length of the array,decayEpoch,decay rate 
		function: build an ANN model according to user input

	* sigmoid
		input:float 
		output:float 
		function: calculate the sigmoid value of the input

	* setEpochs
		input: int 
		function:set the learning rate to the input

	* train
		input: a vector of float vector (training data), a float vector (ground-truth data)
		fucntion: use the input and target value to train the ANN model 

	* getAcc
		input: a vector of float vector (test data) 
		output: void
		fucntion: use the input to get accuaracy of the result

	* forward
		input: a float vector 
		output: a pointer to a MyMatrix object
		fucntion: use the input to do feed forward and return the value in  ouput layer in that MyMatrix object

	* storeWeight
		function: save the weights to "/matrix_data.txt" 

	* loadWeight
		function: load the weights from  "/input_matrix_data.txt" 
	
	* predict
		input: a float vector 
		output: int
		fucntion: use the input to predict the result

	* getLoss
		input: vector of vector, vector
		output: void
		function: a function serves as a media between whole dataset to a single data. 
				Calls in totalLoss function to get loss from a single data.
	

## Experimental results

In our experiments, the accuracy was 12% and the validation loss was 0.51 after first epoch. After 25 epochs, the accuracy exceeds 50% while loss decreased to 0.33. Within 60 epochs, the accuracy exceeds 70% and loss decreased to 0.22. Within 90 epochs, the accuracy reached 80% with loss value 0.173376. The change of accuracy and loss become more slowly in the following 100 epochs. At epoch 200, the accuracy reached 87.6% and loss became 0.11. The change of accuracy and loss become more trivial in the following 100 epochs, so we conclude that accuracy value and loss value are both converged. At epoch 300, the accuracy reached 89% and loss bacame 0.096. The detailed experiment results are stored in **nohup.out**

## Authors

[Gloria Tan](https://github.com/g7an) and [Tingyao Li](https://github.com/BearerOfTheCurse) contributed equally to the project.
	



