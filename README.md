# COMP3046-Group-Project
REMIND:
	1. You can use "g++ -std=c++11 MyANN.cpp MyMatrix.cpp test2.cpp globalFunctions.cpp -o test2" and "./test2" to run the program
	2. In the test2.cpp, you can adjust the parameters of the ANN model and set the model to "train" or not by adjust the value of "bool train",
	   If train==true, the model will load the training data , train the model and do the prediction. If train==false, the model can load data from 
	   a file and use these pre-trained weight to do predition directly

Design of The ANN class :

(1)class member

	1.float r : learning rate;
	2.int epochs: number of epoches;
	3.int decayEpoch : number of epoches between two learing rate reduction.
	4.float decay : the rate of learning rate reduction.
	5.int num_hidLayer : the number of hidden layers
	6.int* total_neurons : number of neurons in each hidden layer
	7.int batch_size : the size of mini-batch size;
	8.std::vector<MyMatrix<float> *> hidWeight : a vector of matrixes to store the weights of the ANN model
	9.std::vector<MyMatrix<float> *> delta_w : a vector of matrixes to store the change of  weights in each mini-batch
	10.MyMatrix<float> *input : a matrix pointer to store the input data
	11.MyMatrix<float> *target: a matrix pointer to store the target data
	12.std::vector<MyMatrix<float> *> outH: to store the output value in every hidden layer and output layer.
	13.std::vector<float *> bias : store the value of bias for each hidden layer and output layer.
	14.std::vector<float *> bias : store the value of change of bias 
	15.MyMatrix<float> *partError : store part of the backpropagation result.
	16.MyMatrix<float> *netH: store part of the feed forward result.

(2)class functions
 
	1.MyANN(float, int, int,int*,int,int,float);
		input:learning rate,epochs, mini-batch size, an array contains the number of neurons in each layer,
			the length of the array,decayEpoch,decay rate 
		function: build an ANN model according to user input

	2.sigmoid
		input:float 
		output:float 
		function:calculate the sigmoid value of the input
	3.setEpochs
		input: int 
		function:set the learning rate to the input

	4.train
		input: a vector of float vector (raw data), a float vector (target data)
		fucntion: use the input and target value to train the ANN model 

	5.predict
		input: a float vector 
		output: int
		fucntion: use the input to predict the result

	6.forward
		input: a float vector 
		output: a pointer to a MyMatrix object
		fucntion: use the input to do feed forward and return the value in  ouput layer in that MyMatrix object

	7.storeWeight
		function: save the weights to "/matrix_data.txt" 

	8.loadWeight
		function: load the weights from  "/input_matrix_data.txt" 

Experimental results
	
	In the first 180 epoches, the loss decrease from 3 to 0.8. In the second 100 epoches, the loss decrease from 0.8 to 0.27. In the last 200 epoches
	,the loss decrease from 0.27 to 0.197. The traing process of the last 200 epoches can be found in "nohup.out". The time for each epoch is quite  
	stable and each epoch will cost around 40s.

Contribution of Each member (Tan Shuyao(Tan), Li Tingyao(Li))

	Tan does the most part of the design of the ANN model. Li does the most part of experiments and documentation. Both of us do the implementation and 
	debugging part.
	



