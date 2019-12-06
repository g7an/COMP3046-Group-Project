# COMP3046-Group-Project
REMIND:
	1. You can use "g++ -std=c++11 MyANN.cpp MyMatrix.cpp run.cpp globalFunctions.cpp -o run" and "./run" to run the program
	2. In the run.cpp, you can adjust the parameters of the ANN model and set the model to "train" or not by adjust the value of "bool train",
	   If train==true, the model will load the training data , train the model and do the prediction. If train==false, the model can load data from 
	   a file and use these pre-trained weight to do predition directly
	3. You can adjust the number of hidden layers and the number of neurons in each hidden layer by input number in "layerSize" array,which is at the beginning of the "run.cpp". You can also adjust the learning rate by changing the first parameter of the MyANN constructor, or adjust the number of epoches by changing the second parameter, or adjust the size of mini-batch by changing the third parameter. Please do not change the rest of parameters.
	4. The trained weight and bias data will be stored in two seperate file "matrix_data_v2.txt" and "bias.txt". If you want the ANN model to use the trained weight and bias, please change their names to "input_matrix_data.txt" and "input_bias.txt" respectively. 

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
		function: calculate the sigmoid value of the input

	3.setEpochs
		input: int 
		function:set the learning rate to the input

	4.train
		input: a vector of float vector (training data), a float vector (ground-truth data)
		fucntion: use the input and target value to train the ANN model 

	5.getAcc
		input: a vector of float vector (test data) 
		output: void
		fucntion: use the input to get accuaracy of the result

	6.forward
		input: a float vector 
		output: a pointer to a MyMatrix object
		fucntion: use the input to do feed forward and return the value in  ouput layer in that MyMatrix object

	7.storeWeight
		function: save the weights to "/matrix_data.txt" 

	8.loadWeight
		function: load the weights from  "/input_matrix_data.txt" 
	
	9.predict
		input: a float vector 
		output: int
		fucntion: use the input to predict the result

	10.getLoss
		input: vector of vector, vector
		output: void
		function: a function serves as a media between whole dataset to a single data. 
				Calls in totalLoss function to get loss from a single data.
	

Experimental results

	In our experiments, the accuracy was 12% and the validation loss was 0.51 after first epoch. 
	After 25 epochs, the accuracy exceeds 50% while loss decreased to 0.33. 
	Within 60 epochs, the accuracy exceeds 70% and loss decreased to 0.22. 
	Within 90 epochs, the accuracy reached 80% with loss value 0.173376. 
	The change of accuracy and loss become more slowly in the following 100 epochs. 
	At epoch 200, the accuracy reached 87.6% and loss became 0.11.
	The change of accuracy and loss become more trivial in the following 100 epochs,
	so we conclude that accuracy value and loss value are both converged.
	At epoch 300, the accuracy reached 89% and loss bacame 0.096.
	The detailed experiment results are stored in "nohup.out"

Contribution of Each member (Tan Shuyao(Tan), Li Tingyao(Li))

	Tan does the most part of the design of the ANN model. Li does the most part of documentation, and more parts of debugging. Both of us do the implementation and experiments part.
	



