#include "ANN.h"

#include "MyVector.h"
#include "MyVector.cpp"
#include "MyMatrix.h"
#include "MyMatrix.cpp"
#include "globalFunctions.cpp"

#include <typeinfo>

#include <iostream>
#include <fstream>
#include <iostream>
#include <string>
#include <queue>
#include <vector>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
using namespace std;

template <class T, class N>
MyVector<T> *vecAdd(MyVector<T> &x, MyVector<T> &y, N a);

template <class T>
T vecDot(MyVector<T> &x, MyVector<T> &y);

template <class T>
MyMatrix<T> *matAdd(MyMatrix<T> &x, MyMatrix<T> &y);

template <class T>
MyMatrix<T> *matSub(MyMatrix<T> &x, MyMatrix<T> &y);

template <class T>
MyMatrix<T> *matVecMul(MyMatrix<T> &x, MyVector<T> &y);

template <class T>
MyMatrix<T> *vecMatMul(MyVector<T> &x, MyMatrix<T> &y);

template <class T>
MyMatrix<T> *matMatMul(MyMatrix<T> &x, MyMatrix<T> &y);

/*
int main(void){
srand((unsigned)time(NULL));
float lr = 0.5;							//learning rate



return 0;
}
*/

ANN::ANN(float lr, int epochs) : lr(lr), epochs(epochs)
{
	// cout << "Input number of hidden layer: " << endl;
	// cin >> num_hidLayer;
	// cout << "fuck" << endl;
	num_hidLayer = 2;
	const int layer = num_hidLayer;
	int ele = 0;
	int prevLayer = 784;
	num_neurons = new int[layer];
	for (int i = 0; i < num_hidLayer; i++)
	{
		num_neurons[i] = 10;
		// 	ele += num_neurons[i] * (prevLayer + 1);
		// 	prevLayer = num_neurons[i];
	}
	const int num_weight = num_hidLayer;
	// (hidden layer + 1) 层 weight matrix，每个里面neuron个元素
	const int num_matrix = layer + 1;
	// cout << num_matrix << endl;
	hidWeight = new MyVector<MyMatrix<float> *>(num_matrix); // row 0: input_to_hidden_weight; row_last: hide_to_output_weight
	//hidWeight = new MyVector<void *>(num_matrix); // row 0: input_to_hidden_weight; row_last: hide_to_output_weight
	// initialization of weights
	netH = new MyVector<MyMatrix<float> *>(num_matrix);
	outH = new MyVector<MyMatrix<float> *>(num_matrix);
	outHBias = new MyVector<MyMatrix<float> *>(num_matrix);  
	for (int i = 0; i < num_hidLayer + 1; i++)
	{
		hidWeight->arr[i] = new MyMatrix<float>(num_neurons[i], prevLayer + 1);
		for (int j = 0; j < hidWeight->arr[i]->dim()[0]; j++)
		{
			for (int k = 0; k < hidWeight->arr[i]->dim()[1]; k++)
			{
				hidWeight->arr[i]->n2Arr[i][j] = random();
			}
		}
		prevLayer = num_neurons[i];
	}
	// hidWeight->arr[0]->print();
	inputToHideWeight = new MyMatrix<float>(10, 785);
	input = new MyMatrix<float>(785, 1);
	// netH = new MyMatrix<float>(10, 1);				  //weight*input
	// outH = new MyMatrix<float>(10, 1);				  //after sigmoid
	hideToOutputWeight = new MyMatrix<float>(10, 11); //hidden->out
	output = new MyMatrix<float>(10, 1);
	target = new MyMatrix<float>(10, 1); // ground-truth
	partError = new MyMatrix<float>(10, 1);
}

ANN::~ANN()
{
	delete inputToHideWeight;
	// delete hidWeight;
	delete input;
	delete netH;
	delete outH;
	delete hideToOutputWeight;
	delete output;
	delete target;
	delete partError;
}

inline float ANN::sigmoid(float input)
{ //calculate sigmoid function
	return 1 / (1 + exp(-input));
}

inline float ANN::random()
{ //return a float random number between 0 and 1
	return rand() / float(RAND_MAX);
}

void ANN::setEpochs(int e)
{
	epochs = e;
}

void ANN::setLR(float f)
{
	lr = f;
}

void ANN::initializeWeight()
{
	srand((unsigned)time(NULL));

	//	cout << "first weight" << endl;
	// set value to hidden weights
	for (int i = 0; i < inputToHideWeight->dim()[0]; i++)
	{
		for (int j = 0; j < inputToHideWeight->dim()[1]; j++)
		{
			inputToHideWeight->n2Arr[i][j] = random(); //shuffle data between [0, 1]
													   //			cout << inputToHideWeight->n2Arr[i][j] << " ";
		}
		//		cout << endl;
	}

	//	cout << "second weight" << endl;
	for (int i = 0; i < hideToOutputWeight->dim()[0]; i++)
	{
		for (int j = 0; j < hideToOutputWeight->dim()[1]; j++)
		{
			hideToOutputWeight->n2Arr[i][j] = random();
			//			cout << hideToOutputWeight->n2Arr[i][j] << " ";
		}
	}
}

void ANN::train(vector<vector<float> > in, vector<float> t)
{
	const int vector_size = in[0].size();
	float test = inputToHideWeight->n2Arr[0][0];
	if (test < -100000 || test > 100000)
	{
		initializeWeight();
	}

	const int batch_size = 256;
	const int steps = in.size() / batch_size + 1; // how many batches to finish an epoch

	// MyMatrix<float> *outHBias = new MyMatrix<float>(11, 1);
	MyMatrix<float> *tmp = NULL;
	MyMatrix<float> *tmpWeight = new MyMatrix<float>(10, 10);
	MyMatrix<float> *tmpTrans = NULL;
	MyMatrix<float> *sum_in = NULL;
	MyMatrix<float> *sum_out = NULL;

	vector<float> batch_data(vector_size);
	for (int round = 0; round < steps; round++)
	{
		for (int turn = 0; turn < batch_size; turn++)
		{ //training begin
			// load in one data for training
			batch_data = in[turn + round * batch_size];

#pragma omp parallel for num_threads(4)

 			for (int i = 0; i < 10; i++)
			{
				i == t[turn + round * batch_size] ? target->n2Arr[i][0] = 1 : target->n2Arr[i][0] = 0; //make single target number into a target matrix
			}

#pragma omp parallel for num_threads(4)
			/* feed forward function:
				input: vector to store input data for this turn
				netH: vector of matrix to store the net output (wx + b) in each node
				netWeight: matrix to store the weight of one layer
				net: matrix to store net value of one layer
				outH: vector of matrix to store sigmoid output of a node
				outHBias: vector of matrix to store sigmoid output plus bias
				recursively do matrix multiplication and sigmoid calculation until the last layer
			*/
			for (int i = 0; i < vector_size; i++)
			{
				input->n2Arr[i][0] = batch_data[i];
			}
			int foo = input->dim()[0] - 1;
			input->n2Arr[foo][0] = 1; //set bias			
			netH->arr[0] = input; 
			for(int i = 0; i < num_hidLayer + 1; i++){
				MyMatrix<float> *netWeight = hidWeight->arr[i];
				MyMatric<float> *net = netH->arr[i];
				netH->arr[i] = matMatMul(*netWeight, *net);
				for (int j = 0; j < netH->arr[i]->dim()[0]; i++){ //change netH to outH, outBias is out matrix plus bias
					outH->arr[i]->n2Arr[j][0] = sigmoid(netH->arr[i]->n2Arr[j][0]);
					outHBias->->arr[i]->n2Arr[j][0] = sigmoid(netH->arr[i]->n2Arr[j][0]);
				}
				if (i == num_hidLayer)
					break;
				foo = outHBias->dim()[0] - 1;
				outHBias->arr[i]->n2Arr[foo][0] = 1; //complete the outBias
				netH->arr[i + 1] = outHBias->arr[i];
			}
			tmp = matSub(*output, *target);
// #pragma omp parallel for num_threads(4)
			// for (int i = 0; i < netH->arr[0]->dim()[0]; i++)
			// { //change netH to outH, outBias is out matrix plus bias
			// 	outH->n2Arr[i][0] = sigmoid(netH->arr[0]->n2Arr[i][0]);
			// 	outHBias->n2Arr[i][0] = sigmoid(netH->arr[0]->n2Arr[i][0]);
			// }

			// foo = outHBias->dim()[0] - 1;

			// outHBias->n2Arr[foo][0] = 1; //complete the outBias

			// output = matMatMul(*hideToOutputWeight, *outHBias);

// #pragma omp parallel for num_threads(4)
			// for (int i = 0; i < output->dim()[0]; i++)
			// {
			// 	output->n2Arr[i][0] = sigmoid(output->n2Arr[i][0]); //finish forward propagation
			// }

			//			cout << "output: " << endl;
			//			output->print();
			//			cout << endl;

#pragma omp parallel for num_threads(4)
			for (int i = 0; i < hideToOutputWeight->dim()[0]; i++)
			{
				for (int j = 0; j < hideToOutputWeight->dim()[1] - 1; j++)
				{
					tmpWeight->n2Arr[i][j] = hideToOutputWeight->n2Arr[i][j]; // last hidden weight
				}
			}

			//					cout << "tmpWeight: "<< endl;
			//					tmpWeight->print();
			//					cout << endl;

			 // Etotal / output = out - target

			//					cout << "tmp: "<< endl;
			//					tmp->print();
			//					cout << endl;

#pragma omp parallel for num_threads(4)
			for (int i = 0; i < output->dim()[0]; i++)
			{
				float outTmp = output->n2Arr[i][0];
				partError->n2Arr[i][0] = lr * outTmp * (1 - outTmp) * (tmp->n2Arr[i][0]); // out / net = out(1-out)

			} //complete the partError with lr

			//					cout << "partError: "<< endl;
			//					partError->print();
			//					cout << endl;

			// tmpTrans = partError->transpose();                            // backpropagation of weight between input layer and hidden layer
			tmpTrans = tmpWeight->transpose();
			// tmpTrans->print();
			// partError->print();
			tmp = matMatMul(*tmpTrans, *partError);

			//			cout << "tmp: " << endl;
			//			tmp->print();
			//			cout << endl;

#pragma omp parallel for num_threads(4)
			for (int i = 0; i < outH->dim()[0]; i++)
			{
				float outTmp = outH->n2Arr[i][0];
				tmp->n2Arr[i][0] = tmp->n2Arr[i][0] * outTmp * (1 - outTmp);
			}
			// tmp = tmp->transpose();

			tmpTrans = input->transpose();
			// cout << "row:" << tmpTrans->dim()[0] << "col: " << tmpTrans->dim()[1] << endl;
			tmp = matMatMul(*tmp, *tmpTrans);

			const int r = tmp->dim()[0];
			const int c = tmp->dim()[1];
			sum_in = new MyMatrix<float>(r, c);
			sum_in = matAdd(*sum_in, *tmp);
			// sum_in->print();
			// inputToHideWeight = matAdd(*inputToHideWeight, *tmp);

			tmpTrans = outHBias->transpose(); // backpropagation of weight between hidden layer and output layer
			tmp = matMatMul(*partError, *tmpTrans);
			const int row = tmp->dim()[0];
			const int col = tmp->dim()[1];
			sum_out = new MyMatrix<float>(row, col);
			sum_out = matAdd(*sum_out, *tmp);
			// sum_out = matAdd(*sum_out, *tmp);

			// cout << "sum_out: " << sum_out << endl;
			// cout << "size of tmp :" << tmp->dim()[0] << "*" << tmp->dim()[1] << endl;

			// hideToOutputWeight = matAdd(*hideToOutputWeight, *tmp);

			//		cout << "new hidden weight" <<endl;
			//		hideToOutputWeight->print();

			//		cout << endl;

			//		cout << "output: " << output << endl;
			//		cout << endl;
		}
		// sum_in = sum_in / batch_size;
		for (int i = 0; i < sum_in->dim()[0]; i++)
		{
			for (int j = 0; j < sum_in->dim()[1]; j++)
			{
				sum_in->n2Arr[i][j] = sum_in->n2Arr[i][j] / (float)batch_size;
			}
		}
		for (int i = 0; i < sum_out->dim()[0]; i++)
		{
			for (int j = 0; j < sum_out->dim()[1]; j++)
			{
				sum_out->n2Arr[i][j] = sum_out->n2Arr[i][j] / (float)batch_size;
			}
		}
		inputToHideWeight = matAdd(*inputToHideWeight, *sum_in);
		hideToOutputWeight = matAdd(*hideToOutputWeight, *sum_out);
		// cout << "fuck" << endl;
	}

	delete tmp;
	delete tmpTrans;
	delete tmpWeight;
	delete outHBias;
}
float ANN::predict(vector<float> in)
{

	MyMatrix<float> *outHBias = new MyMatrix<float>(11, 1);

	for (int i = 0; i < in.size(); i++)
	{ // input data into input matrix
		input->n2Arr[i][0] = in[i];
	}

	int foo = input->dim()[0] - 1;
	input->n2Arr[foo][0] = 1; //set bias

	//		cout << "input: " << endl;
	//		input->print();
	//		cout << endl;

	netH->arr[0] = matMatMul(*inputToHideWeight, *input); //forward propagation

	//		cout << "netH: " << endl;
	//netH->print();
	//		cout << endl;

	for (int i = 0; i < netH->arr[0]->dim()[0]; i++)
	{ //change netH to outH, outBias is out matrix plus bias
		outH->n2Arr[i][0] = sigmoid(netH->arr[0]->n2Arr[i][0]);
		outHBias->n2Arr[i][0] = sigmoid(netH->arr[0]->n2Arr[i][0]);
	}

	//		cout << "outH: " << endl;
	//outH->print();
	//		cout << endl;

	foo = outHBias->dim()[0] - 1;

	outHBias->n2Arr[foo][0] = 1; //complete the outBias

	//		cout << "outHBias: " << endl;
	//outHBias->print();
	//		cout << endl;

	output = matMatMul(*hideToOutputWeight, *outHBias);

	//		cout << "output: " << endl;
	//tmp->print();
	//		cout << endl;

	for (int i = 0; i < output->dim()[0]; i++)
	{
		output->n2Arr[i][0] = sigmoid(output->n2Arr[i][0]); //finish forward propagation
	}

	float max = output->n2Arr[0][0]; //change the output matrix into the real result
	for (int i = 1; i < output->dim()[0]; i++)
	{
		if (output->n2Arr[i][0] > max)
			max = output->n2Arr[i][0];
	}

	delete outHBias;
	return max;

	//	return output;
}

void ANN::storeWeight()
{
	hideToOutputWeight->out();
	inputToHideWeight->out();
}
void ANN::loadWeight()
{

	vector<MyMatrix<float> *> load;
	MyMatrix<float>::in(load);

	hideToOutputWeight = load[0];
	inputToHideWeight = load[1];
}
