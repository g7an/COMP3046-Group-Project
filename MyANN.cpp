// with lr decay; activation function :sigmoid 
#include "MyANN.h"
//#include "MyVector.h"
//#include "MyVector.cpp"
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
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <random>
using namespace std;

template<class T >
MyMatrix<T>* d_Relu(MyMatrix<T> &x);

template <class T>
void updateDelta_bias(MyMatrix<T> &x, float *bias);

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

template <class T>
MyMatrix<T> *eleMul(MyMatrix<T> &x, MyMatrix<T> &y);

template <class T>
MyMatrix<T> *d_sigmoid(MyMatrix<T> &x);

template<class T >
MyMatrix<T>* d_CrossEntropy(MyMatrix<T> &t,MyMatrix<T> &x);

float MyANN::normalDis(){
	std::default_random_engine e; //引擎
	std::normal_distribution<double> n(0, 1); //均值, 方差
	return n(e); 
}
float MyANN::sigmoid(float input)
{ //calculate sigmoid function
	return 1 / (1 + exp(-input));
}

float MyANN::Relu(float input)
{ //calculate sigmoid function
	return input > 0 ? input : 0;
}

inline float MyANN::random()
{ //return a float random number between 0 and 1
	float r = rand() / 50.0f;
	//float r = rand();
	return r / RAND_MAX;
}

void MyANN::setEpochs(int e)
{
	epochs = e;
}

void MyANN::setLR(float f)
{
	lr = f;
}
MyANN::MyANN(float lr, int epochs, int batch_size, int decayEpoch, float decay) : lr(lr), epochs(epochs), batch_size(batch_size), decayEpoch(decayEpoch), decay(decay)
{
	srand((unsigned)time(NULL));

	num_hidLayer = 1;
	int num_weights = num_hidLayer + 1;

	total_neurons = new int[num_hidLayer + 2]; // num nodes in all layers
	for (int i = 1; i <= num_hidLayer; i++)
	{
		total_neurons[i] = 100; //num_neurons[i];
	}
	total_neurons[0] = 784;
	total_neurons[num_hidLayer + 1] = 10;

	hidWeight.reserve(num_weights);
	delta_w.reserve(num_weights);
	outH.reserve(num_weights); // it include from first hidden layer to output layer

	bias.reserve(num_weights); //the bias initialize
	delta_bias.reserve(num_weights);


	for (int i = 0; i < num_weights; i++)
	{
		hidWeight[i] = new MyMatrix<float>(total_neurons[i + 1], total_neurons[i]);
		delta_w[i] = new MyMatrix<float>(total_neurons[i + 1], total_neurons[i]);

		//cout<<"hidWeight: "<<hidWeight[i]->dim()[0]<<" : "<<hidWeight[i]->dim()[1]<<endl;

		for (int r = 0; r < hidWeight[i]->dim()[0]; r++)
		{
			for (int c = 0; c < hidWeight[i]->dim()[1]; c++)
			{
				hidWeight[i]->n2Arr[r][c] =normalDis(); //random();
			}
		}
	}



	for (int i = 0; i < num_weights; i++)
	{

		bias[i] = new float[total_neurons[i + 1]];
		delta_bias[i] = new float[total_neurons[i + 1]];

		for (int j = 0; j < total_neurons[i + 1]; j++)
		{
			bias[i][j] = normalDis(); 
		}

		outH[i] = new MyMatrix<float>(total_neurons[i + 1], 1); // REMIND: the outH doesn't include input now!
	}


	input = new MyMatrix<float>(784, 1);
	target = new MyMatrix<float>(10, 1); // ground-truth
}

MyANN::~MyANN()
{
	delete netH;
	delete input;
	delete target;
	delete[] total_neurons;
}

void MyANN::train(vector<vector<float> > in, vector<float> t)
{
	const int steps = in.size() / batch_size + 1; // how many batches to finish an epoch
	MyMatrix<float> *tmp = NULL;
	MyMatrix<float> *tmpTrans = NULL; //Li: 这个编译不过，就initialize 成了NULL
	float testLoss = 0;


	for (int epoch = 0; epoch < epochs; epoch++)
	{
		if (epoch % decayEpoch == 0 && epoch != 0){
			lr = lr * decay;
			cout<<"decay"<<endl;
		}
		for (int round = 0; round < steps; round++)
		{
			for (int turn = 0; turn < batch_size; turn++)
			{

				if (turn + round * batch_size == in.size())
					break; //Li: To test whether the index is out of bound

				partError = new MyMatrix<float>(10, 1); //Li:对于每一个数据有一个partError

				for (int i = 0; i < in[0].size(); i++)
				{ //deal with input
					input->n2Arr[i][0] = in[round * batch_size + turn][i];
				}

				for (int i = 0; i < 10; i++)
				{
					target->n2Arr[i][0] = t[round * batch_size + turn] == i ? 1 : 0; //deal with target;
				}

				MyMatrix<float> *net;


				for (int i = 0; i < num_hidLayer + 1; i++)
				{ //forward begin
					if (i == 0)
					{
						net = input;
					}
					else
					{
						net = outH[i - 1];
					}

					netH = matMatMul(*hidWeight[i], *net);

					for (int j = 0; j < netH->dim()[0]; j++)
					{
						outH[i]->n2Arr[j][0] = sigmoid(netH->n2Arr[j][0] + bias[i][j]); // need to plus bias here!
						//outH[i]->n2Arr[j][0] = Relu(netH->n2Arr[j][0] + bias[i][j]); // need to plus bias here!
					}
					delete netH;
				} //forward end;


				//tmp = matSub(*outH[num_hidLayer], *target); // outH[num_hidLayer] is the output now
				tmp = d_CrossEntropy(*target,*outH[num_hidLayer]); // Li: Change the loss function to cross entropy



				//cout<<"tmp: "<<endl;
				//tmp->print();
				//cout<<endl;
				//float foo;

				for (int j = 0; j < outH[num_hidLayer]->dim()[0]; j++)
				{                                                                        //Li: Now the outH[num_hidLayer+1] means the output layer
					float outTmp = outH[num_hidLayer]->n2Arr[j][0];                      //Li: Now the outH[num_hidLayer+1] means the output layer
					partError->n2Arr[j][0] = outTmp * (1 - outTmp) * (tmp->n2Arr[j][0]);  // out / net = out(1-out)  //Li: Maybe something wrong with the subscript of partError and tmp layer?
					/*
					   foo = outTmp > 0 ? 1 : 0;
					   partError->n2Arr[j][0] = foo * (tmp->n2Arr[j][0]);

					 */
				}                                                                        // L-1 ~ 2: L - 2 层 = num_hidlayer

				delete tmp;

				updateDelta_bias(*partError, delta_bias[num_hidLayer]);

				MyMatrix<float> *dSigmoid = NULL;
				MyMatrix<float> *tmpLoss = NULL;
				MyMatrix<float> *tmp2 = NULL;


				for (int i = num_hidLayer; i >= 0; i--)
				{
					if (i == 0)
					{
						net = input;
					}
					else
					{
						net = outH[i - 1];
					}

					if (i != num_hidLayer)
					{

						//cout<<"hidWeight: "<<hidWeight[i]->dim()[0]<<" : "<<hidWeight[i]->dim()[1]<<endl;

						tmpTrans = hidWeight[i+1]->transpose();



						tmpLoss = matMatMul(*tmpTrans, *partError);
						delete partError;

						//cout<<"fuc in1"<<endl;
						//dSigmoid = d_Relu(*outH[i]);
						dSigmoid = d_sigmoid(*outH[i]);

						//cout<<"dSigmoid: "<<dSigmoid->dim()[0]<<" : "<<dSigmoid->dim()[1]<<endl;
						//cout<<"tmpLoss: "<<tmpLoss->dim()[0]<<" : "<<tmpLoss->dim()[1]<<endl;

						partError = eleMul(*dSigmoid, *tmpLoss);

						updateDelta_bias(*partError, delta_bias[i]);

						//cout<<"fuc in2"<<endl;
						delete tmpLoss;
					}

					delete tmpTrans; //xxxxxx

					tmpTrans = net->transpose();

					tmp = matMatMul(*partError, *tmpTrans);

					tmp2 = delta_w[i];
					delta_w[i] = matAdd(*tmp, *tmp2);

					delete tmp2;
					delete dSigmoid;
					delete tmp;
				}
				//cout<<"fuc3"<<endl;
			}

			for (int i = 0; i < num_hidLayer + 1; i++)
			{
				delta_w[i]->smul(lr / (float)batch_size); //refresh weight
				tmp = hidWeight[i];
				hidWeight[i] = matSub(*tmp, *delta_w[i]);
				delete tmp;


				for (int j = 0; j < total_neurons[i + 1]; j++) //refresh bias
				{
					bias[i][j] -= (lr / (float)batch_size) * delta_bias[i][j];

				}
			}

		}

		//cout<<"totalLoss "<<totalLoss(in[0],t[0])<<endl;;
		cout<<"target:  "<<t[0]<<endl;;
		cout<<"totalLoss(Cross) "<<lossWithCrossE(in[0],t[0])<<endl;;
		/*
		   cout<<"output"<<endl;
		   outH[num_hidLayer]->print();
		   cout<<endl;
		 */

	}
}

MyMatrix<float>* MyANN::forward(vector<float> in){

	for (int i = 0; i < in.size(); i++)
	{ //deal with input
		input->n2Arr[i][0] = in[i];
	}

	MyMatrix<float> *net;

	for (int i = 0; i < num_hidLayer + 1; i++)
	{ //forward begin
		if (i == 0)
		{
			net = input;
		}
		else
		{
			net = outH[i - 1];
		}

		netH = matMatMul(*hidWeight[i], *net);

		for (int j = 0; j < netH->dim()[0]; j++)
		{
			outH[i]->n2Arr[j][0] = sigmoid(netH->n2Arr[j][0] + bias[i][j]); // need to plus bias here!
			//outH[i]->n2Arr[j][0] = Relu(netH->n2Arr[j][0] + bias[i][j]); // need to plus bias here!
		}
		delete netH;
	} //forward end;
	return outH[num_hidLayer];

}

float MyANN::lossWithCrossE(vector<float> in, float t){

	for (int i = 0; i < 10; i++)
	{
		target->n2Arr[i][0] = t == i ? 1 : 0; //deal with target;
	}

	MyMatrix<float> *tmp =forward(in);
	tmp->print();

	float tar;
	float pre;

	float loss = 0;
	for(int i=0; i<10;i++){

		tar= target->n2Arr[i][0];
		pre= tmp->n2Arr[i][0];

		loss += -tar*log(pre) - (1-tar) * log(1-pre);
	}

	return loss;

}
float MyANN::totalLoss(vector<float> in, float t)
{

	for (int i = 0; i < 10; i++)
	{
		target->n2Arr[i][0] = t == i ? 1 : 0; //deal with target;
	}

	MyMatrix<float> *tmp =forward(in);

	float foo = 0;
	float loss = 0;
	for(int i=0; i<10;i++){
		foo = target->n2Arr[i][0] - tmp->n2Arr[i][0];
		loss += foo * foo;
	}
	return 0.5*loss;
	/*
	   for (int i = 0; i < in.size(); i++)
	   { //deal with input
	   input->n2Arr[i][0] = in[i];
	   }


	   MyMatrix<float> *net;

	   for (int i = 0; i < num_hidLayer + 1; i++)
	   { //forward begin
	   if (i == 0)
	   {
	   net = input;
	   }
	   else
	   {
	   net = outH[i - 1];
	   }

	   netH = matMatMul(*hidWeight[i], *net);

	   for (int j = 0; j < netH->dim()[0]; j++)
	   {
	   outH[i]->n2Arr[j][0] = sigmoid(netH->n2Arr[j][0] + bias[i][j]); // need to plus bias here!
	//outH[i]->n2Arr[j][0] = Relu(netH->n2Arr[j][0] + bias[i][j]); // need to plus bias here!
	}
	delete netH;
	} //forward end;
	 */

}

float MyANN::predict(std::vector<float>){
	return 0;
}

void MyANN::storeWeight(){

}
void MyANN::loadWeight(){

}
