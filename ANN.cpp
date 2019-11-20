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

template<class T >
MyMatrix<T>* eleMul(MyMatrix<T> &x, MyMatrix<T>&y);

template<class T >
MyMatrix<T>* d_sigmoid(MyMatrix<T> &x);

/*
int main(void){
srand((unsigned)time(NULL));
float lr = 0.5;							//learning rate



return 0;
}
*/

ANN::ANN(float lr, int epochs, int batch_size, int decayEpoch, float decay) : lr(lr), epochs(epochs), batch_size(batch_size), decayEpoch(decayEpoch), decay(decay)
{
	// cout << "Input number of hidden layer: " << endl;
	// cin >> num_hidLayer;
	// cout << "fuck" << endl;
	srand((unsigned)time(NULL));
	num_hidLayer = 3;
	// const int layer = num_hidLayer;
	int prevLayer = 784;
	//num_neurons = new int[num_hidLayer];
	int* total_neuron = new int[num_hidLayer + 1];
	for (int i = 0; i < num_hidLayer; i++)
	{
		//num_neurons[i] = 10;
		// 	ele += num_neurons[i] * (prevLayer + 1);
		// 	prevLayer = num_neurons[i];
		total_neuron[i] = 10; //num_neurons[i];
	}

	// (hidden layer + 1) 层 weight matrix，每个里面neuron个元素
	const int num_matrix = num_hidLayer + 1;
	const int num_allLayers = num_hidLayer + 2;                //Li: 所有层的数量
	total_neuron[num_hidLayer] = 10;

	// cout << num_matrix << endl;
	hidWeight = new MyVector<MyMatrix<float> *>(num_matrix); // row 0: input_to_hidden_weight; row_last: hide_to_output_weight
	//hidWeight = new MyVector<void *>(num_matrix); // row 0: input_to_hidden_weight; row_last: hide_to_output_weight
	// initialization of weights
	netH = new MyVector<MyMatrix<float> *>(num_allLayers);
	outH = new MyVector<MyMatrix<float> *>(num_allLayers);
	outHBias = new MyVector<MyMatrix<float> *>(num_allLayers);
	tmpWeight = new MyVector<MyMatrix<float> *>(num_matrix);
	delta_w = new MyVector<MyMatrix<float> *>(num_matrix);
	//partError = new MyVector<MyMatrix<float> *>(num_matrix);                                   // Li: partError 是不是只有一个就行？   
	for (int i = 0; i < (num_hidLayer + 1); i++)                                       //Li: 因为delta_w 的大小和hidWeight 大小相同，因此把train 中的初始化移动到这里来
	{
		hidWeight->arr[i] = new MyMatrix<float>(total_neuron[i], prevLayer + 1);
		delta_w->arr[i] = new MyMatrix<float>(total_neuron[i], prevLayer + 1);
		tmpWeight->arr[i] = new MyMatrix<float>(total_neuron[i], prevLayer);           //Li: tmpWeight is one column less than other two weight,so preLayer do not need to +1

		cout << i << ": " << hidWeight->arr[i]->dim()[0] << " " << hidWeight->arr[i]->dim()[1] << endl;
		for (int j = 0; j < hidWeight->arr[i]->dim()[0]; j++)
		{
			for (int k = 0; k < hidWeight->arr[i]->dim()[1]; k++)
			{
				hidWeight->arr[i]->n2Arr[j][k] = random();
				// cout << i << endl;
			}
		}
		cout << "neuron" << i << ": " << total_neuron[i] << endl;
		prevLayer = total_neuron[i];
		cout << "prevLayer; " << prevLayer << endl;
	}

	for (int i = 0; i < num_hidLayer + 1; i++){                                               //Li: initialize all hidden layer here
		outH->arr[i + 1] = new MyMatrix<float>(total_neuron[i], 1);                           //Li: use i+1 as 0 is for input
		outHBias->arr[i + 1] = new MyMatrix<float>(total_neuron[i] + 1, 1);
	}
	outH->arr[num_allLayers + 1] = new MyMatrix<float>(10, 1);                           // Li: There two lines for initialize the output layer
	//inputToHideWeight = new MyMatrix<float>(10, 785);
	input = new MyMatrix<float>(785, 1);
	target = new MyMatrix<float>(10, 1); // ground-truth
	delete[] total_neuron;

}

ANN::~ANN()
{
	delete  tmpWeight;
	delete  hidWeight;
	delete  delta_w;

	delete  netH;
	delete  outH;
	delete  outHBias;

	//delete partError;

	delete input;
	delete target;

	//delete partError;
}

float ANN::sigmoid(float input)
{ //calculate sigmoid function
	return 1 / (1 + exp(-input));
}

inline float ANN::random()
{ //return a float random number between 0 and 1
	float r = rand() / 50.0f;
	return r / RAND_MAX;
}

void ANN::setEpochs(int e)
{
	epochs = e;
}

void ANN::setLR(float f)
{
	lr = f;
}

void ANN::train(vector<vector<float> > in, vector<float> t)
{
	const int vector_size = in[0].size();
	/*
	float test = inputToHideWeight->n2Arr[0][0];
	if (test < -100000 || test > 100000)
	{
	initializeWeight();
	}
	*/

	//const int batch_size = 10;
	const int steps = in.size() / batch_size + 1; // how many batches to finish an epoch
	MyMatrix<float> *tmp = NULL;
	MyMatrix<float> *tmpTrans = NULL;                 //Li: 这个编译不过，就initialize 成了NULL
	float testLoss = 0;
	/*
		MyMatrix<float> *sum_in = NULL;
		MyMatrix<float> *sum_out = NULL;
		*/
	vector<float> batch_data(vector_size);

	for (int epoch = 0; epoch < epochs; epoch++){
		if (epoch % decayEpoch == 0 && epoch != 0)
			lr = lr*decay;
		for (int round = 0; round < steps; round++)
		{
			for (int turn = 0; turn < batch_size; turn++)
			{ //training begin
				// load in one data for training

				if (turn + round * batch_size == in.size()) break;     //Li: To test whether the index is out of bound

				partError = new MyMatrix<float>(10, 1);                                 //Li:对于每一个数据有一个partError

				batch_data = in[turn + round * batch_size];
				//#pragma omp parallel for num_threads(4)

				for (int i = 0; i < 10; i++)
				{
					i == t[turn + round * batch_size] ? target->n2Arr[i][0] = 1 : target->n2Arr[i][0] = 0; //make single target number into a target matrix
				}
				//#pragma omp parallel for num_threads(4)
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
				outH->arr[0] = input; // first layer: input = output = data_in 
				outHBias->arr[0] = input;                                                   //Li: I remove the comment in this line, as in backpro the outHBias[0] will be used

				MyMatrix<float> *netWeight = hidWeight->arr[0];
				MyMatrix<float> *net;
				for (int i = 1; i < num_hidLayer + 2; i++){                                 //Li: 这里改成了是+2 让outH 做到输出层
					if (i == 1)
						net = input;
					else
						net = outHBias->arr[i - 1];
					//cout << net->dim()[0] << " " << net->dim()[1] << " " << netWeight->dim()[0] << " " << netWeight->dim()[1] << endl;
					
					netH->arr[i] = matMatMul(*netWeight, *net);// 10 * 1
					cout << "netH" << i << " dim " << netH->arr[i]->dim()[0] << ", " << netH->arr[i]->dim()[1] << endl;
					//				outH->arr[i] = netH->arr[i];                                              //Li: I've already initialize the outH and outBias in construtor, so there two lines are useless
					//outHBias->arr[i] = new MyMatrix<float>(netH->arr[i]->dim()[0] + 1, 1);
					for (int j = 0; j < netH->arr[i]->dim()[0]; j++){ //change netH to outH, outBias is out matrix plus bias
						outH->arr[i]->n2Arr[j][0] = sigmoid(netH->arr[i]->n2Arr[j][0]);
						outHBias->arr[i]->n2Arr[j][0] = sigmoid(netH->arr[i]->n2Arr[j][0]);
					}
					if (i == num_hidLayer + 1)                                               //Li: Now the outH[num_hidLayer+1] means the output layer
						break;
					foo = outHBias->arr[i]->dim()[0] - 1;
					outHBias->arr[i]->n2Arr[foo][0] = 1; //complete the outBias

					//netH->arr[i + 1] = outHBias->arr[i];                                  //Li: Why we need to use netH to store outBias? 
					netWeight = hidWeight->arr[i];
					// delete netH->arr[i];
				}														// Li: 不是很清楚这段代码的作用，就comment 掉了

				// add bias to outHBias
				tmp = matSub(*outH->arr[num_hidLayer + 1], *target); // (a - y) in last layer                //Li: Now the outH[num_hidLayer+1] means the output layer
				testLoss = 0;

				for (int i = 0; i < 10; i++){
					testLoss += tmp->n2Arr[i][0] * tmp->n2Arr[i][0]*0.5;
					
				}

				//outH->arr[num_hidLayer] = tmp; // (a - y) in last layer     //Li: 我觉得最后一层的outH 是需要在反传中使用的，就把这行comment 掉了
				/*
					backprop function:
					1. error in last layer
					2. calculate error (store) in every layer from L to 2
					3. update weights & bias in each layer
					*/


				for (int dim = 0; dim < num_hidLayer + 1; dim++){
					for (int i = 0; i < hidWeight->arr[dim]->dim()[0]; i++){
						for (int j = 0; j < (hidWeight->arr[dim]->dim()[1] - 1); j++){
							tmpWeight->arr[dim]->n2Arr[i][j] = hidWeight->arr[dim]->n2Arr[i][j]; // last hidden weight without bias
						}
					}
				}

				for (int j = 0; j < netH->arr[num_hidLayer + 1]->dim()[0]; j++){			 //Li: Now the outH[num_hidLayer+1] means the output layer	
					float outTmp = netH->arr[num_hidLayer + 1]->n2Arr[j][0];                //Li: Now the outH[num_hidLayer+1] means the output layer
					partError->n2Arr[j][0] = outTmp * (1 - outTmp) * (tmp->n2Arr[j][0]); // out / net = out(1-out)  //Li: Maybe something wrong with the subscript of partError and tmp layer?

				} // L-1 ~ 2: L - 2 层 = num_hidlayer
				
				delete tmp;

				/*
				for(int i = num_hidLayer; i > 0; i--){
				MyMatrix<float>* dummy = matMatMul(*tmpWeight->arr[i], *partError->arr[i]);
				partError->arr[i - 1] = dummy;
				}
				// calculate aggregate delta_w: from first weight to last weight
				// 第一层input * 第二层 partError + ... + 第（n-1）层activation output * 第n层partError; store in a sum function
				for(int i = 0; i < num_hidLayer + 1; i++){
				tmpTrans = outHBias->arr[i]->transpose();
				tmp = matMatMul(*partError->arr[i], *tmpTrans);

				delta_w->arr[i] = new MyMatrix<float>(tmp->dim()[0], tmp->dim()[1]); //? put this to constructor

				delta_w->arr[i] = matAdd(*tmp, *delta_w->arr[i]);
				}
				*/																			//Li: original code

				MyMatrix<float>* dSigmoid = NULL;
				MyMatrix<float>* tmpLoss= NULL;
				MyMatrix<float>* tmp2= NULL;


				for (int i = num_hidLayer; i >= 0; i--){                                 //Li: Now the outH[num_hidLayer+1] means the output layer and outH[0] is input	
				cout << "FUck" << endl;
					if (i != num_hidLayer){												//Li: 如果 i = num_hidLayer+1 , part partError 已经在partError 中，因此直接计算delta_w

						tmpTrans = tmpWeight->arr[i + 1]->transpose();

				
						tmpLoss = matMatMul(*tmpTrans, *partError);									// Li: partError 对上层输入的偏导
						
						delete partError;
						cout << "i is " << i << " tmpLoss: " << tmpLoss->dim()[0] << ", " << tmpLoss->dim()[1] << endl;
						cout << "netH: " << netH->arr[i + 1]->dim()[0] << ", " << netH->arr[i + 1]->dim()[1] << endl;

						

						dSigmoid = d_sigmoid(*netH->arr[i + 1]);                            // Li: out 对 net 的偏导
						
						cout << "dSigmoid: " << dSigmoid->dim()[0] << ", " << dSigmoid->dim()[1] << endl;

						partError = eleMul(*tmpLoss, *dSigmoid);													//Li: partError 对 net 的偏导, 本层的partError(partError) 计算完成 P29 第一个式子
						cout << "NM$L" << endl;
						delete tmpLoss;


					}
				cout << "fuCK" << endl;
				
					delete tmpTrans;      //xxxxxx

					tmpTrans = outHBias->arr[i]->transpose();


					tmp = matMatMul(*partError, *tmpTrans);

					//delete tmpTrans;      //xxxxxx

					tmp2 = delta_w->arr[i];
					delta_w->arr[i] = matAdd(*tmp, *tmp2);

					delete tmp2;
					delete dSigmoid;
					delete tmp;
				}
				//one data train finish
				delete partError;
			}


			for (int i = 0; i < num_hidLayer + 1; i++){
				delta_w->arr[i]->smul(lr / (float)batch_size);            //Li: I've write the function of matrix mult number, so...
			}


			// update weight and bias
			for (int i = 0; i < num_hidLayer + 1; i++){
				tmp = hidWeight->arr[i];
				hidWeight->arr[i]= matSub(*tmp, *delta_w->arr[i]);
				delete tmp;
			}


		}

			float tmpLoss = totalLoss(in[89], t[89]);
			cout << "epoch: " << epoch << "total Loss: " << tmpLoss << endl;
			cout << "output layer: " << endl;
			outH->arr[num_hidLayer + 1]->print();

			//this->storeWeight();
	}

	//delete tmp;
	delete tmpTrans;

}
 float ANN::predict(vector<float> in)
 {

			int vector_size = in.size();
			cout << "in: " << endl;
			for (int i = 0; i < vector_size; i++)
			{
				input->n2Arr[i][0] = in[i];
				cout << in[i] << " ";
			}
			cout << endl;
			int foo = input->dim()[0] - 1;
			input->n2Arr[foo][0] = 1; //set bias			

			netH->arr[0] = input;
			outH->arr[0] = input; // first layer: input = output = data_in 
			outHBias->arr[0] = input;                                                   //Li: I remove the comment in this line, as in backpro the outHBias[0] will be used

			MyMatrix<float> *netWeight = hidWeight->arr[0];
			MyMatrix<float> *net;
			for (int i = 1; i < num_hidLayer + 2; i++){                                  //Li: 这里改成了是+2 让outH 做到输出层
				if (i == 1)
					net = input;
				else
					net = outHBias->arr[i - 1];
				netH->arr[i] = matMatMul(*netWeight, *net);// 10 * 1

				cout << "netH" << i << ": "<< endl;
				netH->arr[i]->print();

				for (int j = 0; j < netH->arr[i]->dim()[0]; j++){ //change netH to outH, outBias is out matrix plus bias
					outH->arr[i]->n2Arr[j][0] = sigmoid(netH->arr[i]->n2Arr[j][0]);
					cout << netH->arr[i]->n2Arr[j][0] << " "; 
					cout << "\nsigmoid: " << sigmoid((double)netH->arr[i]->n2Arr[j][0]) << endl; 

					outHBias->arr[i]->n2Arr[j][0] = sigmoid(netH->arr[i]->n2Arr[j][0]);

				}
				cout << endl;
				cout << "layer " << i << ": " << endl;
				outH->arr[i]->print();

				if (i == num_hidLayer + 1)                                               //Li: Now the outH[num_hidLayer+1] means the output layer
					break;
				foo = outHBias->arr[i]->dim()[0] - 1;
				outHBias->arr[i]->n2Arr[foo][0] = 1; //complete the outBias

				netWeight = hidWeight->arr[i];
			}

			int pos= 0;
			cout << "output:" << endl;
			outH->arr[num_hidLayer + 1]->print();

			float max= outH->arr[num_hidLayer + 1]->n2Arr[0][0] ;
			for (int i = 0; i < 10; i++){
				if (outH->arr[num_hidLayer + 1]->n2Arr[i][0] > max){
					max = outH->arr[num_hidLayer + 1]->n2Arr[i][0];
					pos = i;
				}														// (a - y) in last layer                //Li: Now the outH[num_hidLayer+1] means the output layer
			}
			cout << "\nsigmoid: " << sigmoid(77.0628) << endl; 
			return pos;

 }

 float ANN::totalLoss(vector<float> in ,float target)
 {

			partError = new MyMatrix<float>(10, 1);                                 //Li:对于每一个数据有一个partError
			vector<float> tmpTarget;
			for (int i = 0; i < 10; i++){
				i == target ? tmpTarget.push_back(1) : tmpTarget.push_back(0);
			}
			

			/* feed forward function:
				input: vector to store input data for this turn
				netH: vector of matrix to store the net output (wx + b) in each node
				netWeight: matrix to store the weight of one layer
				net: matrix to store net value of one layer
				outH: vector of matrix to store sigmoid output of a node
				outHBias: vector of matrix to store sigmoid output plus bias
				recursively do matrix multiplication and sigmoid calculation until the last layer
				*/
			int vector_size =in.size();
			for (int i = 0; i < vector_size; i++)
			{
				input->n2Arr[i][0] = in[i];
			}

			int foo = input->dim()[0] - 1;
			input->n2Arr[foo][0] = 1; //set bias			

			netH->arr[0] = input;
			outH->arr[0] = input; // first layer: input = output = data_in 
			outHBias->arr[0] = input;                                                   //Li: I remove the comment in this line, as in backpro the outHBias[0] will be used

			MyMatrix<float> *netWeight = hidWeight->arr[0];
			MyMatrix<float> *net;
			for (int i = 1; i < num_hidLayer + 2; i++){                                  //Li: 这里改成了是+2 让outH 做到输出层
				if (i == 1)
					net = input;
				else
					net = outHBias->arr[i - 1];
				//cout << net->dim()[0] << " " << net->dim()[1] << " " << netWeight->dim()[0] << " " << netWeight->dim()[1] << endl;
				netH->arr[i] = matMatMul(*netWeight, *net);// 10 * 1
				//				outH->arr[i] = netH->arr[i];                                              //Li: I've already initialize the outH and outBias in construtor, so there two lines are useless
				//outHBias->arr[i] = new MyMatrix<float>(netH->arr[i]->dim()[0] + 1, 1);

				for (int j = 0; j < netH->arr[i]->dim()[0]; j++){ //change netH to outH, outBias is out matrix plus bias
					outH->arr[i]->n2Arr[j][0] = sigmoid(netH->arr[i]->n2Arr[j][0]);
					outHBias->arr[i]->n2Arr[j][0] = sigmoid(netH->arr[i]->n2Arr[j][0]);
				}
				if (i == num_hidLayer + 1)                                               //Li: Now the outH[num_hidLayer+1] means the output layer
					break;
				foo = outHBias->arr[i]->dim()[0] - 1;
				outHBias->arr[i]->n2Arr[foo][0] = 1; //complete the outBias

				//netH->arr[i + 1] = outHBias->arr[i];                                  //Li: Why we need to use netH to store outBias? 
				netWeight = hidWeight->arr[i];
			}
			/*
			// first outHBias: input + bias
			outHBias->arr[0] = new MyMatrix<float>(785, 1);
			outHBias->arr[0]->n2Arr[vector_size][0] = 1;                 // input lack of a bias
			int count;
			for(int i = 0; i < vector_size; i++){
			outHBias->arr[0]->n2Arr[i][0] = input->n2Arr[i][0];
			}
			*/															// Li: 不是很清楚这段代码的作用，就comment 掉了

			// add bias to outHBias
			float error = 0;
			for (int i = 0; i < 10; i++){
				float foo = tmpTarget[i] - outH->arr[num_hidLayer + 1]->n2Arr[i][0];
				error += foo*foo;
			}
			return error*0.5;

 }
void ANN::storeWeight()
{
	for (int i = 0; i < hidWeight->size(); i++){
		hidWeight->arr[i]->out();
	}
}
void ANN::loadWeight()
{
	vector<MyMatrix<float> *> load;
	MyMatrix<float>::in(load);


	for (int i = 0; i < hidWeight->size(); i++){
		hidWeight->arr[i]= load[i];
	}

	/*
	hideToOutputWeight = load[0];
	inputToHideWeight = load[1];
	*/
}
