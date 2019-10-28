#include "ANN.h"

//#include"MyVector.h"
//#include"MyVector.cpp"
#include "MyMatrix.h"
#include "MyMatrix.cpp"
#include"globalFunctions.cpp"

#include<iostream>
#include<fstream>
#include<iostream>
#include<string>
#include<queue>
#include<vector>
#include<cstdio>
#include<algorithm>
#include<cmath>
#include<stdlib.h>
#include<time.h>
#include<omp.h>
using namespace std;


template<class T, class N>
MyVector<T>* vecAdd(MyVector<T> &x, MyVector<T> &y, N a);


template<class T>
T vecDot(MyVector<T> &x, MyVector<T> &y);


template<class T >
MyMatrix<T>* matAdd(MyMatrix<T> &x, MyMatrix<T>&y);

template<class T >
MyMatrix<T>* matSub(MyMatrix<T> &x, MyMatrix<T>&y);


template<class T >
MyMatrix<T>* matVecMul(MyMatrix<T> &x, MyVector<T>&y);

template<class T >
MyMatrix<T>* vecMatMul(MyVector<T>&x, MyMatrix<T> &y);

template<class T >
MyMatrix<T>* matMatMul(MyMatrix<T> &x, MyMatrix<T>&y);



/*
int main(void){
srand((unsigned)time(NULL));
float fai = 0.5;							//learning rate



return 0;
}
*/

ANN::ANN(float fai = 0.5, int epochs = 1000) :fai(fai), epochs(epochs)
{
	inputToHideWeight = new MyMatrix<float>(10, 785);
	input = new MyMatrix<float>(785, 1);
	netH = new MyMatrix<float>(10, 1);
	outH = new MyMatrix<float>(10, 1);
	hideToOutputWeight = new MyMatrix<float>(10, 11);
	output = new MyMatrix<float>(10, 1);
	target = new MyMatrix<float>(10, 1);
	partError = new MyMatrix<float>(10, 1);

}


ANN::~ANN()
{
	delete inputToHideWeight;
	delete input;
	delete netH;
	delete outH;
	delete hideToOutputWeight;
	delete output;
	delete target;
	delete partError;
}

inline float ANN::sigmoid(float input){				//calculate sigmoid function

	return 1 / (1 + exp(-input));
}

inline float ANN::random(){								//return a float random number between 0 and 1
	return rand() / float(RAND_MAX);
}

void ANN::setEpochs(int e){
	epochs = e;
}

void ANN::setFai(float f){
	fai = f;
}

void ANN::initializeWeight(){
	srand((unsigned)time(NULL));

	//	cout << "first weight" << endl;
	for (int i = 0; i < inputToHideWeight->dim()[0]; i++){
		for (int j = 0; j < inputToHideWeight->dim()[1]; j++){
			inputToHideWeight->n2Arr[i][j] = random();
			//			cout << inputToHideWeight->n2Arr[i][j] << " ";
		}
		//		cout << endl;
	}


	//	cout << "second weight" << endl;
	for (int i = 0; i < hideToOutputWeight->dim()[0]; i++){
		for (int j = 0; j < hideToOutputWeight->dim()[1]; j++){
			hideToOutputWeight->n2Arr[i][j] = random();
			//			cout << hideToOutputWeight->n2Arr[i][j] << " ";

		}
		//		cout << endl;

	}
}

void ANN::train(vector<vector<float>> in, vector<float> t){

	float test = inputToHideWeight->n2Arr[0][0];
	if (test < -100000 || test>100000){
		initializeWeight();
	}

	cout << endl;

	const int bach = 256;


	MyMatrix<float> *outHBias = new MyMatrix<float>(11, 1);
	MyMatrix<float> *tmp = NULL;
	MyMatrix<float> *tmpWeight = new MyMatrix<float>(10, 10);
	MyMatrix<float> *tmpTrans = NULL;


	for (int round = 0; round < epochs; round++){


		for (int turn = 0; turn < bach; turn++){              //training begin

#pragma omp parallel for num_threads(4) 
			for (int i = 0; i < 10; i++){
				i == t[turn] ? target->n2Arr[i][0] = 1 : target->n2Arr[i][0] = 0;   //make single target number into a target matrix
//				cout << "target: " << target->n2Arr[i][0] << endl;
			}


#pragma omp parallel for num_threads(4) 
			for (int i = 0; i < in[turn].size(); i++){			// input data into input matrix
				input->n2Arr[i][0] = in[turn][i];
			}

			int foo = input->dim()[0] - 1;
			input->n2Arr[foo][0] = 1;                            //set bias


			//		cout << "input: " << endl;
			//		input->print();
			//		cout << endl;

			netH = matMatMul(*inputToHideWeight, *input);			//forward propagation

			//		cout << "netH: " << endl;
			//netH->print();
			//		cout << endl;


#pragma omp parallel for num_threads(4) 
			for (int i = 0; i < netH->dim()[0]; i++){                          //change netH to outH, outBias is out matrix plus bias
				outH->n2Arr[i][0] = sigmoid(netH->n2Arr[i][0]);
				outHBias->n2Arr[i][0] = sigmoid(netH->n2Arr[i][0]);
			}


			//		cout << "outH: " << endl;
			//outH->print();
			//		cout << endl;


			foo = outHBias->dim()[0] - 1;

			outHBias->n2Arr[foo][0] = 1;                                       //complete the outBias

			//		cout << "outHBias: " << endl;
			//outHBias->print();
			//		cout << endl;

			output = matMatMul(*hideToOutputWeight, *outHBias);


#pragma omp parallel for num_threads(4) 
			for (int i = 0; i < output->dim()[0]; i++){
				output->n2Arr[i][0] = sigmoid(output->n2Arr[i][0]);        //finish forward propagation
			}

//			cout << "output: " << endl;
//			output->print();
//			cout << endl;


#pragma omp parallel for num_threads(4) 
			for (int i = 0; i < hideToOutputWeight->dim()[0]; i++){
				for (int j = 0; j < hideToOutputWeight->dim()[1] - 1; j++){
					tmpWeight->n2Arr[i][j] = hideToOutputWeight->n2Arr[i][j];
				}

			}



//					cout << "tmpWeight: "<< endl;
//					tmpWeight->print();
//					cout << endl;



			tmp = matSub(*output, *target);                           // Etotal / output = out - target

//					cout << "tmp: "<< endl;
//					tmp->print();
//					cout << endl;

#pragma omp parallel for num_threads(4) 
			for (int i = 0; i < output->dim()[0]; i++){
				float outTmp = output->n2Arr[i][0];
				partError->n2Arr[i][0] = fai*outTmp*(1 - outTmp)*(tmp->n2Arr[i][0]);       // out / net = out(1-out)

			}																				//complete the partError with fai

//					cout << "partError: "<< endl;
//					partError->print();
//					cout << endl;

			tmpTrans = partError->transpose();                            // backpropagation of weight between input layer and hidden layer               

			tmp = matMatMul(*tmpTrans, *tmpWeight);

//			cout << "tmp: " << endl;
//			tmp->print();
//			cout << endl;


#pragma omp parallel for num_threads(4) 
			for (int i = 0; i < outH->dim()[0]; i++){
				float outTmp = outH->n2Arr[i][0];
				tmp->n2Arr[0][i] = tmp->n2Arr[0][i] * outTmp*(1 - outTmp);
			}

			tmp = tmp->transpose();

			tmpTrans = input->transpose();

			tmp = matMatMul(*tmp, *tmpTrans);

			inputToHideWeight = matAdd(*inputToHideWeight, *tmp);


			tmpTrans = outHBias->transpose();                                                 // backpropagation of weight between hidden layer and output layer
			tmp = matMatMul(*partError, *tmpTrans);

//			cout << "size of tmp :" << tmp->dim()[0] << "*" << tmp->dim()[1] << endl;

			hideToOutputWeight = matAdd(*hideToOutputWeight, *tmp);


			//		cout << "new hidden weight" <<endl;
			//		hideToOutputWeight->print();

			//		cout << endl;

			//		cout << "output: " << output << endl;
			//		cout << endl;
			cout << turn <<endl;
		}


	}


	delete tmp;
	delete tmpTrans;
	delete tmpWeight;
	delete outHBias;

}
float ANN::predict(vector<float> in){

	MyMatrix<float> *outHBias = new MyMatrix<float>(11, 1);

	for (int i = 0; i < in.size(); i++){			// input data into input matrix
		input->n2Arr[i][0] = in[i];
	}

	int foo = input->dim()[0] - 1;
	input->n2Arr[foo][0] = 1;                            //set bias


	//		cout << "input: " << endl;
	//		input->print();
	//		cout << endl;

	netH = matMatMul(*inputToHideWeight, *input);			//forward propagation

	//		cout << "netH: " << endl;
	//netH->print();
	//		cout << endl;


	for (int i = 0; i < netH->dim()[0]; i++){                          //change netH to outH, outBias is out matrix plus bias
		outH->n2Arr[i][0] = sigmoid(netH->n2Arr[i][0]);
		outHBias->n2Arr[i][0] = sigmoid(netH->n2Arr[i][0]);
	}


	//		cout << "outH: " << endl;
	//outH->print();
	//		cout << endl;


	foo = outHBias->dim()[0] - 1;

	outHBias->n2Arr[foo][0] = 1;                                       //complete the outBias

	//		cout << "outHBias: " << endl;
	//outHBias->print();
	//		cout << endl;

	output = matMatMul(*hideToOutputWeight, *outHBias);

	//		cout << "output: " << endl;
	//tmp->print();
	//		cout << endl;

	for (int i = 0; i < output->dim()[0]; i++){
		output->n2Arr[i][0] = sigmoid(output->n2Arr[i][0]);        //finish forward propagation
	}


	float max = output->n2Arr[0][0];                                       //change the output matrix into the real result
	for (int i = 1; i < output->dim()[0]; i++){
		if (output->n2Arr[i][0]>max)
			max = output->n2Arr[i][0];
	}

	delete outHBias;
	return max;

	//	return output;
}

void ANN::storeWeight(){
	hideToOutputWeight->out();
	inputToHideWeight->out();
}
void ANN::loadWeight(){

	vector<MyMatrix<float>*> load;
	MyMatrix<float>::in(load);


	hideToOutputWeight = load[0];
	inputToHideWeight = load[1];

}

