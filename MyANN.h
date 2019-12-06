#pragma once
#include "MyMatrix.h"
#include <vector>
class MyANN
{
	float lr;								//learning rate
	int epochs;
	int decayEpoch;
	float decay;
	int num_hidLayer;
	int* total_neurons;
	int batch_size;

	std::vector<MyMatrix<float> *> hidWeight;
	std::vector<MyMatrix<float> *> delta_w;


	MyMatrix<float> *input;
	MyMatrix<float> *partError;
	MyMatrix<float> *target;

	MyMatrix<float> * netH;

	std::vector<MyMatrix<float> *> outH;

    std::vector<float *> bias;
    std::vector<float *> delta_bias;




public:

	MyANN(float, int, int,int*,int,int,float);
	float sigmoid(float);
	float Relu(float);
	float random();
	void setLR(float);
	void setEpochs(int);
	float normalDis();
	void train(std::vector<std::vector<float> >, std::vector<float>, std::vector<std::vector<float> >, std::vector<float>);
	void getAcc(std::vector<std::vector<float> >, std::vector<float>);
	int predict(std::vector<float>);
	float showAcc(std::vector<float> , std::vector<float>);
	MyMatrix<float>* forward(std::vector<float>);

	float totalLoss(std::vector<float>, float);
	float lossWithCrossE(std::vector<float>, float);
	void getLoss(std::vector<std::vector<float> >, std::vector<float>);

	void storeWeight();
	void loadWeight();
	void storeBias();
	void loadBias();



};
