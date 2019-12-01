#pragma once
#include "MyMatrix.h"
#include <vector>
class plusANN 
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
	plusANN(float, int, int,int*,int,int,float);
	~plusANN();

	float sigmoid(float);
	float Relu(float);
	float random();
	void setLR(float);
	void setEpochs(int);
	float normalDis();
	//void initializeWeight();

	void train(std::vector<std::vector<float> >, std::vector<float>);
	void trainPlus(std::vector<std::vector<float> >, std::vector<float>);
	float predict(std::vector<float>);
	MyMatrix<float>* forward(std::vector<float>);

	float totalLoss(std::vector<float>, float);
	float lossWithCrossE(std::vector<float>, float);
	void batchLoss(std::vector<std::vector<float> >, std::vector<float>);

	void storeWeight();
	void loadWeight();




};
