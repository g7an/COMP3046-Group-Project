#pragma once
#include "MyMatrix.h"
#include <vector>
class ANN
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
	ANN(float, int, int, int, float);
	~ANN();

	float sigmoid(float);
	float random();
	void setLR(float);
	void setEpochs(int);
	//void initializeWeight();

	void train(std::vector<std::vector<float> >, std::vector<float>);
	float predict(std::vector<float>);
	float totalLoss(std::vector<float>, float);

	void storeWeight();
	void loadWeight();




};