#pragma once
#include "MyVector.h"
#include "MyMatrix.h"
#include <vector>
class ANN
{
	float lr;								//learning rate
	int epochs;
	int num_hidLayer;
	int* num_neurons;
	MyVector<MyMatrix<float> *> *hidWeight;
	//MyVector<void *> *hidWeight;
	MyMatrix<float> *inputToHideWeight;
	MyMatrix<float> *input;
	// MyMatrix<float> *netH;
	// MyMatrix<float> *outH;
	MyVector<MyMatrix<float> *>* netH;
	MyVector<MyMatrix<float> *>* outH;
	MyVector<MyMatrix<float> *>* outHBias;
	MyMatrix<float> *hideToOutputWeight;
	MyMatrix<float> *output;
	MyMatrix<float> *target;
	MyMatrix<float> *partError;



public:
	ANN(float, int);
	~ANN();
	float sigmoid(float);
	float random();
	void setLR(float);
	void setEpochs(int);

	void initializeWeight();
	void train(std::vector<std::vector<float> >, std::vector<float>);
	float predict(std::vector<float>);
	void storeWeight();
	void loadWeight();

	void generateTarget(float t);



};

