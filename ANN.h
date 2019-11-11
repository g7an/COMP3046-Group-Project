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
	int batch_size;
	MyVector<MyMatrix<float> *>*hidWeight;
	MyVector<MyMatrix<float> *>* delta_w;
	MyVector<MyMatrix<float> *>* tmpWeight;

	MyMatrix<float> *input;

	MyVector<MyMatrix<float> *>* netH;
	MyVector<MyMatrix<float> *>* outH;
	MyVector<MyMatrix<float> *>* outHBias;

	MyMatrix<float> * partError;
	MyMatrix<float> *target;




public:
	ANN(float, int,int);
	~ANN();
	float sigmoid(float);
	float random();
	void setLR(float);
	void setEpochs(int);

	void initializeWeight();
	void train(std::vector<std::vector<float> >, std::vector<float>);
	float predict(std::vector<float>);
	float totalLoss(std::vector<float>, float);

	void storeWeight();
	void loadWeight();

	void generateTarget(float t);



};

