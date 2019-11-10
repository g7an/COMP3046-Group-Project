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
	MyVector<MyMatrix<float> *>*hidWeight;
	MyVector<MyMatrix<float> *>* delta_w;
	MyVector<MyMatrix<float> *>* tmpWeight;

	MyMatrix<float> *input;

	MyVector<MyMatrix<float> *>* netH;
	MyVector<MyMatrix<float> *>* outH;
	MyVector<MyMatrix<float> *>* outHBias;

	MyMatrix<float> * loss;
	MyMatrix<float> *target;




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
	MyMatrix<float>* eleMul(MyMatrix<float>* x, MyMatrix<float>* y);
	MyMatrix<float>* d_sigmoid(MyMatrix<float>* x);

	void generateTarget(float t);



};

