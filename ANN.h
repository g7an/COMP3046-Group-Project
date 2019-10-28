#pragma once
#include "MyMatrix.h"
#include <vector>
class ANN
{
	float fai;								//learning rate
	int epochs;

	MyMatrix<float> *inputToHideWeight;
	MyMatrix<float> *input;
	MyMatrix<float> *netH;
	MyMatrix<float> *outH;
	MyMatrix<float> *hideToOutputWeight;
	MyMatrix<float> *output;
	MyMatrix<float> *target;
	MyMatrix<float> *partError;



public:
	ANN(float, int);
	~ANN();
	float sigmoid(float);
	float random();
	void setFai(float);
	void setEpochs(int);

	void initializeWeight();
	void train(std::vector<std::vector<float>>, std::vector<float>);
	float predict(std::vector<float>);
	void storeWeight();
	void loadWeight();

	void generateTarget(float t);



};

