#include"MyANN.h"
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
#include <math.h> 
#include <sstream>
#include <omp.h>
using namespace std;
using namespace std;

int main()                                                
{
	int layerSize [] ={28*28,50,10};                                                  
	vector< vector<float> > X_train;
	vector<float> y_train;
	vector< vector<float> > X_test;
	vector<float> y_test;
	bool train = true;

	if(train){
		ifstream myfile("data/train.txt");
		// ifstream myfile("data/train_small.txt");

		if (myfile.is_open())
		{
			cout << "Loading data ...\n";
			string line;
			while (getline(myfile, line))
			{
				int x, y;
				vector<float> X;
				stringstream ss(line);
				ss >> y;

				if(y!=3){
					y_train.push_back(y);
				}
				for (int i = 0; i < 28 * 28; i++) {
					ss >> x;
					X.push_back(x / 255.0);
				}

				if(y!=3){
					X_train.push_back(X);
				}
			}

			myfile.close();
			cout << "Loading data finished.\n";
		}
		else
			cout << "Unable to open file" << '\n';
	}

	ifstream myfile2("data/test.txt");

	if (myfile2.is_open())
	{
		cout << "Loading data ...\n";
		string test_line;
		while (getline(myfile2, test_line))
		{
			int test_x, test_y;
			vector<float> test_X;
			stringstream ss(test_line);
			ss >> test_y;
			y_test.push_back(test_y);
			for (int i = 0; i < 28 * 28; i++) {
				ss >> test_x;
				test_X.push_back(test_x / 255.0);
			}
			X_test.push_back(test_X);
		}
		myfile2.close();
		cout << "Loading data finished.\n";
	}
	else
		cout << "Unable to open file" << '\n';

	MyANN annModel(0.01, 300, 64, layerSize, sizeof(layerSize)/sizeof(int),140,1);

	if(train){
		annModel.train(X_train, y_train, X_test, y_test);	
		annModel.storeWeight();
		cout << "store finish" << endl;
	}
	if(!train){
		annModel.loadWeight();
		cout << "Model outcome: " << endl;
		annModel.getAcc(X_test, y_test);
		cout << "predict: " << annModel.predict(X_test[4]) << " real value: " << y_test[4] <<endl;
		cout << "predict: " << annModel.predict(X_test[90]) << " real value: " << y_test[90] <<endl;
		cout << "predict: " << annModel.predict(X_test[2]) << " real value: " << y_test[2] <<endl;
		cout << "predict: " << annModel.predict(X_test[8]) << " real value: " << y_test[8] <<endl;
		cout << "predict: " << annModel.predict(X_test[16]) << " real value: " << y_test[16] <<endl;
		cout << "predict: " << annModel.predict(X_test[32]) << " real value: " << y_test[32] <<endl;
	}
	cout<<"Finish"<<endl;
	return 0;
}


