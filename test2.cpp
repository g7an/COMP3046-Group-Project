#include"ANN.h"

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
#include <random>
#include <omp.h>
using namespace std;


using namespace std;


///*
int main()                                                
{
	ANN annModel(0.5, 50);
                                                        
	vector< vector<float> > X_train;
	vector<float> y_train;
	int cnt = 0;

	ifstream myfile("train.txt");

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
			y_train.push_back(y);

//#pragma omp parallel for num_threads(4)
			for (int i = 0; i < 28 * 28; i++) {
				ss >> x;
				X.push_back(x / 255.0);
			}
			X_train.push_back(X);
		}

		myfile.close();
		cout << "Loading data finished.\n";
	}
	else
		cout << "Unable to open file" << '\n';




	annModel.train(X_train, y_train);

//	annModel.loadWeight();                             // You can remove the // in this line to load an aready trained weight matrix 
//	annModel.storeWeight();                            // You can remove the // in this line to output the trained weight matrix to a local file 
	annModel.storeWeight();                            // You can remove the // in this line to output the trained weight matrix to a local file 
	cout << "store finish" << endl;


	cout << endl;

												       //Remind: I'm sorry that I use the sigmoind funcion rather than sgn, so my false value is 0 rather than -1. 


	return 0;
}
//*/

/*
	float result = annModel.predict(X[49]);           //Thus, I change my final result from 0 to -1 and this is why there is a -1 after the result variable. 
	cout << "target value is -1: " << result-1 << endl;

	result = annModel.predict(X[48]);
	cout << "target value is -1: " << result-1 << endl;

	result = annModel.predict(X[40]);
	cout << "target value is -1: " << result-1 << endl;
	result = annModel.predict(X[43]);
	cout << "target value is -1: " << result-1 << endl;

	cout << endl;

	float result2 = annModel.predict(X[99]);
	cout << "target value is 1: "<< result2 << endl;

	result2 = annModel.predict(X[98]);
	cout << "target value is 1: "<< result2 << endl;


	result2 = annModel.predict(X[83]);
	cout << "target value is 1: " << result2 << endl;

	result2 = annModel.predict(X[86]);
	cout << "target value is 1: " << result2 << endl;
	annModel.storeWeight();
*/


