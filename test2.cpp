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
//#include <random>
#include <omp.h>
using namespace std;

//#include "globalFunctions.cpp"
using namespace std;


///*
int main()                                                
{
	int layerSize [] ={28*28,50,10};
	MyANN annModel(0.01, 100, 64,layerSize,sizeof(layerSize)/sizeof(int),140,1);                                                  
	vector< vector<float> > X_train;
	vector<float> y_train;

	ifstream myfile("data/train.txt");
	//ifstream myfile("data/train.txt");

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
				//X.push_back(x);
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
	/*
	   for(int i=0;i<y_train.size();i++){
	   cout<<y_train[i]<<" ";
	   }
	 */

	annModel.train(X_train, y_train);
	//annModel.loadWeight();                             // You can remove the // in this line to load an aready trained weight matrix 
	//annModel.storeWeight();                            // You can remove the // in this line to output the trained weight matrix to a local file 

	//annModel.storeWeight();                            // You can remove the // in this line to output the trained weight matrix to a local file 
	cout<< annModel.totalLoss(X_train[10],y_train[10])<<endl;
	//cout << "predict: " << annModel.predict(X_train[10]) << " real value: " << y_train[10] <<endl;;
	vector< vector<float> > X_test;
	vector<float> y_test;

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

			//#pragma omp parallel for num_threads(4)
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

	cout << "predict: " << annModel.predict(X_test[4]) << " real value: " << y_test[4] <<endl;
	cout << "predict: " << annModel.predict(X_test[90]) << " real value: " << y_test[90] <<endl;
	cout << "predict: " << annModel.predict(X_test[2]) << " real value: " << y_test[2] <<endl;
	cout << "predict: " << annModel.predict(X_test[8]) << " real value: " << y_test[8] <<endl;
	cout << "predict: " << annModel.predict(X_test[16]) << " real value: " << y_test[16] <<endl;
	cout << "predict: " << annModel.predict(X_test[32]) << " real value: " << y_test[32] <<endl;

	annModel.storeWeight();
	cout << "store finish" << endl;


	cout << endl;


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


