#include<iostream>
#include"Pure.h"
#include<fstream>
#include <sstream>

using namespace std;
int main(void){

	int layerSize [] ={28*28,50,10};
	Pure annModel(0.01, 100, 64,layerSize,sizeof(layerSize)/sizeof(int));                                                  
	vector< vector<float> > X_train;
	vector<float> y_train;

	bool train = true;

	if(train){
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

					y_train.push_back(y);


				for (int i = 0; i < 28 * 28; i++) {
					ss >> x;
					X.push_back(x / 255.0);
					//X.push_back(x);
				}

					X_train.push_back(X);
			}

			myfile.close();
			cout << "Loading data finished.\n";
		}
		else
			cout << "Unable to open file" << '\n';


		//	annModel.loadWeight();        //remove this line!!!!!!!
		cout<<"train begin"<<endl;

		annModel.train(X_train, y_train);

		cout<<"train finish"<<endl;
	}

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

	int cnt = 0;
	int pre = 0;
	for(int i=0;i<X_test.size();i++){
		pre = annModel.predict(X_test[i]);
		if(pre==y_test[i]) cnt++;
	}
	cout<<"predict accuracy: "<<(float)cnt/X_test.size()<<endl;
}
