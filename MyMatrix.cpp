#include "MyMatrix.h"
#include<fstream>
#include<iostream>
#include<omp.h>
using namespace std;


template<class T>
MyMatrix<T>::MyMatrix(int h, int w)
{
	sz[0] = h;
	sz[1] = w;
	n2Arr = new T*[h];
	for (int i = 0; i < h; i++){
		n2Arr[i] = new T[w];
	}
}


template<class T>
MyMatrix<T>::~MyMatrix()
{
	for (int i = 0; i < sz[0]; i++){
		delete[] n2Arr[i];
	}
	delete[] sz;
	delete[] n2Arr;
}

template<class T>
void MyMatrix<T>::print(){
	for (int i = 0; i < sz[0]; i++)
	{
		for (int j = 0; j < sz[1]; j++){
			if (n2Arr[i][j] < 10){
				cout << " ";

			}
			cout << n2Arr[i][j] << " ";
		}
		cout << endl;
	}
	cout << endl;


}

template<class T>
int* MyMatrix<T>::dim(){
	return sz;
}

template<class T>
void MyMatrix<T>::smul(const T& t){
	for (int i = 0; i < sz[0]; i++){
		for (int j = 0; j < sz[1]; j++){
			n2Arr[i][j] *= t;
		}
	}

}

template<class T>
void MyMatrix<T>::out(){
	ofstream out;
	out.open("matrix_data.txt", ios::out | ios::app);
	out << endl;
	out << sz[0] << " " << sz[1] << endl;

	for (int i = 0; i < sz[0]; i++){
		for (int j = 0; j < sz[1]; j++){
			out << n2Arr[i][j] << " ";
		}
		out << endl;
	}
	out.close();

}

template<class T>
MyMatrix<T>* MyMatrix<T>::transpose(){
	/*
	T **tmp = new T*[sz[1]];
	for (int i = 0; i < sz[1]; i++){
		tmp[i] = new T[sz[0]];
	}*/
	MyMatrix<T>* tmp = new MyMatrix<T>(sz[1], sz[0]);

#pragma omp parallel for num_threads(4) 
	for (int i = 0; i < sz[0]; i++){
		for (int j = 0; j < sz[1]; j++){
			tmp->n2Arr[j][i] = n2Arr[i][j];
		}
	}
	return tmp;
/*
	int foo = sz[0];
	sz[0] = sz[1];
	sz[1] = foo;
*/

}

template<class T>
void MyMatrix<T>::in(vector<MyMatrix<T>*> &result){
	int h;
	int w;
	string input1;
	string input2;
	double input3;
	T data;
	MyMatrix<T> *tmp = NULL;

	ifstream in;
	in.open("./input_matrix_data.txt", ios::in);
	while (in >> input1 >> input2)
	{
		h = atoi(input1.c_str());
		w = atoi(input2.c_str());

		tmp = new MyMatrix<T>(h, w);

		for (int i = 0; i < h*w; i++)
		{
			in >> input3;
			tmp->n2Arr[i / w][i%w] = input3;
		}
		result.push_back(tmp);
	}
	in.close();
}
