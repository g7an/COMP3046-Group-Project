﻿#include"MyVector.h"
//#include"MyVector.cpp"
#include "MyMatrix.h"
//#include "MyMatrix.cpp"

#include<iostream>
#include<fstream>
#include<iostream>
#include<string>
#include<queue>
#include<cstdio>
#include<algorithm>
#include<cmath>
#include<stdlib.h>
#include<time.h>
using namespace std;

template<class T, class N>
MyVector<T>* vecAdd(MyVector<T> &x, MyVector<T> &y, N a);


template<class T>
T vecDot(MyVector<T> &x, MyVector<T> &y);


template<class T >
MyMatrix<T>* matAdd(MyMatrix<T> &x, MyMatrix<T>&y);


template<class T >
MyMatrix<T>* matVecMul(MyMatrix<T> &x, MyVector<T>&y);

template<class T >
MyMatrix<T>* vecMatMul(MyVector<T>&x, MyMatrix<T> &y);

template<class T >
MyMatrix<T>* matMatMul(MyMatrix<T> &x, MyMatrix<T>&y);

template<class T >
MyMatrix<T>* matSub(MyMatrix<T> &x, MyMatrix<T>&y);


template<class T, class N>
MyVector<T>* vecAdd(MyVector<T> &x, MyVector<T> &y, N a){
	if (x.size() != y.size()){
		cout << "This vector addition is illegal";
		return NULL;
	}
	MyVector<T>* ptr = new MyVector<T>(x.size());
	for (int i = 0; i < x.size(); i++){
		ptr->arr[i] = a* x.arr[i] + y.arr[i];
	}
	return ptr;
}

template<class T>
T vecDot(MyVector<T> &x, MyVector<T> &y){
	if (x.size() != y.size()){
		cout << "This vector dot multiplication is illegal";
		return NULL;
	}
	T sum = 0;
	for (int i = 0; i < x.size(); i++){
		sum += x.arr[i] * y.arr[i];
	}
	return sum;

}

template<class T >
MyMatrix<T>* matAdd(MyMatrix<T> &x, MyMatrix<T>&y){
	if (x.dim()[0] != y.dim()[0] || x.dim()[1] != y.dim()[1]){
		cout << "This matrix addition is illegal";
		return NULL;
	}
	MyMatrix<T>* tmp = new MyMatrix<T>(x.dim()[0], x.dim()[1]);

	for (int i = 0; i < x.dim()[0]; i++){
		for (int j = 0; j < x.dim()[1]; j++){
			tmp->n2Arr[i][j] = x.n2Arr[i][j] + y.n2Arr[i][j];
		}
	}
	return tmp;
}

template<class T >
MyMatrix<T>* matSub(MyMatrix<T> &x, MyMatrix<T>&y){
	if (x.dim()[0] != y.dim()[0] || x.dim()[1] != y.dim()[1]){
		cout << "This matrix addition is illegal";
		return NULL;
	}
	MyMatrix<T>* tmp = new MyMatrix<T>(x.dim()[0], x.dim()[1]);

	for (int i = 0; i < x.dim()[0]; i++){
		for (int j = 0; j < x.dim()[1]; j++){
			tmp->n2Arr[i][j] = x.n2Arr[i][j] - y.n2Arr[i][j];
		}
	}
	return tmp;

}

template<class T >
MyMatrix<T>* matVecMul(MyMatrix<T> &x, MyVector<T>&y){
	if (x.dim()[1] != y.size()){
		cout << "This vector matrix multiplication is illegal";
		return NULL;
	}
	MyMatrix<T>* tmp = new MyMatrix<T>(x.dim()[0], 1);
	T inter = 0;

	for (int i = 0; i < x.dim()[0]; i++){
		for (int j = 0; j < y.size(); j++){
			inter += x.n2Arr[i][j] * y.arr[j];
		}
		tmp->n2Arr[i][0] = inter;
		inter = 0;
	}
	return tmp;

}


template<class T >
MyMatrix<T>* vecMatMul(MyVector<T>&y, MyMatrix<T>&x){
	if (x.dim()[0] != y.size()){
		cout << "This vector matrix multiplication is illegal";
		return NULL;
	}
	MyMatrix<T>* tmp = new MyMatrix<T>(1, x.dim()[1]);
	T inter = 0;

	for (int i = 0; i < x.dim()[1]; i++){
		for (int j = 0; j < y.size(); j++){
			inter += x.n2Arr[j][i] * y.arr[j];
		}
		tmp->n2Arr[0][i] = inter;
		inter = 0;
	}
	return tmp;

}
template<class T >
MyMatrix<T>* matMatMul(MyMatrix<T> &x, MyMatrix<T>&y){
	if (x.dim()[1] != y.dim()[0]){
		cout << "This matrix multiplication is illegal";
		return NULL;
	}

	MyMatrix<T>* tmp = new MyMatrix<T>(x.dim()[0], y.dim()[1]);

	for (int i = 0; i < x.dim()[0]; i++){
		for (int j = 0; j < y.dim()[1]; j++){
			tmp->n2Arr[i][j] = 0;
		}
	}
	T foo = 0;

	for (int i = 0; i < x.dim()[0]; i++){
		for (int j = 0; j < y.dim()[1]; j++){
			for (int r = 0; r < x.dim()[1]; r++){

				tmp->n2Arr[i][j] += x.n2Arr[i][r] * y.n2Arr[r][j]; 
			}
		}
	}

	return tmp;
}

template<class T >
MyMatrix<T>* eleMul(MyMatrix<T> &x, MyMatrix<T>&y){
	if (x.dim()[0] != y.dim()[0] || x.dim()[1] != y.dim()[1]){
		cout << "Matrix element-wise mult is invalid!" << endl;
		cout<<"x: "<<x.dim()[0]<<" : "<<x.dim()[1]<<endl;
		cout<<"y: "<<y.dim()[0]<<" : "<<y.dim()[1]<<endl;
		return NULL;
	}
	MyMatrix<float>* tmp = new MyMatrix<float>(x.dim()[0], x.dim()[1]);
	for (int i = 0; i < x.dim()[0]; i++){
		for (int j = 0; j < x.dim()[1]; j++){
			tmp->n2Arr[i][j] = x.n2Arr[i][j] * y.n2Arr[i][j];
		}
	}
	return tmp;


}

template<class T >
MyMatrix<T>* d_sigmoid(MyMatrix<T> &x){

	MyMatrix<float>* tmp = new MyMatrix<float>(x.dim()[0], x.dim()[1]);
	for (int i = 0; i < x.dim()[0]; i++){
		for (int j = 0; j < x.dim()[1]; j++){
			// cout << i << ", " << j << endl;
			tmp->n2Arr[i][j] = x.n2Arr[i][j] * (1 - x.n2Arr[i][j]);
		}
	}
	return tmp;
}

template<class T >
MyMatrix<T>* d_Relu(MyMatrix<T> &x){

	MyMatrix<float>* tmp = new MyMatrix<float>(x.dim()[0], x.dim()[1]);
	for (int i = 0; i < x.dim()[0]; i++){
		for (int j = 0; j < x.dim()[1]; j++){
			// cout << i << ", " << j << endl;
			tmp->n2Arr[i][j] = x.n2Arr[i][j] > 0 ? 1 : 0;
		}
	}
	return tmp;
}
template<class T >
void updateDelta_bias(MyMatrix<T> &x, float* bias){
	for(int i=0;i<x.dim()[0];i++){
		bias[i] += x.n2Arr[i][0];
	}

}
template<class T >
MyMatrix<T>* d_CrossEntropy(MyMatrix<T> &t,MyMatrix<T> &x){
	if (x.dim()[0] != t.dim()[0] || x.dim()[1] != t.dim()[1]){
		cout << "They have different size!" << endl;
		cout<<"x: "<<x.dim()[0]<<" : "<<x.dim()[1]<<endl;
		cout<<"y: "<<t.dim()[0]<<" : "<<t.dim()[1]<<endl;
		return NULL;
	}
	float target;
	float predict;

	MyMatrix<float>* tmp = new MyMatrix<float>(x.dim()[0], x.dim()[1]);
	for (int i = 0; i < x.dim()[0]; i++){
		for (int j = 0; j < x.dim()[1]; j++){
			target = t.n2Arr[i][j];
			predict = x.n2Arr[i][j];

			predict = predict<1e-10 ? 1e-10 : predict;
			predict = (1-predict)<1e-10 ? (1-1e-10) : predict;
		
			tmp->n2Arr[i][j] = -target/predict + (1-target)/(1-predict) ;
		}
	}
	return tmp;

}
