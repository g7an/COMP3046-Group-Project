#include<iostream>
#include"MyMatrix.h";
#include"MyMatrix.cpp";
using namespace std;


template<class T >
MyMatrix<T>* matSub(MyMatrix<T> &x, MyMatrix<T>&y);


template<class T >
MyMatrix<T>* matAdd(MyMatrix<T> &x, MyMatrix<T>&y);

template<class T >
MyMatrix<T>* matMatMul(MyMatrix<T> &x, MyMatrix<T>&y);

/*
int main(){
	MyMatrix<float> *matrix1 = new MyMatrix<float>(3, 4);
	MyMatrix<float> *matrix2 = new MyMatrix<float>(3, 4);

	for (int i = 0; i < 3; i++){
		for (int j = 0; j < 4; j++){
			matrix1->n2Arr[i][j] = i * 3 + j;
		}
	}

	for (int i = 0; i < 3; i++){
		for (int j = 0; j < 4; j++){
			matrix2->n2Arr[i][j] = 2* (i * 3 + j);
		}
	}

	matrix1->print();
	cout << endl;
	matrix2->print();
	MyMatrix<float> *tmp = NULL;
	cout << endl;

	tmp = matSub(*matrix2, *matrix1);
	tmp->print();

	tmp = matAdd(*matrix2, *matrix1);
	tmp->print();

	tmp = matrix1->transpose();
	tmp->print();

	tmp = matMatMul(*tmp, *matrix1);
	tmp->print();




	return 0;

}*/