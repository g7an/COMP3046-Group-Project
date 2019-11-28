#include<iostream>
#include"MyMatrix.h"
#include"MyMatrix.cpp"
#include"globalFunctions.cpp"

using namespace std;


template<class T >
MyMatrix<T>* matSub(MyMatrix<T> &x, MyMatrix<T>&y);


template<class T >
MyMatrix<T>* matAdd(MyMatrix<T> &x, MyMatrix<T>&y);

template<class T >
MyMatrix<T>* matMatMul(MyMatrix<T> &x, MyMatrix<T>&y);

template<class T >
void updateDelta_bias(MyMatrix<T> &x, float* bias);

template<class T >
MyMatrix<T>* d_sigmoid(MyMatrix<T> &x);

template<class T >
MyMatrix<T>* eleMul(MyMatrix<T> &x, MyMatrix<T>&y);

template<class T >
MyMatrix<T>* d_CrossEntropy(MyMatrix<T> &t,MyMatrix<T> &x);

//matrix: transpose, mul, sub,smul;

int main(){
	MyMatrix<float> *matrix1 = new MyMatrix<float>(10, 1);
	MyMatrix<float> *matrix2 = new MyMatrix<float>(10, 10);
	float *bias = new float[10];

	for (int i = 0; i < 10; i++){
		bias[i] = 0;
		for (int j = 0; j < 1; j++){
			matrix1->n2Arr[i][j] = i * 3 + 1.1;
		}
	}

	for (int i = 0; i < 10; i++){
		for (int j = 0; j < 10; j++){
			matrix2->n2Arr[i][j] = i * 4 + j*1.2;
		}
	}

	cout<<"matrix"<<endl;
	matrix1->print();
	cout << endl;

	cout<<"matrix2"<<endl;
	matrix2->print();
	MyMatrix<float> *tmp = NULL;
	cout << endl;


	cout<<"matrix1 transpose"<<endl;
	tmp = matrix1->transpose();
	tmp->print();

	cout<<"mat2 mul Mat1"<<endl;
	tmp = matMatMul(*matrix2, *matrix1);
	tmp->print();

	cout<<"d_CrossEntropy(tmp,matrix1)"<<endl;
	tmp = d_CrossEntropy(*tmp,*matrix1);
	tmp->print();

	cout<<"tmp sub matrix1"<<endl;
	tmp = matSub(*tmp, *matrix1);
	tmp->print();

	cout<<"matrix1 mul:3"<<endl;
	matrix1->smul(3);
	matrix1->print();

	cout<<"updateDelta_bias with matrix1"<<endl;
	updateDelta_bias(*matrix1, bias);

	for(int i=0;i<10;i++){
		cout<<bias[i]<<" ";
	}
	cout<<endl;

	for (int i = 0; i < 10; i++){
		bias[i] = 0;
		for (int j = 0; j < 1; j++){
			matrix1->n2Arr[i][j] = 0.5;
		}
	}
	cout<<"new matrix1"<<endl;
	matrix1->print();

	cout<<"d_sigmoid(matrix1)"<<endl;
	tmp = d_sigmoid(*matrix1);
	tmp->print();

	cout<<"tmp.*matrix1"<<endl;
	tmp = eleMul(*tmp,*matrix1);
	tmp->print();





	delete [] bias;
	return 0;

}
