#include "MyVector.h"
#include<math.h>
#include<string>
#include<fstream>
#include <vector>
#include<iostream>
using namespace std;

template<class T>
MyVector<T>::MyVector(int len)
{
	arr = new T[len];
	this->len = len;
}


template<class T>
int MyVector<T>::size(){
	return len;
}


template<class T>
double MyVector<T>::mean(){
	T tmp = 0;
	for (int i = 0; i < len; i++){
		tmp += arr[i];
	}
	return (tmp / (double)len);
}


template<class T>
void MyVector<T>::print(){
	for (int i = 0; i < len; i++)
	{
		cout << arr[i] << " ";
	}
	cout << endl;
}

template<class T>
MyVector<T>::~MyVector()
{
	delete [] arr;
}


template<class T>
T MyVector<T>::L1(){
	T tmp = 0;
	for (int i = 0; i < len; i++)
	{
		tmp += abs(arr[i]);
	}
	return tmp;
}

template<class T>
double MyVector<T>::euc(){
	T sum = 0;
	for (int i = 0; i < len; i++)
	{
		sum += pow(arr[i], 2);
	}
	return sqrt(sum);

}

template<class T>
void MyVector<T>::smul(const T &t){
	for (int i = 0; i < len; i++)
	{
		arr[i] *= t;
	}
}

template<class T>
void MyVector<T>::out(){
	ofstream out;
	out.open("vector_data.txt", ios::out | ios::app);

	out << endl;

	out << len << " ";
	for (int i = 0; i <len; i++){
		out << arr[i] << " ";
	}
	out.close();
}


template<class T>
void MyVector<T>::in(vector<MyVector<T>*> &result){
	int num;
	string input1;
	string input2;
	T data;
	MyVector<T> *tmp = NULL;

	ifstream in;
	in.open("./input_vector_data.txt", ios::in);
	while (in >> input1)
	{
		num = atoi(input1.c_str());
		tmp = new MyVector<T>(num);

		for (int i = 0; i < num; i++)
		{
			in >> input2;
			data = atoi(input2.c_str());
			tmp->arr[i] = data;
		}
		result.push_back(tmp);
	}
	in.close();
}

