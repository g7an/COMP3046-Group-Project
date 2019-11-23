#pragma once
#pragma once
#include<vector>
template<class T>
class MyMatrix
{
	int* sz ;

public:
	T **n2Arr;
	MyMatrix(int, int);
	~MyMatrix();
	int* dim();
	void smul(const T &t);
	void out();
	void print();
	MyMatrix<T>* transpose();
	static void in(std::vector<MyMatrix<T>*> &);

};
