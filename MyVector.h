#pragma once
#include<queue>
template <class T>
class MyVector
{
	int len;
public:
	T *arr;

	MyVector(int len);
	int size();
	double mean();
	T L1();
	double euc();
	void smul(const T &);
	void print();
	void out();
	static void in(std::vector<MyVector<T>*>&);
	~MyVector();
};

