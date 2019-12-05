#include "Pure.h"
#include<iostream>
using namespace std;

void print(float* x, int x_rows, int x_columns ){
	for(int i=0;i<x_rows;i++){
		for(int j=0;j<x_columns;j++){

			cout<<x[i*x_columns+ j]<<" ";
		}
		cout<<endl;
	}
}

int main(void){
	int wat [] = {10,10,10};
	Pure *pureM = new Pure(10.0f,10,10,wat,10);

	float* matrix1 = new float[30]; // 5*6
	float* matrix2 = new float[30]; // 6*5
	float* bias= new float[6]; // 1*6
	float* result = NULL;
	float* result2 = NULL;

	for(int i=0;i<30;i++){
		matrix1[i] = i * 1.1f;
		matrix2[i] = i * 1.3f;
	}

	for(int i=0;i<6;i++){
		bias[i] = i*1.0f;
	}

	cout<<"matrix1"<<endl;
	print(matrix1,5,6);

	cout<<"matrix2"<<endl;
	print(matrix2,6,5);

	cout<<"store"<<endl;
	pureM->out(matrix1,5,6);
	pureM->out(matrix2,6,5);
	cout<<"store finish"<<endl;

	/*
	result = new float[5*6];
	pureM->netToOut(result,matrix1,bias,5,6);


	cout<<"bias"<<endl;
	print(bias,1,6);

	cout<<"netToOut(net:matrix1)"<<endl;
	print(result,5,6);

	pureM->eleMulDsigmoid(result,matrix1,5,6);
	cout<<"eleMulDsigmoid(partError:result, outH:matrix1)"<<endl;
	print(result,5,6);

	pureM->smul(result,7,5,6);
	cout<<"result * 7"<<endl;
	print(result,5,6);


	result2 = new float[6*5];
	pureM->transpose(result2,matrix1,5,6);
	cout<<"matrix1 transpose (result2)"<<endl;
	print(result2,6,5);

	pureM->updateD_bias(bias,matrix1,5,6);
	cout<<"update delta bias, d_b:bias, partError:matrix1"<<endl;
	cout<<"bias"<<endl;
	print(bias,1,6);



	   result = new float[5*5];
	   pureM->matMatMul(result,matrix1,matrix2,5,6,5);


	   cout<<"matrix1 * matrix2"<<endl;
	   print(result,5,5);

	   delete[] result;

	   result = new float[6*5];
	   pureM->transpose(result,matrix1,5,6);
	   cout<<"matrix1 transpose"<<endl;
	   print(result,6,5);

	   pureM->matAdd(result,matrix2,6,5);
	   cout<<"matrix1 transpose plus matrix2"<<endl;
	   print(result,6,5);


	   result2 = new float[6*5];

	   pureM->matSub(result2,result,matrix2,6,5);
	   cout<<"shoudl be equal to matrix1 transpose "<<endl;
	   print(result2,6,5);


	   pureM->clean(result2,6,5);
	   pureM->clean(result,6,5);
	   cout<<"clean the two results"<<endl;
	   print(result,6,5);
	   print(result2,6,5);
	 */

	delete[] matrix1;
	delete[] matrix2;
	delete[] result;
	delete[] result2;

	return 0;
}
