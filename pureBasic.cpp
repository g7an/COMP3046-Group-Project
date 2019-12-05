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
	Pure *pureM = new Pure(10.0f,10,10,wat,3);


	//float* matrix1 = new float[30]; // 5*6
	//float* matrix2 = new float[30]; // 6*5
	cout<<"fuc"<<endl;

	float part[]= {0.0f, 0.13273f ,0.0f ,-0.138802f,0.148142f,0.147739f,0.0876975f,0.146456f,0.103982f,0.11529f,0.009716f,-0.106283f,0.14782f,0.140975f,0.144031f,0.107229f,0.144601f,0.147801f,0.0669302f,0.10461f,0.0267995f,0.0916956f,0.138014f,0.129232f,0.147727f,0.135652f,-0.147928f,0.119648f,0.115036f,0.085217f,0.0718115f,0.147293f,0.0941118f,0.113986f,-0.0559046f,0.143532f,0.134491f,0.142988f,0.0762779f,0.12945f,0.0255601f,0.145337f,0.114007f,0.145694f,0.133194f,-0.0116981f,0.117452f,0.147085f,0.0524762f,0.13734f,0.0204725f,-0.0965944f,0.141445f,0.127209f,0.139522f,0.0972024f,0.134951f,0.14754f,0.0876894f,0.13141f,-0.116724f,0.144783f,0.0969846f,0.139618f,0.145951f,0.135115f,0.145725f,0.113419f,0.0294317f,0.11631f,0.0147266f,0.106122f,-0.13145f,0.128986f,0.147485f,0.103036f,0.135806f,0.148113f,0.0740535f,0.12036f}; // 8*10

	cout<<"fuc*"<<endl;
	float* bias= new float[8]; // 1*8
	float* result = NULL;
	//float* result2 = NULL;
	/*

	for(int i=0;i<30;i++){
		matrix1[i] = i * 1.1f;
		matrix2[i] = i * 1.3f;
	}

	   for(int i=0;i<80;i++){
	   part[i] = 0.022f * i;
	   }
	 */
	for(int i=0;i<8;i++){
		bias[i] = 1.0f;
	}

	//cout<<"matrix1"<<endl;
	//print(matrix1,5,6);

	cout<<"part"<<endl;
	print(part,8,10);

	cout<<"bias"<<endl;
	print(bias,1,8);

	cout<<"bias*part"<<endl;
	result = new float[10];
	pureM->matMatMul(result,bias,part,1,8,10);
	print(result,1,10);

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

	//delete[] matrix1;
	//delete[] matrix2;
	delete[] result;
	//delete[] result2;

	return 0;
}
