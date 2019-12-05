#include "Pure.h"
#include <iostream>
#include <fstream>
#include <iostream>
#include <cmath>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <random>
#include <chrono>
#include <string>
using namespace std;
<<<<<<< HEAD
void init(float *x,int x_rows,int x_columns){
	for (int i = 0; i < x_rows*x_columns; i++){
		x[i] = 0;
	}
}
=======
>>>>>>> e134b56323fa6d7b9f6362c5f3a84918987d2347

void outDebug(string name, float *x,int x_rows, int x_columns){
	ofstream out;
	out.open("matrix_data.txt", ios::out | ios::app);
	out << endl;
	out << name <<endl;;
	out << x_rows  << " " << x_columns << endl;

	for (int i = 0; i < x_rows; i++){
		for (int j = 0; j < x_columns; j++){
			out <<x[i*x_columns + j] << " ";
		}
		out << endl;
	}
	out.close();
}

void Pure::print(float* x, int x_rows, int x_columns ){
	for(int i=0;i<x_rows;i++){
		for(int j=0;j<x_columns;j++){

			cout<<x[i*x_columns+ j]<<" ";
		}
		cout<<endl;
	}
}

Pure::Pure(float lr, int epochs, int batch_size, int *layerSize, int layerSizeLen) : lr(lr), epochs(epochs), batch_size(batch_size)
{
	num_hidLayer = layerSizeLen - 2; // without the input and output layer
	int num_weights = num_hidLayer + 1;

	total_neurons = new int[num_hidLayer + 2]; // num nodes in all layers
	for (int i = 1; i <= num_hidLayer; i++)
	{
		total_neurons[i] = layerSize[i]; //num_neurons[i];
	}
	total_neurons[0] = 784;
	total_neurons[num_hidLayer + 1] = 10;

	hidWeight = new float *[num_weights];
	delta_w = new float *[num_weights];
	outH = new float *[num_weights];
	bias = new float *[num_weights];
	delta_bias = new float *[num_weights];

	auto seed = std::chrono::system_clock::now().time_since_epoch().count(); //seed
	std::default_random_engine dre(seed);                                    //engine
	std::uniform_real_distribution<float> di(-1, 1);                         //distribution

	for (int i = 0; i < num_weights; i++)
	{

		outH[i] = new float[batch_size * total_neurons[i + 1]];
		for (int j = 0; j < batch_size * total_neurons[i + 1]; j++)
		{
			outH[i][j] = 0;
		}

		hidWeight[i] = new float[total_neurons[i] * total_neurons[i + 1]];
		delta_w[i] = new float[total_neurons[i] * total_neurons[i + 1]];

		for (int j = 0; j < total_neurons[i] * total_neurons[i + 1]; j++)
		{
			hidWeight[i][j] = di(dre);
			delta_w[i][j] = 0;
		}

		bias[i] = new float[total_neurons[i + 1]];
		delta_bias[i] = new float[total_neurons[i + 1]];

		for (int j = 0; j < total_neurons[i + 1]; j++)
		{
			bias[i][j] = di(dre);
			delta_bias[i][j] = 0;
		}
	}
	input = new float[total_neurons[0] * batch_size];
	target = new float[total_neurons[num_weights] * batch_size];

	netH = NULL;
	partError= NULL;

}

Pure::~Pure()
{
	for (int i = 0; i < num_hidLayer + 1; i++)
	{
		delete[] outH[i];
		delete[] hidWeight[i];
		delete[] bias[i];
		delete[] delta_w[i];
		delete[] delta_bias[i];
	}

	delete[] outH;
	delete[] hidWeight;
	delete[] bias;
	delete[] delta_w;
	delete[] delta_bias;
	delete[] input;
	delete[] target;
	//delete[] netH;
	delete[] partError;
	delete[] total_neurons;
}
void Pure::matMatMul(float *result, float *x, float *y, int x_rows, int x_columns, int y_columns)
{


<<<<<<< HEAD
	for (int i = 0; i < x_rows*y_columns; i++){
		result[i] = 0;
	}
=======
>>>>>>> e134b56323fa6d7b9f6362c5f3a84918987d2347
	//    delete[] result;
	//   result = new float[x_rows * y_columns];
	int i = 0;
	int j = 0;
	int k = 0;


	for (i = 0; i < x_rows; i++)
	{
		for (j = 0; j < y_columns; j++)
		{

			for (k = 0; k < x_columns; k++)
			{
				result[i * y_columns + j] += x[i * x_columns + k] * y[k * y_columns + j];
			}
		}
	}

}

void Pure::transpose(float *result, float *x, int x_rows, int x_columns)
{
	//   delete[] result;
	//  result = new float[x_rows * x_columns];

	int i = 0;
	int j = 0;

	for (i = 0; i < x_columns; i++)
	{
		for (j = 0; j < x_rows; j++)
		{
			result[i * x_rows + j] = x[j * x_columns + i];
		}
	}
}

void Pure::netToOut(float *result, float *net, float *bias, int net_rows, int net_columns)
{
	int i = 0;
	int j = 0;
	float foo = 0;

	for (i = 0; i < net_rows; i++)
	{
		for (j = 0; j < net_columns; j++)
		{
			foo = net[i * net_columns + j] + bias[j];
			result[i * net_columns + j] = 1 / (1 + exp(-1 * foo));
		}
	}
}

void Pure::eleMulDsigmoid(float *partError, float *outH, int outH_rows, int outH_columns) // update partError
{
	int i = 0;
	int j = 0;
	float foo = 0;

	for (i = 0; i < outH_rows; i++)
	{
		for (j = 0; j < outH_columns; j++)
		{
			foo = outH[i * outH_columns + j];
			partError[i * outH_columns + j] *= foo * (1 - foo);
		}
	}
}

void Pure::matAdd(float *result, float *x, int x_rows, int x_columns) // update delta_*, weight, bias;
{
	int i = 0;
	int j = 0;

	for (i = 0; i < x_rows; i++)
	{
		for (j = 0; j < x_columns; j++)
		{
			result[i * x_columns + j] += x[i * x_columns + j];
		}
	}
}
void Pure::matSub(float *result, float *x, float *y, int x_rows, int x_columns)
{ //the first step to produce a partError

	//delete[] result;
	//result = new float[x_rows * x_columns];
	for (int i = 0; i < x_rows * x_columns; i++)
	{
		result[i] = x[i] - y[i];
	}
}

void Pure::smul(float *result, float y, int x_rows, int x_columns) // multiple -lr/batch_size
{
	int i = 0;
	for (i = 0; i < x_rows * x_columns; i++)
	{
		result[i] *= y;
	}
}

void Pure::updateD_bias(float *d_bias, float *partError, int partError_rows, int partError_columns)
{
	float *foo = new float[partError_columns];
	float *bar = new float[partError_rows];

	for (int i = 0; i < partError_rows; i++)
	{
<<<<<<< HEAD
		bar[i] = 1.0f;
	}

//outDebug("bar",bar,1,partError_rows);
//outDebug("partError",partError,partError_rows,partError_columns);

	matMatMul(foo, bar, partError, 1, partError_rows, partError_columns);

//outDebug("foo",foo,1,partError_columns);

	matAdd(d_bias, foo, 1, partError_columns);

//outDebug("d_bias",d_bias,1,partError_columns);

=======
		bar[i] = 1;
	}
	matMatMul(foo, bar, partError, 1, partError_rows, partError_columns);
	matAdd(d_bias, foo, 1, partError_columns);
>>>>>>> e134b56323fa6d7b9f6362c5f3a84918987d2347
	delete[] foo;
	delete[] bar;
}

void Pure::clean(float *x, int x_rows, int x_columns)
{
	int i = 0;
	for (i = 0; i < x_rows * x_columns; i++)
	{
		x[i] *= 0;
	}
}

float Pure::trainOneBatch(std::vector<std::vector<float>> x, std::vector<float> y)
{
	float loss = 0;
	for (int i = 0; i < batch_size; i++)
	{
		for (int j = 0; j < total_neurons[0]; j++)
		{
			input[i * total_neurons[0] + j] = x[i][j];
		}
	}


	for (int i = 0; i < batch_size; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			target[i * 10 + j] = y[i] == j ? 1 : 0;
		}
	} //initialization of input and target;

	float *net = NULL;

	for (int i = 0; i < num_hidLayer + 1; i++)
	{ //forward begin
		if (i == 0)
		{
			net = input;
		}
		else
		{
			net = outH[i - 1];
		}
		netH = new float[batch_size*total_neurons[i+1]];
<<<<<<< HEAD

=======
>>>>>>> e134b56323fa6d7b9f6362c5f3a84918987d2347
		matMatMul(netH, net, hidWeight[i], batch_size, total_neurons[i], total_neurons[i+1]);

		netToOut(outH[i], netH, bias[i], batch_size, total_neurons[i + 1]);

		delete [] netH;

	} //forward end;

<<<<<<< HEAD
/*
outDebug("out[0]",outH[0],batch_size,total_neurons[num_hidLayer]);
*/
outDebug("output",outH[num_hidLayer],batch_size,total_neurons[num_hidLayer + 1]);
outDebug("target",target,batch_size,total_neurons[num_hidLayer + 1]);
=======
>>>>>>> e134b56323fa6d7b9f6362c5f3a84918987d2347

	partError = new float[batch_size*10];

	matSub(partError, outH[num_hidLayer], target, batch_size, 10);
<<<<<<< HEAD


=======
>>>>>>> e134b56323fa6d7b9f6362c5f3a84918987d2347
	//backpropagation begin
	for (int i = 0; i < 10*batch_size; i++)
	{
		loss += partError[i] * partError[i];
	}

<<<<<<< HEAD

	eleMulDsigmoid(partError, outH[num_hidLayer], batch_size, total_neurons[num_hidLayer + 1]);

//outDebug("partError",partError,batch_size,total_neurons[num_hidLayer + 1]);

	updateD_bias(delta_bias[num_hidLayer], partError, batch_size, total_neurons[num_hidLayer + 1]);

//outDebug("delta_bias[1]",delta_bias[num_hidLayer],1,total_neurons[num_hidLayer + 1]);
=======
outDebug("output",outH[num_hidLayer],batch_size,total_neurons[num_hidLayer + 1]);

	eleMulDsigmoid(partError, outH[num_hidLayer], batch_size, total_neurons[num_hidLayer + 1]);

	updateD_bias(delta_bias[num_hidLayer], partError, batch_size, total_neurons[num_hidLayer + 1]);
>>>>>>> e134b56323fa6d7b9f6362c5f3a84918987d2347
	/*
	   vector<float *>tmp2(num_hidLayer+1);
	   vector<float *>tmpTrans1(num_hidLayer+1);

	   vector<float *>tmpTrans(num_hidLayer);
	   vector<float *>tmpLoss(num_hidLayer);
	 */

	float* tmp2;
	float* tmpLoss;
	float* tmpTrans;


	for (int i = num_hidLayer; i >= 0; i--)
	{
		if (i == 0)
		{
			net = input;
		}
		else
		{
			net = outH[i - 1];
		}

		if (i != num_hidLayer)
		{
			tmpLoss= new float[batch_size * total_neurons[i + 2]];
			for (int j = 0; j < batch_size * total_neurons[i + 2]; j++)
			{
				tmpLoss[j] = partError[j];
			}

			delete [] partError;
			//cout<<"hidWeight: "<<hidWeight[i]->dim()[0]<<" : "<<hidWeight[i]->dim()[1]<<endl;

			tmpTrans= new float[total_neurons[i+1]*total_neurons[i+2]];
			transpose(tmpTrans, hidWeight[i + 1], total_neurons[i + 1], total_neurons[i + 2]);

			partError= new float[batch_size * total_neurons[i+1]];
			matMatMul(partError, tmpLoss, tmpTrans, batch_size, total_neurons[i + 2], total_neurons[i + 1]);

			eleMulDsigmoid(partError, outH[i], batch_size, total_neurons[i + 1]);

			delete[] tmpLoss;
			delete[] tmpTrans;

			updateD_bias(delta_bias[i], partError, batch_size, total_neurons[i + 1]);
		}


		tmpTrans = new float[total_neurons[i]*batch_size];

		transpose(tmpTrans, net, batch_size, total_neurons[i]);


		tmp2 = new float[total_neurons[i]*total_neurons[i+1]];
		matMatMul(tmp2, tmpTrans, partError, total_neurons[i], batch_size, total_neurons[i + 1]);


		matAdd(delta_w[i], tmp2, total_neurons[i], total_neurons[i + 1]);

		delete[] tmpTrans;
		delete[] tmp2;

	}


	//backpropagation end


	//update weights and bias

	for (int i = 0; i < num_hidLayer + 1; i++)
	{

		smul(delta_w[i],(-lr/batch_size),total_neurons[i],total_neurons[i+1]);
<<<<<<< HEAD
/*
if(i == num_hidLayer){
outDebug("delta_w[1]",delta_w[i],total_neurons[i],total_neurons[i + 1]);
}
*/

=======
>>>>>>> e134b56323fa6d7b9f6362c5f3a84918987d2347
		matAdd(hidWeight[i],delta_w[i],total_neurons[i],total_neurons[i+1]);

		clean(delta_w[i],total_neurons[i],total_neurons[i+1]);

		//print(delta_w[i],total_neurons[i],total_neurons[i+1]);


		smul(delta_bias[i],(-lr/batch_size),1,total_neurons[i+1]);
<<<<<<< HEAD

/*
if(i == num_hidLayer){
outDebug("delta_bias[1]",delta_bias[i],1,total_neurons[i+ 1]);
outDebug("bias[1]",bias[i],1,total_neurons[i+ 1]);
}
*/

=======
>>>>>>> e134b56323fa6d7b9f6362c5f3a84918987d2347
		matAdd(bias[i],delta_bias[i],1,total_neurons[i+1]);

		clean(delta_bias[i],1,total_neurons[i+1]);

		//print(delta_bias[i],1,total_neurons[i+1]);
	}

	//cout<<"batch_loss: "<<0.5*loss/batch_size<<endl;

	return loss;
}

float Pure::train(std::vector<std::vector<float>> x, std::vector<float> y){
	int batch_num = x.size()/batch_size;


	float totalLoss = 0;
	vector<vector<float>> batch_x(batch_size);
	vector<float> batch_y(batch_size);

	for(int i=0;i<epochs;i++){

		for(int j = 0;j<batch_num;j++){

			batch_x.assign(x.begin()+j*batch_size,x.begin()+(j+1)*batch_size);
			batch_y.assign(y.begin()+j*batch_size,y.begin()+(j+1)*batch_size);

			totalLoss += trainOneBatch(batch_x,batch_y);

			batch_x.clear();
			batch_y.clear();
		}
		cout<<"testLoss: "<<0.5*totalLoss/(batch_num*batch_size)<<endl;
		totalLoss = 0;
	}


}
int Pure::predict(std::vector<float> x){

	float* in = new float[total_neurons[0]];
	int result = 0;

	float** tmpOut= new float* [num_hidLayer+1];

	for (int i = 0; i < total_neurons[0]; i++)
	{
		in[i] = x[i];
	}

	for (int i = 0; i < num_hidLayer+1; i++)
	{

		tmpOut[i] = new float[total_neurons[i + 1]];
		for (int j = 0; j < total_neurons[i + 1]; j++)
		{
			tmpOut[i][j] = 0;
		}
	}


	//initialization of input 

	float *net = NULL;

	for (int i = 0; i < num_hidLayer + 1; i++)
	{ //forward begin
		if (i == 0)
		{
			net = in;
		}
		else
		{
			net = tmpOut[i - 1];
		}
		netH = new float[total_neurons[i+1]];
		matMatMul(netH, net, hidWeight[i], 1, total_neurons[i], total_neurons[i+1]);

		netToOut(tmpOut[i], netH, bias[i], 1 , total_neurons[i + 1]);

		delete [] netH;

	} //forward end;


	float max = tmpOut[num_hidLayer][0];
	for(int i=1;i<10;i++){
		if(max > tmpOut[num_hidLayer][i]){
			max = tmpOut[num_hidLayer][i];
			result = i;
		}
	}

	delete[] in;

	for (int i = 0; i < num_hidLayer+1; i++)
	{

		delete [] tmpOut[i];
	}
	delete [] tmpOut;

	return result;
}

void Pure::out(float *x,int x_rows, int x_columns){
	ofstream out;
	out.open("matrix_data.txt", ios::out | ios::app);
	out << endl;
	out << x_rows  << " " << x_columns << endl;

	for (int i = 0; i < x_rows; i++){
		for (int j = 0; j < x_columns; j++){
			out <<x[i*x_columns + j] << " ";
		}
		out << endl;
	}
	out.close();
}
