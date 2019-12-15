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
#include <sstream>
#include <cuda.h>
using namespace std;

#define TILE_SIZE (32)
#define WIDTH 4
/*
   cudaError_t checkCuda(cudaError_t result){
   if(result != cudaSuccess){
   cout<< "Runtime error: " << cudaGetErrorString(result)<<endl;;
   }
   return result;
   }
 */
__device__ inline float* Pitch2DMemPtr(float* BaseAddress, size_t Row, size_t Column, size_t pitch){

	return (float*)((char*)BaseAddress + Row * pitch) + Column;
}

__global__ void d_transpose(float *dev_a,size_t MatAHeight,size_t MatAWidth,size_t MatAPitch,
		float *dev_r,size_t MatRPitch){


	__shared__ float cache[TILE_SIZE][TILE_SIZE+1];

	const unsigned int tidx = threadIdx.x;
	const unsigned int tidy = threadIdx.y;

	unsigned int x= blockIdx.x*TILE_SIZE + threadIdx.x;
	unsigned int y= blockIdx.y*TILE_SIZE + threadIdx.y;

	for(int i=0;i<TILE_SIZE;i+=WIDTH){
		cache[tidy+i][tidx] =  (x<MatAWidth && y+i<MatAHeight) ? *Pitch2DMemPtr(dev_a,y+i,x,MatAPitch) : 0;

	}
	__syncthreads();

	y = blockIdx.x*TILE_SIZE + threadIdx.y;
	x = blockIdx.y*TILE_SIZE + threadIdx.x;

	for(int i=0;i<TILE_SIZE;i+=WIDTH){
		if(x<MatAHeight && y+i <MatAWidth){
			*Pitch2DMemPtr(dev_r,y+i,x,MatRPitch) = cache[tidx][tidy+i]; 
		}
	}

}
__global__ void matrixMul(float *dev_a,const size_t MatAHeight, const size_t MatAWidth, const size_t MatAPitch, 
		float *dev_b, const size_t MatBHeight,const size_t MatBWidth, const size_t MatBPitch,
		float *dev_r, const size_t MatRPitch){
	__shared__ float cacheA[TILE_SIZE][TILE_SIZE];
	__shared__ float cacheB[TILE_SIZE][TILE_SIZE];

	const unsigned int tidx = threadIdx.x;
	const unsigned int tidy = threadIdx.y;

	const unsigned int x = blockIdx.x*TILE_SIZE + tidx;
	const unsigned int y = blockIdx.y*TILE_SIZE + tidy;

	float result = 0;

	for(int i=0;i<MatAWidth;i+=TILE_SIZE){

		cacheA[tidy][tidx] =(tidx+i<MatAWidth && y<MatAHeight)? *Pitch2DMemPtr(dev_a,y,tidx+i,MatAPitch): 0;
		cacheB[tidx][tidy] =(tidy+i<MatBHeight && x<MatBWidth) ? *Pitch2DMemPtr(dev_b,tidy+i,x,MatBPitch) : 0;

		__syncthreads();

		for(int j=0;j<TILE_SIZE;j++){
			result += cacheA[tidy][j] * cacheB[tidx][j];
		}
		__syncthreads();
	}

	if(x<MatBWidth && y<MatAHeight){
		*Pitch2DMemPtr(dev_r,y,x,MatRPitch) = result;
	}

}


void init(float *x,int x_rows,int x_columns){
	for (int i = 0; i < x_rows*x_columns; i++){
		x[i] = 0;
	}
}

void outDebug(string name, float *x,int x_rows, int x_columns){
	ofstream out;
	out.open("debug.txt", ios::out | ios::app);
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


	for (int i = 0; i < x_rows*y_columns; i++){
		result[i] = 0;
	}
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

void Pure::dev_matMatMul(float *dev_result,size_t MatRPitch, float *dev_x, size_t MatXPitch,float *dev_y,size_t MatYPitch,int x_rows, int x_columns, int y_columns)
{
	/*
	   checkCuda(cudaMallocPitch(&dev_x,&MatXPitch,sizeof(float)*x_columns,x_rows));
	   checkCuda(cudaMallocPitch(&dev_y,&MatYPitch,sizeof(float)*y_columns,x_columns));
	   checkCuda(cudaMallocPitch(&dev_result,&MatRPitch,sizeof(float)*y_columns,x_rows));
	 */

	const unsigned int gridWidth= x_columns>y_columns? x_columns: y_columns;//1;
	const unsigned int gridHeight= x_rows>x_columns?x_rows: x_columns;//1;

	dim3 blocks((gridWidth+TILE_SIZE-1)/TILE_SIZE,(gridHeight+TILE_SIZE-1)/TILE_SIZE);
	dim3 threads(TILE_SIZE,TILE_SIZE);


	matrixMul<<<blocks,threads>>>(dev_x,x_rows,x_columns,MatXPitch,
			dev_y,x_columns,y_columns,MatYPitch,
			dev_result,MatRPitch);


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

void Pure::dev_transpose(float *dev_r,size_t MatRPitch, float *dev_x,size_t MatXPitch, int x_rows, int x_columns){

	dim3 blocks ((x_columns+TILE_SIZE-1)/TILE_SIZE, (x_rows+TILE_SIZE-1)/TILE_SIZE);
	dim3 threads(TILE_SIZE,WIDTH);

	d_transpose<<<blocks,threads>>>(dev_x,x_rows,x_columns,MatXPitch,dev_r,MatRPitch);
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

__device__ void Pure::matAdd(float *result, float *x, int x_rows, int x_columns) // update delta_*, weight, bias;
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
  
    if( (Row < x_rows) && (Col < x_columns) )
        result[Row * n + Col] = result[Row * n + Col] + x[Row * n + Col];
}

__device__ void Pure::matSub(float *result, float *x, float *y, int x_rows, int x_columns)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
  
    if( (Row < x_rows) && (Col < x_columns) )
        result[Row * n + Col] = x[Row * n + Col] - y[Row * n + Col];

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
		bar[i] = 1.0f;
	}

	//outDebug("bar",bar,1,partError_rows);
	//outDebug("partError",partError,partError_rows,partError_columns);

	matMatMul(foo, bar, partError, 1, partError_rows, partError_columns);

	//outDebug("foo",foo,1,partError_columns);

	matAdd(d_bias, foo, 1, partError_columns);

	//outDebug("d_bias",d_bias,1,partError_columns);
	/*
	   for(int i=0;i<partError_rows;i++){

	   bar[i] = 1;
	   }
	   matMatMul(foo, bar, partError, 1, partError_rows, partError_columns);
	   matAdd(d_bias, foo, 1, partError_columns);
	 */
	delete[] foo;
	delete[] bar;
}

void Pure::clean(float *x, int x_rows, int x_columns)
{
	int i = 0;
	for (i = 0; i < x_rows * x_columns; i++)
	{
		x[i] = 0;
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

		matMatMul(netH, net, hidWeight[i], batch_size, total_neurons[i], total_neurons[i+1]);

		netToOut(outH[i], netH, bias[i], batch_size, total_neurons[i + 1]);

		delete [] netH;

	} //forward end;

	/*
	   outDebug("out[0]",outH[0],batch_size,total_neurons[num_hidLayer]);
	   outDebug("output",outH[num_hidLayer],batch_size,total_neurons[num_hidLayer + 1]);
	   outDebug("target",target,batch_size,total_neurons[num_hidLayer + 1]);
	 */

	partError = new float[batch_size*10];

	matSub(partError, outH[num_hidLayer], target, batch_size, 10);


	//backpropagation begin
	for (int i = 0; i < 10*batch_size; i++)
	{
		loss += partError[i] * partError[i];
	}


	eleMulDsigmoid(partError, outH[num_hidLayer], batch_size, total_neurons[num_hidLayer + 1]);

	//outDebug("partError",partError,batch_size,total_neurons[num_hidLayer + 1]);

	updateD_bias(delta_bias[num_hidLayer], partError, batch_size, total_neurons[num_hidLayer + 1]);

	//outDebug("delta_bias[1]",delta_bias[num_hidLayer],1,total_neurons[num_hidLayer + 1]);
	//outDebug("output",outH[num_hidLayer],batch_size,total_neurons[num_hidLayer + 1]);

	//eleMulDsigmoid(partError, outH[num_hidLayer], batch_size, total_neurons[num_hidLayer + 1]);

	//updateD_bias(delta_bias[num_hidLayer], partError, batch_size, total_neurons[num_hidLayer + 1]);
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
		/*
		   if(i == num_hidLayer){
		   outDebug("delta_w[1]",delta_w[i],total_neurons[i],total_neurons[i + 1]);
		   }
		 */
		matAdd(hidWeight[i],delta_w[i],total_neurons[i],total_neurons[i+1]);

		clean(delta_w[i],total_neurons[i],total_neurons[i+1]);

		//print(delta_w[i],total_neurons[i],total_neurons[i+1]);


		smul(delta_bias[i],(-lr/batch_size),1,total_neurons[i+1]);
		/*

		   if(i == num_hidLayer){
		   outDebug("delta_bias[1]",delta_bias[i],1,total_neurons[i+ 1]);
		   outDebug("bias[1]",bias[i],1,total_neurons[i+ 1]);
		   }

		 */
		matAdd(bias[i],delta_bias[i],1,total_neurons[i+1]);

		clean(delta_bias[i],1,total_neurons[i+1]);

		//print(delta_bias[i],1,total_neurons[i+1]);
	}

	//cout<<"batch_loss: "<<0.5*loss/batch_size<<endl;

	return loss;
}

void Pure::train(std::vector<std::vector<float>> x, std::vector<float> y){
	int batch_num = x.size()/batch_size;
	cout<<"batch_num: "<<batch_num<<endl;

	float totalLoss = 0;
	vector<vector<float>> batch_x(batch_size);
	vector<float> batch_y(batch_size);

	for(int i=0;i<epochs;i++){

		std::chrono::steady_clock sc;
		auto start = sc.now();

		for(int j = 0;j<batch_num;j++){

			batch_x.assign(x.begin()+j*batch_size,x.begin()+(j+1)*batch_size);
			batch_y.assign(y.begin()+j*batch_size,y.begin()+(j+1)*batch_size);

			totalLoss += trainOneBatch(batch_x,batch_y);

			batch_x.clear();
			batch_y.clear();
		}

		auto end = sc.now();
		auto time_span = static_cast<std::chrono::duration<double>>(end - start);
		cout<<"Time taken for this epoch: "<< time_span.count()<<endl;;
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
		if(max < tmpOut[num_hidLayer][i]){
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

void Pure::storeWeight(){
	ofstream out;
	out.open("matrix_data.txt", ios::out | ios::app);

	for(int k=0;k<num_hidLayer+1;k++){
		out <<total_neurons[k] * total_neurons[k+1]<<" ";
		for (int i = 0; i < total_neurons[k]*total_neurons[k+1]; i++){
			out <<hidWeight[k][i] << " ";
		}
		out << endl;
	}
	out.close();
	storeBias();
}

void Pure::storeBias(){
	ofstream out;
	out.open("bias.txt", ios::out | ios::app);

	for(int k=0;k<num_hidLayer+1;k++){
		out <<total_neurons[k+1]<<" ";
		for (int i = 0; i < total_neurons[k+1]; i++){
			out <<bias[k][i] << " ";
		}
		out << endl;
	}
	out.close();
}
void Pure::loadWeight(){

	ifstream myfile("input_matrix_data.txt");
	vector<vector<float>>load;


	if (myfile.is_open())
	{
		cout << "Loading weight...\n";
		string line;
		while (getline(myfile, line))
		{
			int y;
			float x;
			vector<float> X;
			stringstream ss(line);
			ss >> y;

			for (int i = 0; i < y; i++) {
				ss >> x;
				X.push_back(x);
			}

			//if(y!=3){
			load.push_back(X);
			//}
		}

		myfile.close();
		cout << "Loading weight finished.\n";
	}
	else
		cout << "Unable to open file" << '\n';

	//cout<<"loadedWeight: "<<endl;

	for(int i=0;i<load.size();i++){
		for(int j=0;j<load[i].size();j++){
			hidWeight[i][j] = load[i][j];
		}
	}
	loadBias();
	/*

	   for(int i=0;i<load.size();i++){
	   for(int j=0;j<load[i].size();j++){
	   cout<<bias[i][j]<<" ";
	   }
	   cout<<endl;
	   }
	 */

}

void Pure::loadBias(){

	ifstream myfile("input_bias.txt");
	vector<vector<float>>load;


	if (myfile.is_open())
	{
		cout << "Loading bias...\n";
		string line;
		while (getline(myfile, line))
		{
			int y;
			float x;
			vector<float> X;
			stringstream ss(line);
			ss >> y;

			for (int i = 0; i < y; i++) {
				ss >> x;
				X.push_back(x);
			}

			//if(y!=3){
			load.push_back(X);
			//}
		}

		myfile.close();
		cout << "Loading bias finished.\n";
	}
	else
		cout << "Unable to open file" << '\n';

	//cout<<"loadedWeight: "<<endl;

	for(int i=0;i<load.size();i++){
		for(int j=0;j<load[i].size();j++){
			bias[i][j] = load[i][j];
		}
	}
	/*

	   for(int i=0;i<load.size();i++){
	   for(int j=0;j<load[i].size();j++){
	   cout<<bias[i][j]<<" ";
	   }
	   cout<<endl;
	   }
	 */

}
