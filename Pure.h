#pragma once
#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV

#endif
#include "Pure.h"
#include <vector>
#include <cuda.h>
class Pure{

	float lr;								//learning rate
	int epochs;
	int num_hidLayer;
	int* total_neurons;
	int batch_size;

	float **hidWeight;
	float **delta_w;

	float **outH;

    float **bias;
    float **delta_bias;

	float *input;
	float *target;

	float *netH;
	float *partError;

    public:

	Pure(float, int, int,int*,int);
	~Pure();

    void matMatMul(float* result, float* x, float* y,int x_rows, int x_columns,int y_columns);

    void dev_matMatMul(float *dev_result,size_t MatRPitch, float *dev_x, size_t MatXPitch,float *dev_y,size_t MatYPitch,int x_rows, int x_columns, int y_columns);

    void transpose(float* result, float* x, int x_rows, int x_columns);
    void dev_transpose(float *dev_r,size_t MatRPitch, float *dev_x,size_t MatXPitch, int x_rows, int x_columns);

    void netToOut(float* result, float* net, float* bias,int net_rows, int net_columns);
    void eleMulDsigmoid(float* partError, float* outH,int outH_rows, int outH_columns);

    void matAdd(float* result, float* x,int x_rows, int x_columns);


    void matSub(float* result, float* x,float* y, int x_rows, int x_columns);

    void smul(float* result, float y,int x_rows, int x_columns);
    
    void updateD_bias(float* d_bias,float* partError, int partError_rows,int partError_columns);

    void clean(float* x,int x_rows,int x_columns);

    void out(float* x,int x_rows,int x_columns);

    void print(float* x,int x_rows,int x_columns);

    float trainOneBatch(std::vector<std::vector<float>>x, std::vector<float>y);

    void train(std::vector<std::vector<float>>x, std::vector<float>y);

    int predict(std::vector<float>x);
	
    void storeWeight(); 
    void storeBias(); 

    void loadWeight(); 
    void loadBias(); 


};

