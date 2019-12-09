#include<iostream>
#include<stdio.h>
#include<chrono>
#include"Pure.h"

using namespace std;

cudaError_t checkCuda(cudaError_t result){
	if(result != cudaSuccess){
		printf("Runtime Error: %s\n",cudaGetErrorString(result));

	}
	return result;
}

__device__ inline float* Pitch2DMemPtr(float* BaseAddress, size_t Row, size_t Column, size_t pitch){

	return (float*)((char*)BaseAddress + Row * pitch) + Column;
}

float* serialTrans(float* a, size_t MatAHeight, size_t MatAWidth){

	float* r = new float[MatAHeight*MatAWidth];
	for(int i=0;i<MatAWidth;i++){
		for(int j=0;j<MatAHeight;j++){
			r[MatAHeight*i + j]= a[MatAWidth* j + i];
		}
	}
	return r;
}


float* serialMatMul(float* a, int MatAHeight,int MatAWidth,
		float* b,int MatBHeight,int MatBWidth
		){

	float*r = new float [MatBHeight * MatBWidth];
	int N = MatBHeight * MatBWidth;


	for(int i=0;i<N;i++){
		r[i]=0; 
	}

	std::chrono::steady_clock sc;
	auto start = sc.now();

	for(int i=0;i<MatAHeight;i++){

		for(int j=0;j<MatBWidth;j++){
			for(int c=0;c<MatAWidth;c++){
				r[i*MatBWidth+j]+= a[i*MatAWidth+c]* b[j+c*MatBWidth];
			}
		}
	}
	auto end = sc.now();
	auto time_span = static_cast<std::chrono::duration<double>>(end - start);
	//printf("Time taken for CPU: %.2fs\n", time_span.count());

	return r;
}
int main(){

	int wat [] = {10,10,10};
	Pure *pureM = new Pure(10.0f,10,10,wat,3);

	float *a,*b,*r,*t,*dev_a,*dev_b,*dev_r,*dev_t; // r = a * b , t = r transpose
	size_t MatAPitch,MatBPitch,MatRPitch,MatTPitch;
	float *CPUr,*CPUt;

	cudaEvent_t start,stop;

	checkCuda(cudaEventCreate(&start));
	checkCuda(cudaEventCreate(&stop));



	const unsigned int MatAHeight =64;// 32;
	const unsigned int MatAWidth= 784;//32;
	const unsigned int MatBHeight = 784;//32;
	const unsigned int MatBWidth= 100;//1;

	//const unsigned int gridWidth= MatAWidth>MatBWidth ? MatAWidth: MatBWidth;//1;
	//const unsigned int gridHeight= MatAHeight>MatBHeight ? MatAHeight: MatBHeight;//1;


	a = new float [MatAHeight * MatAWidth];
	b = new float [MatBHeight * MatBWidth];
	r = new float [MatAHeight * MatBWidth];
	t = new float [MatAHeight * MatBWidth];

	for(int i=0;i<MatAHeight*MatAWidth;i++){
		a[i] = i;
	}

	for(int i=0;i<MatBHeight*MatBWidth;i++){
		b[i] = 2;
	}

	CPUr = serialMatMul(a,MatAHeight,MatAWidth,b,MatBHeight,MatBWidth);

	checkCuda(cudaMallocPitch(&dev_a,&MatAPitch,sizeof(float)*MatAWidth,MatAHeight));
	checkCuda(cudaMallocPitch(&dev_b,&MatBPitch,sizeof(float)*MatBWidth,MatBHeight));
	checkCuda(cudaMallocPitch(&dev_r,&MatRPitch,sizeof(float)*MatBWidth,MatAHeight));

	//dim3 blocks((Nsqrt+TILE_SIZE-1)/TILE_SIZE,(Nsqrt+TILE_SIZE-1)/TILE_SIZE);
	//dim3 blocks((gridWidth+TILE_SIZE-1)/TILE_SIZE,(gridHeight+TILE_SIZE-1)/TILE_SIZE);
	//dim3 threads(TILE_SIZE,TILE_SIZE);



	checkCuda(cudaEventRecord(start,0));

	checkCuda(cudaMemcpy2D(dev_a,MatAPitch,a,sizeof(float)*MatAWidth,sizeof(float)*MatAWidth,MatAHeight,cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy2D(dev_b,MatBPitch,b,sizeof(float)*MatBWidth,sizeof(float)*MatBWidth,MatBHeight,cudaMemcpyHostToDevice));

	pureM->dev_matMatMul(dev_r,MatRPitch,dev_a,MatAPitch,dev_b,MatBPitch,MatAHeight,MatAWidth,MatBWidth);
	/*
	   matrixMul<<<blocks,threads>>>(dev_a,MatAHeight,MatAWidth,MatAPitch,
	   dev_b,MatBHeight,MatBWidth,MatBPitch,
	   dev_r,MatRPitch);
	 */

	checkCuda(cudaMemcpy2D(r,sizeof(float)*MatBWidth,dev_r,MatRPitch,sizeof(float)*MatBWidth,MatAHeight,cudaMemcpyDeviceToHost));

	checkCuda(cudaEventRecord(stop,0));
	checkCuda(cudaEventSynchronize(stop));
	float elapsedTime;
	checkCuda(cudaEventElapsedTime(&elapsedTime,start,stop));
	//cout<<"GPU time: "<<elapsedTime<<endl;


	cudaDeviceSynchronize();//comment this line,when really use it!!!!!



	for(int i = 0;i<MatAHeight*MatBWidth;i++){
		if(r[i] - CPUr[i] > 1e-3 ||  r[i] - CPUr[i] < -1e-3){
			printf("Wrong result! i:%d GPU:%f CPU:%f \n",i,r[i],CPUr[i]);
			break;
		}
	}

	printf("Finish \n");


	CPUt = serialTrans(r,MatAHeight,MatBWidth);



	checkCuda(cudaMallocPitch(&dev_t,&MatTPitch,sizeof(float)*MatAHeight,MatBWidth));
	//checkCuda(cudaMallocPitch(&dev_t,&MatTPitch,sizeof(float)*MatAHeight,MatBWidth));


	pureM->dev_transpose(dev_t,MatTPitch,dev_r,MatRPitch,MatAHeight,MatBWidth);


	checkCuda(cudaMemcpy2D(t,sizeof(float)*MatAHeight,dev_t,MatTPitch,sizeof(float)*MatAHeight,MatBWidth,cudaMemcpyDeviceToHost));

	cudaDeviceSynchronize();//comment this line,when really use it!!!!!

	for(int i = 0;i<MatAHeight*MatBWidth;i++){
		if(t[i] - CPUt[i] > 1e-3 ||  t[i] - CPUt[i] < -1e-3){
			printf("Wrong result! i:%d GPU:%f CPU:%f \n",i,t[i],CPUt[i]);
			break;
		}
	}

	printf("Finish \n");

	delete [] a;
	delete [] b;
	delete [] r;
	delete [] t;
	//delete [] CPUr;
	//delete [] CPUt;
	checkCuda(cudaFree(dev_a));
	checkCuda(cudaFree(dev_b));
	checkCuda(cudaFree(dev_r));
	checkCuda(cudaFree(dev_t));
	checkCuda(cudaEventDestroy(start));
	checkCuda(cudaEventDestroy(stop));

}




