#include<iostream>
#include<stdio.h>
#include <ctime>
#include <chrono>

#define TILE_SIZE (32)
using namespace std;

cudaError_t checkCuda(cudaError_t result){
	if(result != cudaSuccess){
		cout<< "Runtime error: " << cudaGetErrorString(result)<<endl;;
	}
	return result;
}
__device__ inline float* Pitch2DMemPtr(float* BaseAddress, size_t Row, size_t Column, size_t pitch){

	return (float*)((char*)BaseAddress + Row * pitch) + Column;
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
void matrixMultiple(){
	//const int N = 32*32;
	//const int Nsqrt=32;

	float *a,*b,*r,*dev_a,*dev_b,*dev_r;
	size_t MatAPitch,MatBPitch,MatRPitch;
	float *CPUr;

	cudaEvent_t start,stop;

	checkCuda(cudaEventCreate(&start));
	checkCuda(cudaEventCreate(&stop));



	const unsigned int MatAHeight =128;// 32;
	const unsigned int MatAWidth= 785;//32;
	const unsigned int MatBHeight = 785;//32;
	const unsigned int MatBWidth= 100;//1;

	const unsigned int gridWidth= MatAWidth>MatBWidth ? MatAWidth: MatBWidth;//1;
	const unsigned int gridHeight= MatAHeight>MatBHeight ? MatAHeight: MatBHeight;//1;


	a = new float [MatAHeight * MatAWidth];
	b = new float [MatBHeight * MatBWidth];
	r = new float [MatAHeight * MatBWidth];

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
	dim3 blocks((gridWidth+TILE_SIZE-1)/TILE_SIZE,(gridHeight+TILE_SIZE-1)/TILE_SIZE);
	dim3 threads(TILE_SIZE,TILE_SIZE);



	checkCuda(cudaEventRecord(start,0));

	checkCuda(cudaMemcpy2D(dev_a,MatAPitch,a,sizeof(float)*MatAWidth,sizeof(float)*MatAWidth,MatAHeight,cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy2D(dev_b,MatBPitch,b,sizeof(float)*MatBWidth,sizeof(float)*MatBWidth,MatBHeight,cudaMemcpyHostToDevice));


	matrixMul<<<blocks,threads>>>(dev_a,MatAHeight,MatAWidth,MatAPitch,
			dev_b,MatBHeight,MatBWidth,MatBPitch,
			dev_r,MatRPitch);

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

	delete [] a;
	delete [] b;
	delete [] r;
	delete [] CPUr;
	checkCuda(cudaFree(dev_a));
	checkCuda(cudaFree(dev_b));
	checkCuda(cudaFree(dev_r));
	checkCuda(cudaEventDestroy(start));
	checkCuda(cudaEventDestroy(stop));
}

int main(void){
	matrixMultiple();
	return 0;
}
