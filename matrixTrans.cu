#include<iostream>
using namespace std;

#define N 32*32
#define Nsqrt 32
#define TILE_SIZE 16
#define WIDTH 4

cudaError_t cudaCheck (cudaError_t result){
	if(result != cudaSuccess){
		cout<<"Runtime Error: "<<cudaGetErrorString(result)<<endl;
	}
	return result;
} 

__device__ inline float* Pitch2DMemPtr(float* BaseAddress, size_t Row,size_t Column, size_t pitch){
	return (float*)((char*)BaseAddress + Row*pitch)+ Column;

} 
__global__ void transpose(float *dev_a,size_t MatAHeight,size_t MatAWidth,size_t MatAPitch,
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
float* serial(float* a, size_t MatAHeight, size_t MatAWidth){
	float* r = new float[MatAHeight*MatAWidth];
	for(int i=0;i<MatAWidth;i++){
		for(int j=0;j<MatAHeight;j++){
			r[MatAHeight*i + j]= a[MatAWidth* j + i];
		}
	}
	return r;
}
int main(void){
	float *a,*r,*dev_a,*dev_r,*CPUr;
	size_t MatAPitch,MatRPitch;

	const unsigned int MatAHeight = 64; 
	const unsigned int MatAWidth = 784; 

	a = new float [MatAHeight*MatAWidth];
	r = new float [MatAWidth*MatAHeight];

	for(int i=0;i<MatAHeight * MatAWidth;i++){
		a[i] = i;
		r[i] = 0;
	}
	//dim3 blocks ((MatAWidth+TILE_SIZE-1)/TILE_SIZE, (MatAHeight+WIDTH-1)/WIDTH);
	dim3 blocks ((MatAWidth+TILE_SIZE-1)/TILE_SIZE, (MatAHeight+TILE_SIZE-1)/TILE_SIZE);
	dim3 threads(TILE_SIZE,WIDTH);


	cudaCheck(cudaMallocPitch(&dev_a,&MatAPitch,sizeof(float)*MatAWidth,MatAHeight));
	cudaCheck(cudaMallocPitch(&dev_r,&MatRPitch,sizeof(float)*MatAHeight,MatAWidth));

	cudaCheck(cudaMemcpy2D(dev_a,MatAPitch,a,sizeof(float)*MatAWidth,sizeof(float)*MatAWidth,MatAHeight,cudaMemcpyHostToDevice));

	transpose<<<blocks,threads>>>(dev_a,MatAHeight,MatAWidth,MatAPitch,dev_r,MatRPitch);

	cudaCheck(cudaMemcpy2D(r,sizeof(float)*MatAHeight,dev_r,MatRPitch,sizeof(float)*MatAHeight,MatAWidth,cudaMemcpyDeviceToHost));

	cudaDeviceSynchronize();

	CPUr = serial(a,MatAHeight,MatAWidth);


	for(int i=0;i<MatAHeight*MatAWidth;i++){
		if(r[i]-CPUr[i]<-1e-3 || r[i]-CPUr[i]>1e-3)
		{
			cout<<"Wrong result "<<"r[i]: "<<r[i] <<" CPUr[i]: "<<CPUr[i]<<" i: "<<i<<endl;
			break;
		}
	}
	cout<<"finish" <<endl;

	delete [] a;
	delete [] r;
	//delete [] CPUr;
	cudaFree(dev_a);
	cudaFree(dev_r);

	return 0;
}









