// MP 5 Inclusive Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}

#include    <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            wbLog(ERROR, "CUDA error msg: ", cudaGetErrorString(err));    	\
            return -1;                                     \
        }                                                  \
    } while(0)

__global__ void scan(float * v, int len)
{
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from here

//building tree
  int superstride = 1;
  while(superstride < len)
  //for(int superstride = 1; superstride < len; superstride *= (BLOCK_SIZE*2))
  {
    if(blockIdx.x * blockDim.x * 2 * superstride < len)
    {
    //indexing
    int idx1 = (blockIdx.x * blockDim.x *2 + threadIdx.x +1)*superstride-1;
    int idx2 = (blockIdx.x * blockDim.x *2 + threadIdx.x+blockDim.x +1)*superstride-1;

    //loading from global to shared memory
    __shared__ float XY[BLOCK_SIZE*2];

    XY[threadIdx.x] = 0.0f;
    if(idx1 < len) XY[threadIdx.x] = v[idx1];
    XY[threadIdx.x + blockDim.x] = 0.0f;
    if(idx2 < len) XY[threadIdx.x + blockDim.x] = v[idx2];

    __syncthreads();

    //for loop
    for (int stride = 1;stride <= BLOCK_SIZE; stride *= 2)
    {
      int index = (threadIdx.x+1)*stride*2 - 1;
      if(index < 2*BLOCK_SIZE)
	XY[index] += XY[index-stride];
      __syncthreads();
    }

    for (int stride = BLOCK_SIZE/2; stride > 0; stride /= 2)
    {
      __syncthreads();
      int index = (threadIdx.x+1)*stride*2 - 1;
      if(index+stride < 2*BLOCK_SIZE) 
      {
	XY[index + stride] += XY[index];
      }
    }

    //write back
    if(idx1 < len) v[idx1] = XY[threadIdx.x]; 
    if(idx2 < len) v[idx2] = XY[threadIdx.x + blockDim.x];
    __syncthreads();
    }
    superstride *= (BLOCK_SIZE*2);
    __threadfence(); ///@syncblock;
  }
  //turning point
  if (superstride != len)
    superstride /= (BLOCK_SIZE*2);
  superstride /= (BLOCK_SIZE*2);
  while (superstride > 0)
  {
    
    int idxb = ((blockIdx.x+1) * blockDim.x *2)*superstride-1;
    if(idxb+superstride < len)
    {
      float value = v[idxb];
      int idxt1 = ((blockIdx.x+1) * blockDim.x *2 + threadIdx.x +1)*superstride-1;
      int idxt2 = ((blockIdx.x+1) * blockDim.x *2 + threadIdx.x+blockDim.x +1)*superstride-1;
      if(idxt1 < len) v[idxt1] += value;
      if(idxt2 < len && (threadIdx.x+1 < blockDim.x)) v[idxt2] += value; //huge control divergence here :( i am sorry i had no choiche.
    }
    superstride /= (BLOCK_SIZE*2);
//    __syncthreads();
    __threadfence(); ///@syncblock;
  }
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    int numElements; // number of elements in the list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float*) malloc(numElements * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void**)&deviceInput, numElements*sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
    dim3 GridDim ((((numElements-1)/2)/BLOCK_SIZE)+1, 1,1);
    dim3 BlockDim (BLOCK_SIZE, 1, 1);

    wbLog(TRACE, "Grid dimensions = ", GridDim.x, " x ", GridDim.y, " x ", GridDim.z);
    wbLog(TRACE, "Block dimensions = ", BlockDim.x, " x ", BlockDim.y, " x ", BlockDim.z);

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Modify this to complete the functionality of the scan
    //@@ on the deivce
    scan<<<GridDim,BlockDim>>>(deviceInput, numElements);

    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, deviceInput, numElements*sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);

    return 0;
}

