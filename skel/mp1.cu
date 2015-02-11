// MP 1
#include <wb.h>

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(index < len) {
    out[index] = (in1[index]+in2[index]);
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;
  
  args = wbArg_read(argc, argv);
  
  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 = ( float * )wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 = ( float * )wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = ( float * )malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");
  
  wbLog(TRACE, "The input length is ", inputLength);
  
  wbTime_start(GPU, "Allocating GPU memory.");
  ////@@ Allocate GPU memory here
  cudaError_t err;
  
  err = cudaMalloc( (void**) &deviceInput1, inputLength*sizeof(float));
  if (err != cudaSuccess) {
    wbLog(ERROR, "GPU memory allocation error (in1) ", cudaGetErrorString(err));
  }
  
  err = cudaMalloc( (void**) &deviceInput2, inputLength*sizeof(float));
  if (err != cudaSuccess) {
    wbLog(ERROR, "GPU memory allocation error (in2) ", cudaGetErrorString(err));
  }
  
  err = cudaMalloc( (void**) &deviceOutput, inputLength*sizeof(float));
  if (err != cudaSuccess) {
    wbLog(ERROR, "GPU memory allocation error (out) ", cudaGetErrorString(err));
  }
  
  wbTime_stop(GPU, "Allocating GPU memory.");
  
  wbTime_start(GPU, "Copying input memory to the GPU.");
  ////@@ Copy memory to the GPU here
  err = cudaMemcpy(deviceInput1, hostInput1, inputLength*sizeof(float), cudaMemcpyHostToDevice );
  if (err != cudaSuccess) {
    wbLog(ERROR, "GPU memory copy error (in1) ", cudaGetErrorString(err));
  }
  
  err = cudaMemcpy(deviceInput2, hostInput2, inputLength*sizeof(float), cudaMemcpyHostToDevice );
  if (err != cudaSuccess) {
    wbLog(ERROR, "GPU memory copy error (in2) ", cudaGetErrorString(err));
  }
  
  wbTime_stop(GPU, "Copying input memory to the GPU.");
  
  ////@@ Initialize the grid and block dimensions here
  int MaxThreadPerBlock = 1024;
  dim3 GridDim ( (inputLength-1)/MaxThreadPerBlock +1, 1, 1);
  dim3 BlockDim ( MaxThreadPerBlock, 1, 1);
  
  wbLog(TRACE, "Grid dimensions x=", GridDim.x);
  wbLog(TRACE, "Grid dimensions y=", GridDim.y);
  wbLog(TRACE, "Grid dimensions z=", GridDim.z);
  
  wbLog(TRACE, "Block dimensions x=", BlockDim.x);
  wbLog(TRACE, "Block dimensions y=", BlockDim.y);
  wbLog(TRACE, "Block dimensions z=", BlockDim.z);
  
  wbTime_start(Compute, "Performing CUDA computation");
  ////@@ Launch the GPU Kernel here
  vecAdd<<<GridDim,BlockDim>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
  
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");
  
  wbTime_start(Copy, "Copying output memory to the CPU");
  ////@@ Copy the GPU memory back to the CPU here
  err = cudaMemcpy(hostOutput, deviceOutput, inputLength*sizeof(float), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    wbLog(ERROR, "GPU memory copy error (out) ", cudaGetErrorString(err));
  }
  
  wbTime_stop(Copy, "Copying output memory to the CPU");
  
  wbTime_start(GPU, "Freeing GPU Memory");
  ////@@ Free the GPU memory here
  err = cudaFree(deviceInput1);
  if (err != cudaSuccess) {
    wbLog(ERROR, "GPU memoryFree error (in1) ", cudaGetErrorString(err));
  }
  
  err = cudaFree(deviceInput2);
  if (err != cudaSuccess) {
    wbLog(ERROR, "GPU memoryFree error (in2) ", cudaGetErrorString(err));
  }
  
  err = cudaFree(deviceOutput);
  if (err != cudaSuccess) {
    wbLog(ERROR, "GPU memoryFree error (out) ", cudaGetErrorString(err));
  }
  
  wbTime_stop(GPU, "Freeing GPU Memory");
  
  wbSolution(args, hostOutput, inputLength);
  
  free(hostInput1);
  free(hostInput2);
  free(hostOutput);
  
  return 0;
}
