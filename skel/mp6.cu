#include    <wb.h>

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)


#define Mask_width  5
#define Mask_radius 2
#define Channels 3

#define Tile_Width 16
#define OUT_Tile_Width 12 //(Tile_Width - (2* Mask_radius)) // 12

//@@ INSERT CODE HERE

__global__ void imgConvolution(const float* __restrict__ M, float* Img, float* Out,
			       int maskWidth, int maskHeight,
			       int imageWidth, int imageHeight, int imageChannels)
{
  //indexes
  int MaskRadius = 2; //maskWidth/2;
  int idx = blockIdx.x*OUT_Tile_Width + threadIdx.x - MaskRadius;
  int idy = blockIdx.y*OUT_Tile_Width + threadIdx.y - MaskRadius;
  int idc = threadIdx.z; //why create aliases ? //what kind of optimazioton nvcc does ?


  //load tiles to shared memory
  __shared__ float Tile[Tile_Width][Tile_Width][3]; //expression mush have constant value :( //TODO
  if(idx >= 0 && idy >= 0 && idx < imageWidth && idy < imageHeight)
    Tile[threadIdx.x][threadIdx.y][threadIdx.z] = Img[(idy*imageWidth + idx)*imageChannels + idc];
  else
    Tile[threadIdx.x][threadIdx.y][threadIdx.z] = 0.0f;

  __syncthreads();

  //convolve (:
  if((threadIdx.x >= MaskRadius) && (threadIdx.x < (Tile_Width-MaskRadius)) && (threadIdx.y >= MaskRadius) && (threadIdx.y < (Tile_Width - MaskRadius)) //we are inside the calculating_tile
    && (idx < imageWidth) && (idy < imageHeight)) // we are inside the picture
  {
    float value = 0.0f;
    for(int x = 0; x < maskWidth; ++x)
    {
      for(int y = 0; y < maskWidth; ++y)
      {
	value += Tile[threadIdx.x + x - Mask_radius][threadIdx.y + y - Mask_radius ][threadIdx.z] * M[(y*maskWidth) + x];
      }
    }
    //Copy all back
    Out[((idy*imageWidth+idx)*imageChannels) + idc] = value;
  }

}


int main(int argc, char* argv[]) {
    wbArg_t arg;
    int maskRows = 5;
    int maskColumns = 5;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float *  hostInputImageData;
    /*
    float hostInputImageData2[] = {
      0.48f,   0.14f,   0.54f,   0.74f,   0.18f,   0.12f,   0.14f,   0.34f,   0.90f,   0.31f,   0.65f,   0.72f,
      0.85f,   0.00f,   0.66f,   0.19f,   0.27f,   0.97f,   0.87f,   0.35f,   0.06f,   0.93f,   0.93f,   0.69f,
      0.82f,   0.29f,   0.46f,   0.11f,   0.93f,   0.80f,   0.23f,   0.90f,   0.94f,   0.80f,   0.22f,   0.21f,
      0.24f,   0.73f,   0.53f,   0.16f,   0.02f,   0.59f,   0.10f,   0.68f,   0.48f,   0.31f,   0.08f,   0.17f,
      0.88f,   0.01f,   0.64f,   0.94f,   0.23f,   0.04f,   0.05f,   0.82f,   0.36f,   0.89f,   0.27f,   0.22f,
      0.98f,   0.27f,   0.90f,   0.80f,   0.74f,   0.57f,   0.29f,   0.33f,   0.60f,   0.01f,   0.78f,   0.84f,
      0.93f,   0.40f,   0.59f,   0.20f,   0.00f,   0.44f,   0.73f,   0.47f,   0.68f,   0.53f,   0.73f,   0.44f,
      0.19f,   0.22f,   0.78f,   0.40f,   0.36f,   0.68f,   0.12f,   0.60f,   0.18f,   0.13f,   0.30f,   0.33f,
      0.29f,   0.11f,   0.68f,   0.43f,   0.33f,   0.64f,   0.89f,   0.45f,   0.30f,   0.86f,   0.65f,   0.46f,
      0.58f,   0.06f,   0.62f,   0.23f,   0.53f,   0.31f,   0.85f,   0.15f,   0.16f,   0.50f,   0.75f,   0.91f,
      0.51f,   0.95f,   0.44f,   0.60f,   0.91f,   0.96f,   0.11f,   0.50f,   0.24f,   0.01f,   0.57f,   0.33f,
      0.42f,   0.42f,   0.14f,   0.61f,   0.46f,   0.75f,   0.44f,   0.68f,   0.36f,   0.97f,   0.69f,   0.91f,
      0.01f,   0.93f,   0.62f,   0.03f,   0.48f,   0.41f,   0.29f,   0.96f,   0.56f,   0.80f,   0.03f,   0.57f,
      0.56f,   0.81f,   0.40f,   0.31f,   0.10f,   0.18f,   0.14f,   0.90f,   0.31f,   0.05f,   0.11f,   0.41f,
      0.86f,   0.19f,   0.65f,   0.78f,   0.60f,   0.79f,   0.20f,   0.88f,   0.78f,   0.54f,   0.22f,   0.72f,
      0.19f,   0.49f,   0.48f,   0.21f,   0.79f,   0.31f,   0.74f,   0.50f,   0.21f,   0.46f,   0.59f,   0.05f,
      0.85f,   0.06f,   0.57f,   0.83f,   0.18f,   0.01f,   0.12f,   0.80f,   0.28f,   0.03f,   0.72f,   0.18f,
      0.30f,   0.33f,   0.93f,   0.85f,   0.31f,   0.19f,   0.03f,   0.47f,   0.99f,   0.96f,   0.95f,   0.56f,
      0.63f,   0.24f,   0.60f,   0.73f,   0.16f,   0.88f,   0.68f,   0.38f,   0.02f,   0.51f,   0.31f,   0.46f,
      0.92f,   0.31f,   0.47f,   0.74f,   0.10f,   0.05f,   0.23f,   0.92f,   0.21f,   0.24f,   0.61f,   0.03f,
      0.66f,   0.15f,   0.81f,   0.66f,   0.85f,   0.24f,   0.90f,   0.07f,   0.49f,   0.15f,   0.49f,   0.63f,
      0.06f,   0.33f,   0.10f,   0.93f,   0.04f,   0.93f,   0.81f,   0.83f,   0.99f,   0.95f,   0.02f,   0.52f,
      0.91f,   0.13f,   0.57f,   0.65f,   0.66f,   0.71f,   0.59f,   0.03f,   0.97f,   0.22f,   0.76f,   0.40f,
      0.32f,   0.81f,   0.35f,   0.78f,   0.72f,   0.46f,   0.79f,   0.45f,   0.01f,   0.90f,   0.57f,   0.71f
    };*/
    float * hostOutputImageData;
    float * hostMaskData;
    /*
    float hostMaskData2[] = {
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };*/
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;

    arg = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(arg, 0);
    inputMaskFile = wbArg_getInputFile(arg, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);
    //imageWidth = 12;
    //imageHeight = 12;
    //imageChannels = 2;

    // log image and mask sizes;
    wbLog(TRACE, "Mask dimensions = ", maskColumns, " x ", maskRows);
    wbLog(TRACE, "Image dimensions = ", imageWidth, " x ", imageWidth, " x ", imageChannels);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);



    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");


    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData,
               hostMaskData,
               maskRows * maskColumns * sizeof(float),
               cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");


    wbTime_start(Compute, "Doing the computation on the GPU");

    //dimensions
    dim3 GridDim (((imageWidth-1)/OUT_Tile_Width+1), ((imageHeight-1)/OUT_Tile_Width +1), 1); //dimension of the grid wich cover the image
    dim3 BlockDim (Tile_Width, Tile_Width, imageChannels); // dimension of the tile/block

    wbLog(TRACE, "Grid dimensions = ", GridDim.x, " x ", GridDim.y, " x ", GridDim.z);
    wbLog(TRACE, "Block dimensions = ", BlockDim.x, " x ", BlockDim.y, " x ", BlockDim.z);

    //kernel call
    imgConvolution<<<GridDim, BlockDim>>>(deviceMaskData, deviceInputImageData, deviceOutputImageData, maskColumns, maskRows, imageWidth, imageHeight, imageChannels);

    cudaDeviceSynchronize();

    wbTime_stop(Compute, "Doing the computation on the GPU");


    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");
/*
    for(int y = 0; y < 12; ++y)
    {
      for (int x = 0; x < 12; ++x)
      {
	printf("%4.2f, %4.2f ", hostOutputImageData[(12*y+x)*2], hostOutputImageData[(12*y+x)*2 + 1]);
      }
      printf("\n");
    }
*/
    wbSolution(arg, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}
