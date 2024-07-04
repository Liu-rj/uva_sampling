#include <iostream>
#include <math.h>
#include <stdio.h>
#include <assert.h>

#define WARPSIZE 32
#define ITER 400

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess)
  {
    fprintf(stderr, "CUDA Runtime Error: %s\n",
            cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

// CUDA kernel to add elements of two arrays
__global__ void add(int64_t n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int64_t i = index; i < n; i += stride)
  {
    y[i] = x[i];
    for (int j = 0; j < ITER; j++)
    {
      y[i] += j;
    }
  }
}

// CUDA kernel to add elements of two arrays with overlap
__global__ void addWithOverlap(int64_t n, float *x, float *y, float *temp, int *count)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = index / WARPSIZE;
  int lane_id = index % WARPSIZE;
  int stride = WARPSIZE;

  if (warp_id == 0)
  {
    for (int64_t i = lane_id; i < n; i += stride)
    {
      while (i / WARPSIZE >= count[0])
      {
        __threadfence();
      }
      y[i] = temp[i];
      for (int j = 0; j < ITER; j++)
      {
        y[i] += j;
      }
    }
  }
  else
  {
    for (int64_t i = lane_id; i < n; i += stride)
    {
      temp[i] = x[i];
      __threadfence();
      if (lane_id == 0)
        atomicAdd(&count[0], 1);
    }
  }
}

// CUDA kernel to add elements of two arrays with overlap using cudaMemcpyAsync
__global__ void addWithCudaMemcpyAsync(int64_t n, float *x, float *y, float *temp)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  cudaMemcpyAsync(temp, x, n * sizeof(float), cudaMemcpyDefault);

  for (int64_t i = index; i < n; i += stride)
  {
    y[i] = x[i];
    for (int j = 0; j < ITER; j++)
    {
      y[i] += j;
    }
  }
}

int main(void)
{
  int64_t N = 1024 * 1024;
  float *x, *y, *d_temp_1, *d_temp_2;
  int *d_count;

  // Allocate Pinned Memory -- accessible from CPU or GPU
  cudaMallocHost((void **)&x, N * sizeof(float));
  cudaMalloc((void **)&y, N * sizeof(float));
  cudaMalloc((void **)&d_temp_1, N * sizeof(float));
  cudaMalloc((void **)&d_temp_2, N * sizeof(float));
  cudaMalloc((void **)&d_count, sizeof(int));
  float *h_y = (float *)malloc(N * sizeof(float));

  // initialize x and y arrays on the host
  for (int64_t i = 0; i < N; i++)
  {
    x[i] = 1.0f;
    // y[i] = 2.0f;
  }
  cudaMemset(d_count, 0, sizeof(int));

  // calculate result
  float res = 1.0f;
  for (int j = 0; j < ITER; j++)
  {
    res += j;
  }

  // events for timing
  cudaEvent_t startEvent, stopEvent;

  checkCuda(cudaEventCreate(&startEvent));
  checkCuda(cudaEventCreate(&stopEvent));

  // Launch kernel on the GPU without overlap
  checkCuda(cudaEventRecord(startEvent, 0));
  int blockSize = 32;
  int numBlocks = 1;
  add<<<numBlocks, blockSize>>>(N, x, y);
  checkCuda(cudaEventRecord(stopEvent, 0));
  checkCuda(cudaEventSynchronize(stopEvent));
  cudaMemcpy(h_y, y, N * sizeof(float), cudaMemcpyDeviceToHost);

  float time;
  checkCuda(cudaEventElapsedTime(&time, startEvent, stopEvent));
  printf("Add Without Overlap, Elpased Time: %f s\n", time * 1e-3);
  printf("  Host to Device bandwidth (GB/s): %f\n", N * sizeof(float) * 1e-6 / time);

  // Check for errors
  float maxError = 0.0f;
  for (int64_t i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(h_y[i] - res));
  std::cout << "  Max error: " << maxError << std::endl;

  // Launch kernel on the GPU with overlap
  checkCuda(cudaEventRecord(startEvent, 0));
  int blockSize_new = 64;
  int numBlocks_new = 1;
  addWithOverlap<<<numBlocks_new, blockSize_new>>>(N, x, y, d_temp_1, d_count);
  checkCuda(cudaEventRecord(stopEvent, 0));
  checkCuda(cudaEventSynchronize(stopEvent));
  cudaMemcpy(h_y, y, N * sizeof(float), cudaMemcpyDeviceToHost);

  checkCuda(cudaEventElapsedTime(&time, startEvent, stopEvent));
  printf("Add With Overlap, Elpased Time: %f s\n", time * 1e-3);
  printf("  Host to Device bandwidth (GB/s): %f\n", N * sizeof(float) * 1e-6 / time);

  // Check for errors
  maxError = 0.0f;
  for (int64_t i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(h_y[i] - res));
  std::cout << "  Max error: " << maxError << std::endl;

  // Launch kernel on the GPU with cudaMemcpyAsync
  checkCuda(cudaEventRecord(startEvent, 0));
  int blockSize_new_2 = 32;
  int numBlocks_new_2 = 1;
  addWithCudaMemcpyAsync<<<numBlocks_new_2, blockSize_new_2>>>(N, x, y, d_temp_2);
  checkCuda(cudaEventRecord(stopEvent, 0));
  checkCuda(cudaEventSynchronize(stopEvent));
  cudaMemcpy(h_y, y, N * sizeof(float), cudaMemcpyDeviceToHost);

  checkCuda(cudaEventElapsedTime(&time, startEvent, stopEvent));
  printf("Add With cudaMemcpyAsync, Elpased Time: %f s\n", time * 1e-3);
  printf("  Host to Device bandwidth (GB/s): %f\n", N * sizeof(float) * 1e-6 / time);

  // Check for errors
  maxError = 0.0f;
  for (int64_t i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(h_y[i] - res));
  std::cout << "  Max error: " << maxError << std::endl;

  // Free memory
  cudaFreeHost(x);
  cudaFree(y);
  cudaFree(d_temp_1);
  cudaFree(d_temp_2);
  cudaFree(d_count);
  free(h_y);

  return 0;
}