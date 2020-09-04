#include <stdio.h>

#define N  64

__global__ void sharedMatMult( float * a, float * b, float *c){
  __shared__ float aTile[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float bTile[BLOCK_SIZE][BLOCK_SIZE];

  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  float sum = 0.0f;

  for(int k = 0; k < N; k += BLOCK_SIZE){
    aTile[threadIdx.x][threadIdx.y] = a[row * N + threadIdx.y + k];
    bTile[threadIdx.x][threadIdx.y] = b[(threadIdx.x + k) * N + col];
    __syncthreads();

    for(int i = 0; i < BLOCK_SIZE; i++){
      sum += aTile[threadIdx.x][i] * bTile[i][threadIdx.y];
    }
    __syncthreads();
  }
  c[row * N + col] = sum;
}

__global__ void matrixMulGPU( int * a, int * b, int * c )
{
  int val = 0;

  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < N && col < N)
  {
    for ( int k = 0; k < N; ++k )
      val += a[row * N + k] * b[k * N + col];
    c[row * N + col] = val;
  }
}

void matrixMulCPU( int * a, int * b, int * c )
{
  int val = 0;

  for( int row = 0; row < N; ++row )
    for( int col = 0; col < N; ++col )
    {
      val = 0;
      for ( int k = 0; k < N; ++k )
        val += a[row * N + k] * b[k * N + col];
      c[row * N + col] = val;
    }
}

int main()
{
  int *a, *b, *c_cpu, *c_gpu;

  int size = N * N * sizeof (int); // Number of bytes of an N x N matrix

  // Allocate memory
  cudaMallocManaged (&a, size);
  cudaMallocManaged (&b, size);
  cudaMallocManaged (&c_cpu, size);
  cudaMallocManaged (&c_gpu, size);

  // Initialize memory
  for( int row = 0; row < N; ++row )
    for( int col = 0; col < N; ++col )
    {
      a[row*N + col] = row;
      b[row*N + col] = col+2;
      c_cpu[row*N + col] = 0;
      c_gpu[row*N + col] = 0;
    }

  dim3 threads_per_block (8, 8, 1); // A 8 * 8 block threads
  dim3 number_of_blocks ((N + threads_per_block.x - 1)/ threads_per_block.x, (N + threads_per_block.y - 1) / threads_per_block.y, 1);

  matrixMulGPU <<< number_of_blocks, threads_per_block >>> ( a, b, c_gpu );

  cudaDeviceSynchronize(); // Wait for the GPU to finish before proceeding

  // Call the CPU version to check our work
  matrixMulCPU( a, b, c_cpu );

  // Compare the two answers to make sure they are equal
  bool error = false;
  for( int row = 0; row < N && !error; ++row )
    for( int col = 0; col < N && !error; ++col )
      if (c_cpu[row * N + col] != c_gpu[row * N + col])
      {
        printf("FOUND ERROR at c[%d][%d]\n", row, col);
        error = true;
        break;
      }
  if (!error)
    printf("Success!\n");

  // Free all our allocated memory
  cudaFree(a); cudaFree(b);
  cudaFree( c_cpu ); cudaFree( c_gpu );
}
