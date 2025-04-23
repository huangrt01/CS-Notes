### tools

compute sanitizer: https://docs.nvidia.com/cuda/compute-sanitizer/index.html

### cuda debug

CUDA_LAUNCH_BLOCKING=1


# check error instantly

load_keys_kernel<<<grid, BLOCK_SIZE>>>(gpu_keys_ptr, keys_, values_, num_keys,
                                         conflict_zone_size_, capacity_);
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
  throw std::runtime_error(cudaGetErrorString(err));
}


https://discuss.pytorch.org/t/how-do-i-debug-a-cuda-error-from-pytorch/152486/5

If no proper CUDA error checking is performed the next CUDA operation might be running into the “sticky” error
and report the error message, so I think you are right that neither clone()
nor inverse are the root cause of the issue but are just reporting “an error” as the CUDA context is corrupt.


# jupyter notebook

pip3 install wurlitzer
%load_ext wurlitzer



