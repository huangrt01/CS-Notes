

__shared__ T XY[stream_block_size];
__shared__ unsigned int sbid;
__shared__ T previous_sum;

if (threadIdx.x == 0)
    sbid = atomicAdd(DCounter, 1);
__syncthreads();

const int bid = sbid;

int i = bid * stream_block_size + threadIdx.x;
if (i < InputSize)
    XY[threadIdx.x] = X[i];

BrentKung(XY);

if (threadIdx.x == 0) {
    cuda::atomic_ref<int, cuda::thread_scope_device> aref(flags[bid]);
    while (aref.load(cuda::memory_order_acquire) == 0) {}
    previous_sum = scan_value[bid];
    scan_value[bid + 1] = previous_sum + XY[block_size - 1];
    __threadfence();
    atomicAdd(&flags[bid + 1], 1);
}
__syncthreads();

if (i < InputSize)
    Y[i] = previous_sum + XY[threadIdx.x];