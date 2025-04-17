# vector-add-performance:
#            size       Triton        Torch
# 0        4096.0     8.878613     8.827586
# 1        8192.0    17.258427    16.786886
# 2       16384.0    33.391304    33.210811
# 3       32768.0    65.711231    65.361700
# 4       65536.0   127.999995   130.723400
# 5      131072.0   254.673575   245.760006
# 6      262144.0   463.698115   440.825121
# 7      524288.0   795.983841   789.590348
# 8     1048576.0  1260.307736  1232.651986
# 9     2097152.0  1833.174795  1812.055354
# 10    4194304.0  2354.586747  2347.558281
# 11    8388608.0  2864.961804  2826.350464
# 12   16777216.0  3238.011306  3216.490809
# 13   33554432.0  3467.800103  3474.982507
# 14   67108864.0  3608.003548  3616.818680
# 15  134217728.0  3669.289687  3693.794828


import torch

import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,  # *Pointer* to first input vector.
    y_ptr,  # *Pointer* to second input vector.
    output_ptr,  # *Pointer* to output vector.
    n_elements,  # Size of the vector.
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
    # NOTE: `constexpr` so it can be used as a shape value.
):
  # There are multiple 'programs' processing different data. We identify which program
  # we are here:
  pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
  # This program will process inputs that are offset from the initial data.
  # For instance, if you had a vector of length 256 and block_size of 64, the programs
  # would each access the elements [0:64, 64:128, 128:192, 192:256].
  # Note that offsets is a list of pointers:
  block_start = pid * BLOCK_SIZE
  offsets = block_start + tl.arange(0, BLOCK_SIZE)
  # Create a mask to guard memory operations against out-of-bounds accesses.
  mask = offsets < n_elements
  # Load x and y from DRAM, masking out any extra elements in case the input is not a
  # multiple of the block size.
  x = tl.load(x_ptr + offsets, mask=mask)
  y = tl.load(y_ptr + offsets, mask=mask)
  output = x + y
  # Write x + y back to DRAM.
  tl.store(output_ptr + offsets, output, mask=mask)


# %%
# Let's also declare a helper function to (1) allocate the `z` tensor
# and (2) enqueue the above kernel with appropriate grid/block sizes:


def add(x: torch.Tensor, y: torch.Tensor):
  # We need to preallocate the output.
  output = torch.empty_like(x)
  assert x.device == 'cuda' and y.device == 'cuda' and output.device == 'cuda'
  n_elements = output.numel()
  # The SPMD launch grid denotes the number of kernel instances that run in parallel.
  # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
  # In this case, we use a 1D grid where the size is the number of blocks:
  grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
  # NOTE:
  #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
  #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
  #  - Don't forget to pass meta-parameters as keywords arguments.
  add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
  # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
  # running asynchronously at this point.
  return output


# %%
# We can now use the above function to compute the element-wise sum of two `torch.tensor` objects and test its correctness:

torch.manual_seed(0)
size = 98432
x = torch.rand(size, device='cuda')
y = torch.rand(size, device='cuda')
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')

# %%
# Seems like we're good to go!

# %%
# Benchmark
# ---------
#
# We can now benchmark our custom op on vectors of increasing sizes to get a sense of how it does relative to PyTorch.
# To make things easier, Triton has a set of built-in utilities that allow us to concisely plot the performance of our custom ops.
# for different problem sizes.


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(12, 28, 1)
               ],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg=
        'provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name=
        'vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(size, provider):
  x = torch.rand(size, device=DEVICE, dtype=torch.float32)
  y = torch.rand(size, device=DEVICE, dtype=torch.float32)
  quantiles = [0.5, 0.2, 0.8]
  if provider == 'torch':
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y,
                                                 quantiles=quantiles)
  if provider == 'triton':
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y),
                                                 quantiles=quantiles)
  gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
  return gbps(ms), gbps(max_ms), gbps(min_ms)


# %%
# We can now run the decorated function above. Pass `print_data=True` to see the performance number, `show_plots=True` to plot them, and/or
# `save_path='/path/to/results/' to save them to disk along with raw CSV data:
benchmark.run(print_data=True, show_plots=True)
