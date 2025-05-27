# todo(huangruiteng): fix Triton Bwd (Lock) 在 BLOCK_SIZE <= 64 时的正确性 Bug，但由于其性能问题，暂不Fix
# Note:
# 0. Triton Bwd (AtomicAdd) v.s. Triton Bwd (Reorder)
#     - D=256, 临界点50w
#     - D=512, 临界点26w
#     - D=1024，临界点13w
# 1. Triton Bwd (Lock) 性能伴随 BLOCK_SIZE 的增加，严重劣化
# 2. Triton Bwd (Reorder2D) 性能伴随 BLOCK_SIZE 的增加，严重劣化，BLOCK_SIZE_M = 1时最优，增加并行度重要
# 3. Triton Bwd (Sorted) BLOCK_SIZE_UNIQUE=1, BLOCK_SIZE_SEGMENT=128时性能最佳

# index_select-fwd_bwd-performance (D=1024):
# num_indices  Triton Bwd (atomicAdd)  Triton Bwd (Reorder)  Triton Bwd (Reorder2D)  Triton Bwd (RowLock)  PyTorch Bwd (index_select)  PyTorch Bwd (scatter_add)  Compiled PyTorch Bwd (scatter_add)  Triton Bwd (Sorted)  Compiled Triton Bwd (Sorted)
# 0          16.0                0.056784              0.185184                0.183744              0.052512                    0.017216                   0.142640                            0.142656             0.299664                      0.301392
# 1          32.0                0.057120              0.190656                0.188128              0.053504                    0.008800                   0.143168                            0.143920             0.301088                      0.297456
# 2          64.0                0.053808              0.185184                0.185200              0.056096                    0.009088                   0.141424                            0.144576             0.293184                      0.296416
# 3         128.0                0.052736              0.185888                0.183840              0.054432                    0.009888                   0.140640                            0.141488             0.290048                      0.298224
# 4         256.0                0.055792              0.187392                0.189504              0.055072                    0.011584                   0.142256                            0.144384             0.303840                      0.301600
# 5         512.0                0.060752              0.191904                0.191600              0.059392                    0.014560                   0.145344                            0.142784             0.298816                      0.298944
# 6        1024.0                0.056768              0.190848                0.189152              0.058944                    0.020320                   0.145552                            0.144864             0.315712                      0.315616
# 7        2048.0                0.056704              0.190640                0.189488              0.064128                    0.031808                   0.148128                            0.148656             0.347248                      0.348128
# 8        4096.0                0.055712              0.268464                0.267360              0.078560                    0.054944                   0.145824                            0.145680             0.401040                      0.401696
# 9        8192.0                0.057392              0.270976                0.269824              0.097600                    0.103200                   0.207648                            0.205856             0.534016                      0.533040
# 10      16384.0                0.062496              0.269024                0.267360              0.145728                    0.196704                   0.316960                            0.317632             0.656736                      0.654928
# 11      32768.0                0.098048              0.272848                0.270752              0.187248                    0.380448                   0.525920                            0.527792             0.872800                      0.876944
# 12      65536.0                0.176192              0.280352                0.300544              0.283904                    0.750080                   0.942880                            0.944960             1.634576                      1.631872
# 13     131072.0                0.332992              0.306576                0.370960              0.462960                    1.484864                   1.769280                            1.773088             2.513184                      2.521488
# 14     262144.0                0.648624              0.407520                0.509808              0.804288                    2.952512                   3.425632                            3.428240             4.906432                      4.916608
# 15     524288.0                1.284688              0.585856                0.805344              1.476672                    5.895568                   6.746624                            6.743264             9.280399                      9.237375
# 16    1048576.0                2.564128              0.955200                1.357824              2.812736                   11.883968                  13.400128                           13.395680            17.112961                     17.105728

import torch
import math
import unittest
from monotorch.layers.triton_ops.index_select_op import index_select_triton_atomic_add, index_select_triton_reorder, index_select_triton_reorder_2d, index_select_torch_native_scatter_add, index_select_torch_compiled, index_select_triton_row_lock, index_select_triton_batch_lock, index_select_triton_sorted, index_select_triton_sorted_compiled, index_select_torch
import triton
import triton.testing


def custom_repr(self):
  return f'{{Tensor{tuple(self.shape)}: {original_repr(self)}}}'


original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr


class TestIndexSelect(unittest.TestCase):

  def test_index_select(self):
    device = torch.device("cuda:0")
    unique_values = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                                 dtype=torch.float32,
                                 device=device,
                                 requires_grad=True)
    indices = torch.tensor([0, 1, 0, 2], device=device)
    output = index_select_triton_atomic_add(unique_values, indices)
    expected_output = torch.tensor([[1, 2, 3], [4, 5, 6], [1, 2, 3], [7, 8, 9]],
                                   dtype=torch.float32,
                                   device=device)
    grad_output = torch.randn_like(output)
    output.backward(gradient=grad_output)
    self.assertTrue(torch.allclose(output, expected_output))

  def _run_and_check_grad(self, func, unique_values_ref, indices_ref,
                          torch_out_ref, torch_grad_ref, device, N, D,
                          num_indices, label):
    """Helper method to run forward/backward and check gradients."""
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    unique_values = torch.randn(N,
                                D,
                                dtype=torch.float32,
                                device=device,
                                requires_grad=True)
    indices = indices_ref.clone()

    output = func(unique_values, indices)
    output.sum().backward()
    grad = unique_values.grad

    self.assertTrue(
        torch.allclose(torch_out_ref, output),
        f"{label} forward mismatch, torch_out_ref: {torch_out_ref}, output: {output}"
    )
    self.assertTrue(
        torch.allclose(torch_grad_ref, grad, atol=1e-6),
        f"{label} backward mismatch, torch_grad_ref: {torch_grad_ref}, grad: {grad}"
    )

  def test_index_select_grad(self):
    device = torch.device("cuda:0")
    N, D, num_indices = 55, 68, 111
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    unique_values_ref = torch.randn(N,
                                    D,
                                    dtype=torch.float32,
                                    device=device,
                                    requires_grad=True)
    indices_ref = torch.randint(low=0,
                                high=N,
                                size=(num_indices,),
                                device=device)
    torch_out_ref = torch.index_select(unique_values_ref, 0, indices_ref)
    torch_out_ref.sum().backward()
    torch_grad_ref = unique_values_ref.grad.clone()
    unique_values_ref.grad.zero_()
    self._run_and_check_grad(index_select_triton_atomic_add, unique_values_ref,
                             indices_ref, torch_out_ref, torch_grad_ref, device,
                             N, D, num_indices, "Triton Atomic Add")
    self._run_and_check_grad(index_select_triton_reorder, unique_values_ref,
                             indices_ref, torch_out_ref, torch_grad_ref, device,
                             N, D, num_indices, "Triton Reorder")
    self._run_and_check_grad(index_select_triton_reorder_2d, unique_values_ref,
                             indices_ref, torch_out_ref, torch_grad_ref, device,
                             N, D, num_indices, "Triton Reorder 2D")
    self._run_and_check_grad(index_select_triton_row_lock, unique_values_ref,
                             indices_ref, torch_out_ref, torch_grad_ref, device,
                             N, D, num_indices, "Triton Reorder 2D")
    self._run_and_check_grad(index_select_torch, unique_values_ref, indices_ref,
                             torch_out_ref, torch_grad_ref, device, N, D,
                             num_indices, "Torch")
    self._run_and_check_grad(index_select_torch_native_scatter_add,
                             unique_values_ref, indices_ref, torch_out_ref,
                             torch_grad_ref, device, N, D, num_indices,
                             "Torch Scatter Add")
    self._run_and_check_grad(index_select_torch_compiled, unique_values_ref,
                             indices_ref, torch_out_ref, torch_grad_ref, device,
                             N, D, num_indices, "Torch Compiled Scatter Add")
    self._run_and_check_grad(index_select_triton_sorted, unique_values_ref,
                             indices_ref, torch_out_ref, torch_grad_ref, device,
                             N, D, num_indices, "Triton Sorted")
    # self._run_and_check_grad(index_select_triton_batch_lock, unique_values_ref, indices_ref, torch_out_ref, torch_grad_ref, device, N, D, num_indices, "Triton Lock")


def test_batch_index_select_grad(self):
  device = torch.device("cuda:0")
  B, N, D, num_indices = 1025, 96, 129, 2058
  # B, N, D, num_indices = 4, 55, 68, 4096

  torch.manual_seed(42)
  torch.cuda.manual_seed(42)
  unique_values_ref = torch.randn(N,
                                  D,
                                  dtype=torch.float32,
                                  device=device,
                                  requires_grad=True)
  indices_ref = torch.randint(low=0,
                              high=N,
                              size=(B, num_indices),
                              device=device)

  expanded_values = unique_values_ref.unsqueeze(0).expand(B, -1, -1)
  expanded_indices = indices_ref.unsqueeze(-1).expand(-1, -1, D)
  torch_out_ref = torch.gather(expanded_values, 1, expanded_indices)
  torch_out_ref.sum().backward()
  torch_grad_ref = unique_values_ref.grad.clone()
  unique_values_ref.grad.zero_()  # Zero grad for the reference tensor

  self._run_and_check_grad(index_select_triton_atomic_add, unique_values_ref,
                           indices_ref, torch_out_ref, torch_grad_ref, device,
                           N, D, num_indices, "Triton Atomic Add")
  self._run_and_check_grad(index_select_triton_reorder, unique_values_ref,
                           indices_ref, torch_out_ref, torch_grad_ref, device,
                           N, D, num_indices, "Triton Reorder")
  self._run_and_check_grad(index_select_triton_reorder_2d, unique_values_ref,
                           indices_ref, torch_out_ref, torch_grad_ref, device,
                           N, D, num_indices, "Triton Reorder 2D")
  self._run_and_check_grad(index_select_triton_row_lock, unique_values_ref,
                           indices_ref, torch_out_ref, torch_grad_ref, device,
                           N, D, num_indices, "Triton Row Lock")
  self._run_and_check_grad(index_select_torch, unique_values_ref, indices_ref,
                           torch_out_ref, torch_grad_ref, device, N, D,
                           num_indices, "Torch")
  self._run_and_check_grad(index_select_torch_native_scatter_add,
                           unique_values_ref, indices_ref, torch_out_ref,
                           torch_grad_ref, device, N, D, num_indices,
                           "Torch Scatter Add")
  self._run_and_check_grad(index_select_torch_compiled, unique_values_ref,
                           indices_ref, torch_out_ref, torch_grad_ref, device,
                           N, D, num_indices, "Torch Compiled Scatter Add")
  self._run_and_check_grad(index_select_triton_sorted, unique_values_ref,
                           indices_ref, torch_out_ref, torch_grad_ref, device,
                           N, D, num_indices, "Triton Sorted")
  # self._run_and_check_grad(index_select_triton_batch_lock, unique_values_ref, indices_ref, torch_out_ref, torch_grad_ref, device, N, D, num_indices, "Triton Batch Lock")


# --- Performance Benchmark ---
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['num_indices'],
        x_vals=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
        # x_vals=[
        #     8192, 2 * 8192, 4 * 8192, 8 * 8192, 16 * 8192, 32 * 8192, 64 * 8192,
        #     128 * 8192
        # ],
        # x_vals=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 2*8192, 4*8192, 8*8192, 16*8192, 32*8192, 64*8192, 128*8192],
        line_arg='provider',
        line_vals=[
            'triton', 'triton_reorder', 'triton_reorder_2d', 'triton_row_lock',
            'torch', 'scatter_add', 'compiled_scatter_add', 'triton_sorted',
            'triton_sorted_compiled'
        ],
        line_names=[
            'Triton Bwd (atomicAdd)', 'Triton Bwd (Reorder)',
            'Triton Bwd (Reorder2D)', 'Triton Bwd (RowLock)',
            'PyTorch Bwd (index_select)', 'PyTorch Bwd (scatter_add)',
            'Compiled PyTorch Bwd (scatter_add)', 'Triton Bwd (Sorted)',
            'Compiled Triton Bwd (Sorted)'
        ],
        ylabel='ms',
        plot_name='index_select-fwd_bwd-performance (D=1024, num_indices <= 4096)',
        # plot_name= 'index_select-fwd_bwd-performance (D=1024, num_indices > 4096)',
        # plot_name='index_select-fwd_bwd-performance (D=1024)',
        args={
            'D': 1024,
            'dtype': torch.float32,
            'device': 'cuda'
        }))
def benchmark(D, num_indices, provider, dtype, device):
  """
    Benchmark function for index_select forward + backward pass.
    """
  num_unique_value = int(math.sqrt(num_indices) * 2)
  unique_values = torch.randn(num_unique_value,
                              D,
                              dtype=dtype,
                              device=device,
                              requires_grad=True)
  indices = torch.randint(low=0,
                          high=num_unique_value,
                          size=(num_indices,),
                          device=device)
  output_shape = (num_indices, D)
  grad_output = torch.randn(output_shape, dtype=dtype,
                            device=device)  # 预先创建梯度, 原因是对于非标量的backward需要知道梯度形状

  if provider == 'triton':
    func = index_select_triton_atomic_add
  elif provider == 'triton_reorder':
    func = index_select_triton_reorder
  elif provider == 'triton_reorder_2d':
    func = index_select_triton_reorder_2d
  elif provider == 'triton_row_lock':
    func = index_select_triton_row_lock
  # elif provider == 'triton_batch_lock':
  #     func = index_select_triton_batch_lock
  elif provider == 'torch':
    func = index_select_torch
  elif provider == 'scatter_add':
    func = index_select_torch_native_scatter_add
  elif provider == 'compiled_scatter_add':
    func = index_select_torch_compiled
  elif provider == 'triton_sorted':
    func = index_select_triton_sorted
  elif provider == 'triton_sorted_compiled':
    func = index_select_triton_sorted_compiled
  else:
    raise ValueError(f"Unknown provider: {provider}")

  output = func(unique_values, indices)
  def run_op():
    output.backward(gradient=grad_output, retain_graph=True)

  ms, min_ms, max_ms = triton.testing.do_bench(run_op,
                                               warmup=25,
                                               rep=100,
                                               grad_to_none=[unique_values],
                                               quantiles=[0.5, 0.2, 0.8])
  return ms, min_ms, max_ms


if __name__ == '__main__':
  unittest.main(argv=['first-arg-is-ignored'], exit=False)
  benchmark.run(print_data=True, show_plots=True, save_path=".")
