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
# num_indices  Triton Bwd (atomicAdd)  Triton Bwd (Reorder)  Triton Bwd (Reorder2D)  Triton Bwd (RowLock)  PyTorch Bwd (index_add)  PyTorch Bwd (scatter_add)  Compiled PyTorch Bwd (scatter_add)  Triton Bwd (Sorted)  Compiled Triton Bwd (Sorted)
# 0          16.0                0.207264              0.309968                0.306944              0.195808                 0.229504                   0.239104                            0.331200             0.419296                      0.432064
# 1          32.0                0.201664              0.314704                0.315104              0.194688                 0.242848                   0.247392                            0.330704             0.418368                      0.495408
# 2          64.0                0.200064              0.326192                0.321056              0.195104                 0.244352                   0.257968                            0.330272             0.416608                      0.480176
# 3         128.0                0.196592              0.372784                0.375872              0.198720                 0.304224                   0.308160                            0.382640             0.439136                      0.490784
# 4         256.0                0.198800              0.370752                0.371744              0.199088                 0.298480                   0.308736                            0.363072             0.458992                      0.489296
# 5         512.0                0.201200              0.374624                0.374912              0.202624                 0.305264                   0.310816                            0.366752             0.460032                      0.493520
# 6        1024.0                0.200352              0.380720                0.380800              0.200928                 0.307360                   0.317024                            0.369600             0.481936                      0.514528
# 7        2048.0                0.198784              0.384848                0.384800              0.217728                 0.314240                   0.324000                            0.379632             0.511904                      0.533824
# 8        4096.0                0.201184              0.461888                0.463296              0.244832                 0.389120                   0.327360                            0.378368             0.576368                      0.568592
# 9        8192.0                0.208608              0.469664                0.470704              0.274512                 0.431712                   0.378016                            0.401936             0.719168                      0.706544
# 10      16384.0                0.414448              0.649760                0.648288              0.503328                 0.709120                   0.670176                            0.671328             1.018736                      1.012448
# 11      32768.0                0.857632              1.057056                1.058400              0.955264                 1.302864                   1.278832                            1.278608             1.641824                      1.632032
# 12      65536.0                1.795872              1.899600                1.908736              1.880128                 2.505824                   2.548416                            2.556768             3.260160                      3.238848
# 13     131072.0                3.991232              3.975152                4.034272              4.116416                 5.311376                   5.428576                            5.415344             6.175392                      6.168384
# 14     262144.0                7.127200              6.891520                7.004032              7.291200                 9.644160                   9.904816                            9.898720            11.421583                     11.350752
# 15     524288.0               14.576768             13.893024               14.108512             14.765376                19.420416                  20.043840                           19.962336            22.538879                     22.427921
# 16    1048576.0               28.731968             27.119583               27.536352             28.951456                38.458847                  39.523006                           39.525105            43.378334                     43.046688

import torch
import math
import unittest
from monotorch.layers.triton_ops.index_select_op import index_select_triton_atomic_add, index_select_triton_reorder, index_select_triton_reorder_2d,index_select_torch_native_index_add, index_select_torch_native_scatter_add, index_select_torch_compiled, index_select_triton_row_lock, index_select_triton_batch_lock, index_select_triton_sorted, index_select_triton_sorted_compiled
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
                                   dtype=torch.float32, device=device, requires_grad=True)
      indices = torch.tensor([0, 1, 0, 2], device=device)
      output = index_select_triton_atomic_add(unique_values, indices)
      expected_output = torch.tensor([[1, 2, 3], [4, 5, 6], [1, 2, 3], [7, 8, 9]], dtype=torch.float32, device=device)
      grad_output = torch.randn_like(output)
      output.backward(gradient=grad_output)
      self.assertTrue(torch.allclose(output, expected_output))

    def _run_and_check_grad(self, func, unique_values_ref, indices_ref, torch_out_ref, torch_grad_ref, device, N, D, num_indices, label):
        """Helper method to run forward/backward and check gradients."""
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        unique_values = torch.randn(N, D, dtype=torch.float32, device=device, requires_grad=True)
        indices = indices_ref.clone()

        output = func(unique_values, indices)
        output.sum().backward()
        grad = unique_values.grad

        self.assertTrue(torch.allclose(torch_out_ref, output), f"{label} forward mismatch")
        self.assertTrue(torch.allclose(torch_grad_ref, grad, atol=1e-6), f"{label} backward mismatch, torch_grad_ref: {torch_grad_ref}, grad: {grad}")

    def test_index_select_grad(self):
      device = torch.device("cuda:0")
      N, D, num_indices = 55, 68, 111
      torch.manual_seed(42)
      torch.cuda.manual_seed(42)
      unique_values_ref = torch.randn(N, D, dtype=torch.float32, device=device, requires_grad=True)
      indices_ref = torch.randint(low=0, high=N, size=(num_indices,), device=device)
      torch_out_ref = index_select_torch_native_scatter_add(unique_values_ref, indices_ref)
      torch_out_ref.sum().backward()
      torch_grad_ref = unique_values_ref.grad.clone()
      unique_values_ref.grad.zero_()
      self._run_and_check_grad(index_select_triton_atomic_add, unique_values_ref, indices_ref, torch_out_ref, torch_grad_ref, device, N, D, num_indices, "Triton Atomic Add")
      self._run_and_check_grad(index_select_triton_reorder, unique_values_ref, indices_ref, torch_out_ref, torch_grad_ref, device, N, D, num_indices, "Triton Reorder")
      self._run_and_check_grad(index_select_triton_reorder_2d, unique_values_ref, indices_ref, torch_out_ref, torch_grad_ref, device, N, D, num_indices, "Triton Reorder 2D")
      self._run_and_check_grad(index_select_triton_row_lock, unique_values_ref, indices_ref, torch_out_ref, torch_grad_ref, device, N, D, num_indices, "Triton Reorder 2D")
      self._run_and_check_grad(index_select_torch_native_index_add, unique_values_ref, indices_ref, torch_out_ref, torch_grad_ref, device, N, D, num_indices, "Native Index Add")
      self._run_and_check_grad(index_select_torch_native_scatter_add, unique_values_ref, indices_ref, torch_out_ref, torch_grad_ref, device, N, D, num_indices, "Native Scatter Add")
      self._run_and_check_grad(index_select_torch_compiled, unique_values_ref, indices_ref, torch_out_ref, torch_grad_ref, device, N, D, num_indices, "Compiled Scatter Add")
      self._run_and_check_grad(index_select_triton_sorted, unique_values_ref, indices_ref, torch_out_ref, torch_grad_ref, device, N, D, num_indices, "Triton Sorted")
      # self._run_and_check_grad(index_select_triton_batch_lock, unique_values_ref, indices_ref, torch_out_ref, torch_grad_ref, device, N, D, num_indices, "Triton Lock")


def test_batch_index_select_grad(self):
      device = torch.device("cuda:0")
      B, N, D, num_indices = 1025, 96, 129, 2058
      # B, N, D, num_indices = 4, 55, 68, 4096

      torch.manual_seed(42)
      torch.cuda.manual_seed(42)
      unique_values_ref = torch.randn(N, D, dtype=torch.float32, device=device, requires_grad=True)
      indices_ref = torch.randint(low=0, high=N, size=(B, num_indices), device=device)

      expanded_values = unique_values_ref.unsqueeze(0).expand(B, -1, -1)
      expanded_indices = indices_ref.unsqueeze(-1).expand(-1, -1, D)
      torch_out_ref = torch.gather(expanded_values, 1, expanded_indices)
      torch_out_ref.sum().backward()
      torch_grad_ref = unique_values_ref.grad.clone()
      unique_values_ref.grad.zero_() # Zero grad for the reference tensor

      self._run_and_check_grad(index_select_triton_atomic_add, unique_values_ref, indices_ref, torch_out_ref, torch_grad_ref, device, N, D, num_indices, "Triton Atomic Add")
      self._run_and_check_grad(index_select_triton_reorder, unique_values_ref, indices_ref, torch_out_ref, torch_grad_ref, device, N, D, num_indices, "Triton Reorder")
      self._run_and_check_grad(index_select_triton_reorder_2d, unique_values_ref, indices_ref, torch_out_ref, torch_grad_ref, device, N, D, num_indices, "Triton Reorder 2D")
      self._run_and_check_grad(index_select_triton_row_lock, unique_values_ref, indices_ref, torch_out_ref, torch_grad_ref, device, N, D, num_indices, "Triton Reorder 2D")
      self._run_and_check_grad(index_select_torch_native_index_add, unique_values_ref, indices_ref, torch_out_ref, torch_grad_ref, device, N, D, num_indices, "Native Index Add")
      self._run_and_check_grad(index_select_torch_native_scatter_add, unique_values_ref, indices_ref, torch_out_ref, torch_grad_ref, device, N, D, num_indices, "Native Scatter Add")
      self._run_and_check_grad(index_select_torch_compiled, unique_values_ref, indices_ref, torch_out_ref, torch_grad_ref, device, N, D, num_indices, "Compiled Scatter Add")
      self._run_and_check_grad(index_select_triton_sorted, unique_values_ref, indices_ref, torch_out_ref, torch_grad_ref, device, N, D, num_indices, "Triton Sorted")
      # self._run_and_check_grad(index_select_triton_batch_lock, unique_values_ref, indices_ref, torch_out_ref, torch_grad_ref, device, N, D, num_indices, "Triton Lock")




# --- Performance Benchmark ---
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['num_indices'],
        # x_vals=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
        x_vals=[8192, 2*8192, 4*8192, 8*8192, 16*8192, 32*8192, 64*8192, 128*8192],
        # x_vals=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 2*8192, 4*8192, 8*8192, 16*8192, 32*8192, 64*8192, 128*8192],
        line_arg='provider',
        line_vals=['triton', 'triton_reorder', 'triton_reorder_2d', 'triton_row_lock', 'index_add', 'scatter_add', 'compiled_scatter_add', 'triton_sorted', 'triton_sorted_compiled'],
        line_names=[
            'Triton Bwd (atomicAdd)',
            'Triton Bwd (Reorder)',
            'Triton Bwd (Reorder2D)',
            'Triton Bwd (RowLock)',
            'PyTorch Bwd (index_add)',
            'PyTorch Bwd (scatter_add)',
            'Compiled PyTorch Bwd (scatter_add)',
            'Triton Bwd (Sorted)',
            'Compiled Triton Bwd (Sorted)'
        ],
        ylabel='ms',
        # plot_name='index_select-fwd_bwd-performance (D=1024, num_indices <= 4096)',
        plot_name='index_select-fwd_bwd-performance (D=1024, num_indices > 4096)',
        # plot_name='index_select-fwd_bwd-performance (D=1024)',
        args={'D': 1024, 'dtype': torch.float32, 'device': 'cuda'}
    )
)
def benchmark(D, num_indices, provider, dtype, device):
    """
    Benchmark function for index_select forward + backward pass.
    """
    num_unique_value = int(math.sqrt(num_indices) * 2)
    unique_values = torch.randn(num_unique_value, D, dtype=dtype, device=device, requires_grad=True)
    indices = torch.randint(low=0, high=num_unique_value, size=(num_indices,), device=device)
    output_shape = (num_indices, D)
    grad_output = torch.randn(output_shape, dtype=dtype, device=device) # 预先创建梯度, 原因是对于非标量的backward需要知道梯度形状

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
    elif provider == 'index_add':
        func = index_select_torch_native_index_add
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

    def run_op():
        vals_run = unique_values.clone().detach().requires_grad_(True)
        inds_run = indices.clone()
        output = func(vals_run, inds_run)
        output.backward(gradient=grad_output)

    ms, min_ms, max_ms = triton.testing.do_bench(run_op, warmup=25, rep=100, quantiles=[0.5, 0.2, 0.8])
    return ms, min_ms, max_ms

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    benchmark.run(print_data=True, show_plots=True, save_path=".")