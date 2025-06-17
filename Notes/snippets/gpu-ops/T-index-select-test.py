# todo(huangruiteng):
# bf16 benchmark
# fix Triton Bwd (Lock) 在 BLOCK_SIZE <= 64 时的正确性 Bug，但由于其性能问题，暂不Fix
#
# Note:
# 0. Triton Bwd (AtomicAdd) v.s. Triton Bwd (Reorder)
#     - D=256, 临界点50w
#     - D=512, 临界点26w
#     - D=1024，临界点13w
# 1. Triton Bwd (Lock) 性能伴随 BLOCK_SIZE 的增加，严重劣化
# 2. Triton Bwd (Reorder2D) 性能伴随 BLOCK_SIZE 的增加，严重劣化，BLOCK_SIZE_M = 1时最优，增加并行度重要
# 3. Triton Bwd (Sorted) BLOCK_SIZE_UNIQUE=1, BLOCK_SIZE_SEGMENT=128时性能最佳

# index_select-fwd_bwd-performance (unique_ratio=0.02,D=512):
#     num_indices  Triton Bwd (atomicAdd)  Triton Bwd (atomicAdd Relaxed)  Triton Bwd (Reorder)  Triton Bwd (Reorder2D)  Triton Bwd (RowLock)  PyTorch Bwd (index_select)  PyTorch Bwd (scatter_add)  Compiled PyTorch Bwd (scatter_add)  Triton Bwd (Sorted)  Compiled Triton Bwd (Sorted)
# 0          64.0                0.095136                        0.091072              0.220992                0.217456              0.102528                    0.026720                   0.155056                            0.152864             0.320320                      0.321152
# 1         128.0                0.093888                        0.091072              0.215648                0.213920              0.139136                    0.037280                   0.177872                            0.178144             0.365952                      0.365664
# 2         256.0                0.124896                        0.130304              0.247808                0.245376              0.138080                    0.039280                   0.189328                            0.188640             0.369184                      0.370944
# 3         512.0                0.131728                        0.129040              0.263488                0.259728              0.108416                    0.026720                   0.152480                            0.150096             0.336768                      0.434720
# 4        1024.0                0.142688                        0.143392              0.257440                0.257088              0.160672                    0.060832                   0.224512                            0.225152             0.435616                      0.428864
# 5        2048.0                0.145520                        0.144864              0.257616                0.260624              0.164352                    0.062016                   0.225376                            0.224336             0.433536                      0.434144
# 6        4096.0                0.144288                        0.144864              0.333408                0.291104              0.103840                    0.032544                   0.153408                            0.155616             0.354480                      0.355904
# 7        8192.0                0.093168                        0.092800              0.291520                0.295136              0.104896                    0.056544                   0.166464                            0.166384             0.411424                      0.407648
# 8       16384.0                0.092256                        0.093168              0.291904                0.291168              0.105248                    0.104256                   0.199296                            0.198432             0.442304                      0.441200
# 9       32768.0                0.091488                        0.092224              0.291264                0.293024              0.123936                    0.198080                   0.306016                            0.307888             0.514400                      0.512448
# 10      65536.0                0.103648                        0.104160              0.298400                0.299552              0.174096                    0.382080                   0.514096                            0.514592             0.658192                      0.655856
# 11     131072.0                0.169888                        0.132800              0.308160                0.313376              0.284208                    0.748768                   0.927808                            0.927344             0.945696                      0.989824
# 12     262144.0                0.332512                        0.254208              0.333232                0.381600              0.522208                    1.487520                   1.759008                            1.757184             1.557088                      1.553392
# 13     524288.0                0.677248                        0.568384              0.453744                0.562528              1.029936                    2.998448                   3.457248                            3.460576             2.836480                      2.830336
# 14    1048576.0                1.608608                        1.475248              0.685440                0.922720              2.085440                    6.100736                   6.889904                            6.888960             5.348960                      5.343264

# index_select-fwd_bwd-performance (unique_ratio=0.2,D=512):
#     num_indices  Triton Bwd (atomicAdd)  Triton Bwd (atomicAdd Relaxed)  Triton Bwd (Reorder)  Triton Bwd (Reorder2D)  Triton Bwd (RowLock)  PyTorch Bwd (index_select)  PyTorch Bwd (scatter_add)  Compiled PyTorch Bwd (scatter_add)  Triton Bwd (Sorted)  Compiled Triton Bwd (Sorted)
# 0          64.0                0.089568                        0.089536              0.208832                0.209520              0.087872                    0.025536                   0.149920                            0.152352             0.321440                      0.325664
# 1         128.0                0.089632                        0.088096              0.213824                0.210880              0.089040                    0.025440                   0.148384                            0.151296             0.319712                      0.319088
# 2         256.0                0.088624                        0.086960              0.212128                0.210464              0.088992                    0.025312                   0.150560                            0.152176             0.324160                      0.323616
# 3         512.0                0.089408                        0.091648              0.215776                0.216576              0.092544                    0.027712                   0.154688                            0.152416             0.326752                      0.327440
# 4        1024.0                0.094944                        0.096128              0.220928                0.213168              0.090368                    0.026592                   0.152896                            0.150400             0.325088                      0.330848
# 5        2048.0                0.089248                        0.093344              0.216096                0.215392              0.093568                    0.027744                   0.155392                            0.153504             0.331136                      0.330720
# 6        4096.0                0.091472                        0.093632              0.289168                0.289952              0.091216                    0.033088                   0.151200                            0.149120             0.349696                      0.352736
# 7        8192.0                0.090992                        0.088960              0.291296                0.289728              0.088416                    0.057440                   0.161856                            0.162032             0.415488                      0.409072
# 8       16384.0                0.090080                        0.090496              0.295168                0.290416              0.091520                    0.105888                   0.201440                            0.200736             0.457440                      0.451392
# 9       32768.0                0.090336                        0.090544              0.298336                0.295936              0.101920                    0.199968                   0.311536                            0.312736             0.558032                      0.557296
# 10      65536.0                0.102944               p         0.101472              0.298352                0.301248              0.152608                    0.392320                   0.529088                            0.529504             0.839680                      0.846160
# 11     131072.0                0.234144                        0.218928              0.323456                0.381888              0.294080                    0.787008                   0.969312                            0.968960             1.415792                      1.410976
# 12     262144.0                0.511536                        0.488192              0.449984                0.573600              0.578592                    1.575968                   1.847648                            1.846528             2.534080                      2.531648
# 13     524288.0                1.077760                        1.027216              0.727632                1.014592              1.178432                    3.156416                   3.608608                            3.613120             4.805728                      4.798368
# 14    1048576.0                2.215552                        2.114592              1.274144                1.892224              2.473184                    6.320336                   7.135968                            7.149376             9.369776                      9.326368

# index_select-fwd_bwd-performance (unique_ratio=1,D=512):
#     num_indices  Triton Bwd (atomicAdd)  Triton Bwd (atomicAdd Relaxed)  Triton Bwd (Reorder)  Triton Bwd (Reorder2D)  Triton Bwd (RowLock)  PyTorch Bwd (index_select)  PyTorch Bwd (scatter_add)  Compiled PyTorch Bwd (scatter_add)  Triton Bwd (Sorted)  Compiled Triton Bwd (Sorted)
# 0          64.0                0.086224                        0.087040              0.202816                0.200704              0.084704                    0.022752                   0.140976                            0.145984             0.311840                      0.316064
# 1         128.0                0.086544                        0.086400              0.201632                0.203712              0.087360                    0.024928                   0.143104                            0.144864             0.318560                      0.316528
# 2         256.0                0.086816                        0.087840              0.208192                0.202880              0.087552                    0.024288                   0.143584                            0.143136             0.326944                      0.313984
# 3         512.0                0.087392                        0.088192              0.210080                0.210432              0.090752                    0.026944                   0.146912                            0.146464             0.318048                      0.318400
# 4        1024.0                0.089696                        0.089824              0.209312                0.208784              0.090560                    0.025952                   0.160512                            0.147152             0.323456                      0.322688
# 5        2048.0                0.089728                        0.089696              0.210960                0.215456              0.094240                    0.024864                   0.158656                            0.151712             0.324192                      0.320848
# 6        4096.0                0.086880                        0.085728              0.280688                0.285728              0.087344                    0.034432                   0.151760                            0.149232             0.352832                      0.344416
# 7        8192.0                0.088800                        0.089424              0.284032                0.286016              0.089632                    0.059680                   0.156864                            0.157360             0.419520                      0.416832
# 8       16384.0                0.090720                        0.088448              0.283904                0.285504              0.089856                    0.112512                   0.205712                            0.206432             0.569376                      0.567360
# 9       32768.0                0.098304                        0.098352              0.293632                0.310000              0.104896                    0.219360                   0.327520                            0.327744             0.879488                      0.872864
# 10      65536.0                0.166448                        0.160928              0.339888                0.428096              0.186720                    0.430944                   0.559760                            0.561696             1.482048                      1.480576
# 11     131072.0                0.334176                        0.323904              0.475808                0.680928              0.369952                    0.852800                   1.030400                            1.031296             2.690752                      2.690272
# 12     262144.0                0.672992                        0.648672              0.743488                1.164416              0.752704                    1.697056                   1.966512                            1.968448             5.071584                      5.077664
# 13     524288.0                1.349008                        1.300944              1.287696                2.160480              1.521728                    3.384608                   3.842784                            3.837600             9.890144                      9.936384
# 14    1048576.0                2.700160                        2.607440              2.367264                4.152032              3.063712                    6.781216                   7.586800                            7.592096            19.490928                     19.503280


import torch
import math
import unittest
from monotorch.layers.triton_ops.index_select_op import index_select_triton_atomic_add, index_select_triton_atomic_add_relaxed, index_select_triton_reorder, index_select_triton_reorder_2d, index_select_torch_native_scatter_add, index_select_torch_compiled, index_select_triton_row_lock, index_select_triton_batch_lock, index_select_triton_sorted, index_select_triton_sorted_compiled, index_select_torch
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
    self._run_and_check_grad(index_select_triton_atomic_add_relaxed, unique_values_ref,
                             indices_ref, torch_out_ref, torch_grad_ref, device,
                             N, D, num_indices, "Triton Atomic Add Relaxed")
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
    B, N, D, num_indices = 1025, 96, 1290, 2058
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
    self._run_and_check_grad(index_select_triton_atomic_add_relaxed,
                             unique_values_ref, indices_ref, torch_out_ref,
                             torch_grad_ref, device, N, D, num_indices,
                             "Triton Atomic Add Relaxed")
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
        # x_vals=[64, 128, 256, 512, 1024, 2048, 4096, 8192],
        # x_vals=[
        #     2 * 8192, 4 * 8192, 8 * 8192, 16 * 8192, 32 * 8192, 64 * 8192,
        #     128 * 8192
        # ],
        x_vals=[64, 128, 256, 512, 1024, 2048, 4096, 8192, 2*8192, 4*8192, 8*8192, 16*8192, 32*8192, 64*8192, 128*8192],
        line_arg='provider',
        line_vals=[
            'triton_atomicadd', 'triton_atomicadd_relaxed', 'triton_reorder', #'triton_reorder_2d', 'triton_row_lock',
            # 'torch', 'scatter_add', 'compiled_scatter_add', 'triton_sorted',
            # 'triton_sorted_compiled'
        ],
        line_names=[
            'Triton Bwd (atomicAdd)', 'Triton Bwd (atomicAdd Relaxed)','Triton Bwd (Reorder)',
            # 'Triton Bwd (Reorder2D)', 'Triton Bwd (RowLock)',
            # 'PyTorch Bwd (index_select)', 'PyTorch Bwd (scatter_add)',
            # 'Compiled PyTorch Bwd (scatter_add)', 'Triton Bwd (Sorted)',
            # 'Compiled Triton Bwd (Sorted)'
        ],
        ylabel='ms',
        # plot_name='index_select-fwd_bwd-performance (unique_ratio=0.02,D=512,num_indices <= 8192)',
        # plot_name= 'index_select-fwd_bwd-performance (unique_ratio=0.02,D=512,num_indices > 8192)',
        plot_name='index_select-fwd_bwd-performance (unique_ratio=0.02,D=512)',
        args={
            'D': 512,
            'unique_ratio': 0.02,
            'dtype': torch.float32,
            'device': 'cuda'
        }))
def benchmark(D, num_indices, provider, unique_ratio, dtype, device):
  """
    Benchmark function for index_select forward + backward pass.
    """
  num_unique_value = int(num_indices * unique_ratio) or 1
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

  if provider == 'triton_atomicadd':
    func = index_select_triton_atomic_add
  elif provider == 'triton_atomicadd_relaxed':
    func = index_select_triton_atomic_add_relaxed
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
