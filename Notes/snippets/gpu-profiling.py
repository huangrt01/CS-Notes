### Overview

1.pytorch Profiler

2. Nsight systems
- 非侵入
- OS、CUDA API、通信信息，多GPU性能分析支持更完善

3. Nsight Compute
- 优化GPU算子


### GPU Monitoring

# nvitop

https://github.com/XuehaiPan/nvitop

# nvidia-smi

https://gist.github.com/padeoe/771c4972ae185c9a7d3d497fa4e1ecab

alias nvidia-info='nvidia-smi && (nvidia-smi |tr -s " "|grep -Eo "| [0123456789]+ N/A N/A [0-9]{3,} .*"|awk -F" " '\''{system("s=$(cat /proc/"$4"/cmdline| tr \"\\0\" \" \");u=$(ps -o uname= -p "$4");echo "$1"sep"$4"sep$u sep"$7"sep$s" ) }'\''|sed "s/sep/\t/g")'


# 查询频率
nvidia-smi --query-gpu=pstate,clocks.mem,clocks.sm,clocks.gr --format=csv

# clocks.current.memory [MHz], clocks.current.sm [MHz], clocks.current.graphics [MHz]
# 9751 MHz, 1695 MHz, 1695 MHz

# 查询GPU支持的clock组合
nvidia-smi --query-supported-clocks=gpu_name,mem,gr --format=csv

# 设置persistent mode
sudo nvidia-smi -pm 1

# 固定GPU时钟
nvidia-smi -ac 9751,1530 # <memory, graphics>


### time - 1
class TimePytorchFunction:

  def __init__(self, func, *args):
    self.start = torch.cuda.Event(enable_timing=True)
    self.end = torch.cuda.Event(enable_timing=True)
    self.result = None
    self.func = func
    self.args = args

  def __enter__(self):
    # 预热
    for _ in range(5):
      self.func(*self.args)
    self.start.record()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.end.record()
    torch.cuda.synchronize()
    elapsed_time = self.start.elapsed_time(self.end)
    print(f"{self.func.__name__} elapsed time: {elapsed_time} ms")

  def run(self):
    self.result = self.func(*self.args)
    return self.result

b = torch.randn(10000, 10000).cuda()
with TimePytorchFunction(torch.square, b) as timer:
    result = timer.run()

### time - 2

# Timing utilities
import gc, time
start_time = None


def start_timer():
  global start_time
  gc.collect()
  torch.cuda.empty_cache()
  torch.cuda.reset_max_memory_allocated()
  torch.cuda.synchronize()
  start_time = time.perf_counter()


def end_timer_and_print(local_msg):
  torch.cuda.synchronize()
  end_time = time.perf_counter()
  print("\n" + local_msg)
  print("Total execution time = {:.3f} sec".format(end_time - start_time))
  print("Max memory used by tensors = {} MB, reserved = {} MB".format(
      torch.cuda.max_memory_allocated() // 1e6, torch.cuda.max_memory_reserved() // 1e6))

### set seed

def set_seed(seed: int = 37) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # 适用于所有PyTorch后端，包括CPU和所有CUDA设备
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"设置随机数种子为{seed}")

### PyTorch Profiler

model = torchvision.models.resnet18().cuda()
inputs = torch.randn(5, 3, 224, 224, device="cuda")

for _ in range(5):
  model(inputs)

from torch.profiler import profile, record_function, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
  with record_function("model_inference"):
    model(inputs)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=15))
prof.export_chrome_trace("trace.json")


### 显存

https://pytorch.org/memory_viz

import torch

torch.cuda.memory._record_memory_history()
torch.cuda.reset_max_memory_allocated() # 可传入device

with torch.inference_mode():
    shape = [256, 1024, 1024, 1]
    x1 = torch.randn(shape, device="cuda:0")
    x2 = torch.randn(shape, device="cuda:0")

    # Multiplication
    y = x1 * x2

torch.cuda.memory._dump_snapshot("logs/traces/vram_profile_example.pickle")
print("Max memory used by tensors = {} MB, reserved = {} MB".format(
      torch.cuda.max_memory_allocated() // 1e6, torch.cuda.max_memory_reserved() // 1e6))

- 无法profile第三方库比如nccl的显存占用


### 打开chrome trace文件
https://ui.perfetto.dev/
chrome://tracing



### With warmup and skip
# https://pytorch.org/docs/stable/profiler.html

# Non-default profiler schedule allows user to turn profiler on and off
# on different iterations of the training loop;
# trace_handler is called every time a new trace becomes available
def trace_handler(prof):
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=-1))
    prof.export_chrome_trace("/tmp/test_trace_" + str(prof.step_num) + ".json")

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],

    # In this example with wait=1, warmup=1, active=2, repeat=1,
    # profiler will skip the first step/iteration,
    # start warming up on the second, record
    # the third and the forth iterations,
    # after which the trace will become available
    # and on_trace_ready (when set) is called;
    # the cycle repeats starting with the next step

    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=2,
        repeat=1),
    on_trace_ready=trace_handler
    # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
    # used when outputting for tensorboard
    ) as p:
        for iter in range(10):
            torch.square(torch.randn(10000, 10000).cuda())
            # send a signal to the profiler that the next iteration has started
            p.step()

--> void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::pow_tensor_scalar_kernel_impl<float, float>(at::TensorIteratorBase&, float)::{lambda(float)#1}, 



### ncu profiler

# install
https://www.bilibili.com/opus/898996578463776788

wget https://developer.nvidia.com/tools-downloads#?dn=nsight-compute-2025.1.1 
再运行
export PATH=$PATH:/usr/local/NVIDIA-Nsight-Compute-2025.1



# usage

ncu --target-processes all sudo python gpu_op_test.py
ncu --set full -o output $(which python) train.py     # basic模式和full模式

ncu --set full divergence


profiling心得
* active occupancy低，可能是：
1）线程计算任务简单，导致warp的创建和调度开销显著大于计算开销  --> 考虑合并简单线程
2）warp的负载不均衡，不同分支的warp无法同步执行 --> 分析代码
