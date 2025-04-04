pytorch quant的计划和分析：
https://dev-discuss.pytorch.org/t/clarification-of-pytorch-quantization-flow-support-in-pytorch-and-torchao/2809/2


### 也参考code-reading-xxx


### AMP
- Intro: https://towardsdatascience.com/the-mystery-behind-the-pytorch-automatic-mixed-precision-library-d9386e4b787e/

- https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
- https://pytorch.org/docs/stable/notes/amp_examples.html#working-with-multiple-gpus
- https://pytorch.org/docs/stable/amp.html
- https://www.digitalocean.com/community/tutorials/automatic-mixed-precision-using-pytorch
- 源码阅读：https://zhuanlan.zhihu.com/p/348554267
- NVIDIA https://www.cs.toronto.edu/ecosystem/documents/AMP-Tutorial.pdf


- 核心API：torch.autocast and torch.cuda.amp.GradScaler

- 支持的CUDA op：https://pytorch.org/docs/stable/amp.html#autocast-op-reference

 Most matrix multiplication, convolutions, and linear activations are fully covered by the amp.autocast,
 however, for reduction/sum, softmax, and loss calculations, the calculations are still performed in FP32
 as they are more sensitive to data range and precision.

- loss scaling
 
* 如何选择loss scale
 - Choose a value so that its product with the maximum absolute gradient value is below 65,504 (the maximum value representable in FP16)
 - dynamic: https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#scalefactor

scaler = torch.cuda.amp.GradScaler()
# start your training code
# ...
with torch.autocast(device_type="cuda"):
  # training code

# wrapping loss and optimizer
scaler.scale(loss).backward()
scaler.step(optimizer)

scaler.update()

mixed precision training doesn’t really resolve the GPU memory issue if the model weight size is much larger than the data batch.
For one thing, only certain layers of the model is casted into FP16 while the rest are still calculated in FP32;
second, weight update need FP32 copies, which still takes much GPU memory;
third, parameters from optimizers like Adam takes much GPU memory during training and the mixed precision training keeps the optimizer parameters unchanged.


### quantize optimizers DEMO


import torch
from torch import Tensor

lr = 1e-4
momentum = 0.9


def sgd(p: Tensor, grad: Tensor, momentum_buffer: Tensor, lr: float, momentum: float):
  new_momentum_buffer = momentum_buffer * momentum + grad
  new_p = p - lr * new_momentum_buffer

  momentum_buffer.copy_(new_momentum_buffer)
  p.copy_(new_p)


def sgd_bf16(p: Tensor, grad: Tensor, momentum_buffer: Tensor, lr: float, momentum: float):
  # upcast to FP32 for accurate calculations
  new_momentum_buffer = momentum_buffer.float() * momentum + grad
  new_p = p - lr * new_momentum_buffer

  momentum_buffer.copy_(new_momentum_buffer)  # implicit downcast to BF16
  p.copy_(new_p)


def sgd_fp8(p: Tensor, grad: Tensor, momentum_buffer: Tensor, lr: float, momentum: float):
  # upcast to FP32 for accurate calculations
  new_momentum_buffer = momentum_buffer.float() * momentum + grad
  new_p = p - lr * new_momentum_buffer

  momentum_buffer.copy_(new_momentum_buffer)  # implicit downcast to FP8
  p.copy_(new_p)


torch.manual_seed(2024)
p = torch.randn(4, 4)
momentum_buffer = torch.randn(4, 4)

for _ in range(5):
  grad = torch.randn(4, 4)
  sgd(p, grad, momentum_buffer, lr, momentum)
print(p)

torch.manual_seed(2024)
p = torch.randn(4, 4)
momentum_buffer = torch.randn(4, 4).bfloat16()

for _ in range(5):
  grad = torch.randn(4, 4)
  sgd_bf16(p, grad, momentum_buffer, lr, momentum)
print(p)

torch.manual_seed(2024)
p = torch.randn(4, 4)
momentum_buffer = torch.randn(4, 4).to(torch.float8_e4m3fn)

for _ in range(5):
  grad = torch.randn(4, 4)
  sgd_fp8(p, grad, momentum_buffer, lr, momentum)
print(p)

DTYPE = torch.float8_e4m3fn


def quantize_fp8(x: Tensor):
  scale = x.abs().max() / torch.finfo(DTYPE).max
  quantized_x = (x / scale.clip(1e-12)).to(DTYPE)
  return quantized_x, scale


def sgd_scaled_fp8(
    p: Tensor,
    grad: Tensor,
    momentum_buffer: Tensor,
    momentum_buffer_scale: Tensor,
    lr: float,
    momentum: float,
):
  # upcast to FP32 for accurate calculations
  new_momentum_buffer = momentum_buffer.float() * momentum_buffer_scale * momentum + grad
  new_p = p - lr * new_momentum_buffer

  new_momentum_buffer_quantized, new_momentum_buffer_scale = quantize_fp8(new_momentum_buffer)
  momentum_buffer.copy_(new_momentum_buffer_quantized)
  momentum_buffer_scale.copy_(new_momentum_buffer_scale)
  p.copy_(new_p)


torch.manual_seed(2024)
p = torch.randn(4, 4)
grad = torch.randn(4, 4)
momentum_buffer = torch.randn(4, 4)
momentum_buffer, momentum_buffer_scale = quantize_fp8(momentum_buffer)

for _ in range(5):
  grad = torch.randn(4, 4)
  sgd_scaled_fp8(p, grad, momentum_buffer, momentum_buffer_scale, lr, momentum)
print(p)

# INT8 + small fraction

x = torch.tensor(10, dtype=torch.int8)
N = 100
for _ in range(N):
  x.copy_(x + 0.2)

print(f"Expected output: {10 + 0.2 * N}")
print(f"INT8 No SR output: {x}")


def sr(x: torch.Tensor):
  rand = torch.rand(x.shape, device=x.device)  # get a random number [0,1)
  return torch.where(rand < x - x.floor(), x.floor() + 1, x.floor())


x = torch.tensor(10, dtype=torch.int8)
for _ in range(N):
  x.copy_(sr(x.float() + 0.2))

print(f"INT8 SR Actual output: {x}")

x = torch.tensor(200, dtype=torch.bfloat16)
N = 100
for _ in range(N):
  x.copy_(x + 1)

print(f"BF16 Expected output: {200 + N}")
print(f"BF16 No SR output: {x}")


def bf16_sr(x: torch.Tensor):  # x is FP32
  x_i32 = x.view(torch.int32)  # cast to INT32 to do bit manipulation
  x_floor = x_i32 & 0xFFFF0000  # truncate the least significant 16 bits

  # this will generate 31 random bits. we only need 16
  rand = torch.empty(x.shape, dtype=torch.int32).random_()
  rand = rand & 0xFFFF

  out = torch.where(rand < (x_i32 & 0xFFFF), x_floor + 0x10000, x_floor)
  return out.view(torch.float32).bfloat16()


x = torch.tensor(200, dtype=torch.bfloat16)
N = 100
for _ in range(N):
  x.copy_(bf16_sr(x.float() + 1))

print(f"BF16 SR output: {x}")




### torchao - QAT

api: https://github.com/pytorch/ao/tree/main/torchao/quantization/qat
distributed training: https://pytorch.org/blog/quantization-aware-training/#fine-tuning-with-torchtune
- tune run --nproc_per_node 8 qat_distributed --config llama3/8B_qat_full


# Theory

# PTQ: x_q is quantized and cast to int8
# scale and zero point (zp) refer to parameters used to quantize x_float
# qmin and qmax refer to the range of quantized values
x_q = (x_float / scale + zp).round().clamp(qmin, qmax).cast(int8)

# QAT: x_fq is still in float
# Fake quantize simulates the numerics of quantize + dequantize
x_fq = (x_float / scale + zp).round().clamp(qmin, qmax)
x_fq = (x_fq - zp) * scale

# Usage

import torch
from torchtune.models.llama3 import llama3
from torchao.quantization.prototype.qat import Int8DynActInt4WeightQATQuantizer

# Smaller version of llama3 to fit in a single GPU
model = llama3(
    vocab_size=4096,
    num_layers=16,
    num_heads=16,
    num_kv_heads=4,
    embed_dim=2048,
    max_seq_len=2048,
).cuda()

# Quantizer for int8 dynamic per token activations +
# int4 grouped per channel weights, only for linear layers
qat_quantizer = Int8DynActInt4WeightQATQuantizer()

# Insert "fake quantize" operations into linear layers.
# These operations simulate quantization numerics during
# training without performing any dtype casting
model = qat_quantizer.prepare(model)

# Standard training loop
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()
for i in range(10):
    example = torch.randint(0, 4096, (2, 16)).cuda()
    target = torch.randn((2, 16, 4096)).cuda()
    output = model(example)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Convert fake quantize to actual quantize operations
# The quantized model has the exact same structure as the
# quantized model produced in the corresponding PTQ flow
# through `Int8DynActInt4WeightQuantizer`
model = qat_quantizer.convert(model)

# inference or generate