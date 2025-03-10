https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html

### depyf: Debugging Tool

https://zhuanlan.zhihu.com/p/730714110


### torch.compile generate Triton Kernel

# compile_square.py

import torch
compiled_square = torch.compile(torch.square)
x = torch.randn(10, 10).cuda()
result = compiled_square(x)

TORCH_LOGS="output_code" python compile_square.py


@torch.compile
def square_2(a):
  a = torch.square(a)
  a = torch.square(a)
  return a

### with DDP

DDP works with TorchDynamo. 
When used with TorchDynamo, apply the DDP model wrapper before compiling the model,
such that torchdynamo can apply DDPOptimizer (graph-break optimizations) based on DDP bucket sizes. 

https://pytorch.org/docs/main/notes/ddp.html#torchdynamo-ddpoptimizer

ddp_model = DDP(model, device_ids=[rank])
ddp_model = torch.compile(ddp_model)