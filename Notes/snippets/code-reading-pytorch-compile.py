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
TORCH_LOGS="inductor"


@torch.compile
def square_2(a):
  a = torch.square(a)
  a = torch.square(a)
  return a


### fusion

fused_model = torch.nn.utils.fusion.fuse_linear_bn_eval(model.linear, model.bn)


### mm.py

部分参考 「code-reading-pytorch-quantization.py」



### TorchDynamo DDPOptimizer

DDP works with TorchDynamo. 
When used with TorchDynamo, apply the DDP model wrapper before compiling the model,
such that torchdynamo can apply DDPOptimizer (graph-break optimizations) based on DDP bucket sizes. 

https://pytorch.org/docs/main/notes/ddp.html#torchdynamo-ddpoptimizer

ddp_model = DDP(model, device_ids=[rank])
ddp_model = torch.compile(ddp_model)

### optimizer

optimizer = torch.optim.AdamW(params)

@torch.compile(fullgraph=False)
def compiled_step():
    optimizer.step()


DDP’s performance advantage comes from overlapping allreduce collectives with computations during backwards.
AotAutograd prevents this overlap when used with TorchDynamo for compiling a whole forward and whole backward graph,
because allreduce ops are launched by autograd hooks _after_ the whole optimized backwards computation finishes.

TorchDynamo’s DDPOptimizer helps by breaking the forward graph at the logical boundaries of DDP’s allreduce buckets during backwards.
Note: the goal is to break the graph during backwards,
and the simplest implementation is to break the forward graphs and then call AotAutograd and compilation on each section.
This allows DDP’s allreduce hooks to fire in-between sections of backwards, and schedule communications to overlap with compute.

相关资料：
https://dev-discuss.pytorch.org/t/torchdynamo-update-9-making-ddp-work-with-torchdynamo/860
torch/_dynamo/optimizations/distributed.py


Debug DDPOptimizer：
set TORCH_LOGS=’ddp_graphs’ for full graph dumps.
For logs without graphs, add any of ‘dynamo’, ‘distributed’, or ‘dist_ddp’ to TORCH_LOGS (for basic info about bucket boundaries).
To disable DDPOptimizer, set torch._dynamo.config.optimize_ddp=False. DDP and TorchDynamo should still work correctly without DDPOptimizer, but with performance degradation.

