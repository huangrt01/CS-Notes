
### mm.py

部分参考 「code-reading-pytorch-quantization.py」



### TorchDynamo DDPOptimizer

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

