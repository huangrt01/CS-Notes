## LLM MLSys

## Intro

* Intro
  * 未来硬件，内存互连很关键
* Memory Efficient Attention with Online Softmax (2021) -> FlashAttention in Megatron-LM (2022) 
* Continuous Batching (2022), Paged Attention (2023) -> vLLM, TensorRT-LLM (2023) 
* Speculative Sampling (2023) -> Everywhere in LLM Serving (2023)
* Sequence Parallel (2023) ->  Megatron-LLM (2023) 

## 推理优化

* KV cache
  * LLM模型预测的时候使用的是KV cache的技术，也就是缓存已经推理出的前t-1个token的KV matrix，那么在第t个token开始就无需再计算这部分KV，直接调用缓存的KV就可以。具体而言，整个MHA在casual mask下，可以表示为： $$Logit_{t_h} = \sum_{i \leq t}softmax(\frac{Q_{t_h}K^T_{i_h}}{\sqrt d})V_{i_h}$$,因此预测第t个token的时候，query的multi head（h表示）需要重新计算，以及第t个key和query的multi head（h表示）表示需要重新计算，其余的就可以直接用预测t-1个token缓存的KV进行计算。整体上会大大节省预测时间。附：但是这部分的KV需要占用GPU缓存，而大模型中缓存占用过多，会导致预测的时候Batch size过小，那么整体的预测吞吐率会降低，所以后续很多工作都在对于KV cache做优化。
* Mooncake：将 P / D 分离进行到底 https://zhuanlan.zhihu.com/p/1711346141

#### 模型训练

* [ByteDance] MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs 
  * https://arxiv.org/pdf/2402.15627
* 字节Ckpt https://mp.weixin.qq.com/s/4pIAZqH01Ib_OGGGD9OWQg
  * ByteCheckpoint ，一个 PyTorch 原生，兼容多个训练框架，支持 Checkpoint 的高效读写和自动重新切分的大模型 Checkpointing 系统。

## 推理框架

* MLLM推理
  * SGLang
  * LMDeploy
  * vLLM



## Vision Model 推理

* Swin：microsoft/swinv2-large-patch4-window12-192-22k
  * pytorch基础使用
  * 1*V100
    * Batch_size=4, qps=32： 显存瓶颈 24G/32G，150W/300W
    * Batch_size=8, qps=20： 显存瓶颈 26G/32G，115W/300W
    * Batch_size=2, qps=27： 显存瓶颈 30G/32G，W/300W
  * 注：qps已经考虑了batch_size

* Dinov2
  * Batch_size=4, qps=50: 显存 14G，120W