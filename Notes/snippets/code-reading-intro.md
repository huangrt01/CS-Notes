
### MLSys


#### PyTorch 二次开发

- PyTorch
- https://github.com/pytorch/pytorch

- TorchRec
  - https://pytorch.org/torchrec/overview.html

- SGLang
  - https://github.com/sgl-project/sglang.git



#### Tf 二次开发

- HugeCTR



#### 通信库

- AllReduce
  - https://github.com/baidu-research/baidu-allreduce



#### 量化

- TorchAO
  - https://github.com/pytorch/ao/tree/main
  - Current best way to access all the pytorch native gpu quantization work, Used by sdxl-fast and segment-anything-fast
- GPT-Fast
  - https://github.com/pytorch-labs/gpt-fast
  - (has GPTQ support for int4)
  - Best for weight only quant
- Segment-Anything-Fast
  - https://github.com/pytorch-labs/segment-anything-fast
  - Best for dynamic quant
- AutoAWQ
  - https://github.com/casper-hansen/AutoAWQ


### Model & Algo

#### LLM

* nanoGPT
  * https://github.com/karpathy/nanoGPT

* DeepSeek-V3
  * https://github.com/deepseek-ai/DeepSeek-V3

#### LM

* the Annotated Transformer
  * https://github.com/harvardnlp/annotated-transformer
* OpenNMT
  * https://github.com/OpenNMT/OpenNMT-py.git

### GPU Op

#### Intro

- GPU-Mode
  - https://github.com/gpu-mode/lectures.git


#### Triton

- GemLite
  - https://github.com/mobiusml/gemlite

- MicroBenchmark
  - https://gist.github.com/HDCharles/287ac5e997c7a8cf031004aad0e3a941

- TritonIndex
  - https://github.com/gpu-mode/triton-index

- FlagGems
  - https://github.com/FlagOpen/FlagGems

- FlagAttention
 - https://github.com/FlagOpen/FlagAttention

- SparseAttention
 - https://github.com/epfml/dynamic-sparse-flash-attention 

- SAM flash attention
 - https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L358
 - https://github.com/pytorch-labs/segment-anything-fast/blob/main/segment_anything_fast/flash_4.py#L13


### Data Pipeline

- ConcurrentDataLoader
  - https://github.com/iarai/concurrent-dataloader/tree/master

### C++

#### 通信相关

- Protobuf
  - https://github.com/protocolbuffers/protobuf