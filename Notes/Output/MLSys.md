[toc]

plethora of ML frameworks：NCCL, Horovod, BytePS, Mesh-TensorFlow, Gpipe, Ray, HugeCTR, DALI

### Introduction

* 100TB model = 50万亿参数
  * 1万亿=1000B=1T，参数存储用 fp16

* [Google Research: Themes from 2021 and Beyond](https://ai.googleblog.com/2022/01/google-research-themes-from-2021-and.html)
  * Trend 1: More Capable, General-Purpose ML Models
    * CoTrain models: PolyViT https://arxiv.org/abs/2111.12993
    * Pathways: https://blog.google/technology/ai/introducing-pathways-next-generation-ai-architecture/
      * 本质是增强神经网络结构的稀疏性，不仅仅是参数的稀疏性
      * Discussion: https://www.zhihu.com/question/495386434/answer/2199374013
  * Trend 2: Continued Efficiency Improvements for ML
    * Continued Improvements in ML Accelerator Performance
      * TPUv4
      * Device-ML: https://ai.googleblog.com/2021/11/improved-on-device-ml-on-pixel-6-with.html
    * Continued Improvements in ML Compilation and Optimization of ML Workloads
      * XLA: https://www.tensorflow.org/xla
      * https://mangpo.net/papers/xla-autotuning-pact2021.pdf
      * GSPMD: https://ai.googleblog.com/2021/12/general-and-scalable-parallelization.html
    * Human-Creativity–Driven Discovery of More Efficient Model Architectures
      * Transformet、ViT
    * Machine-Driven Discovery of More Efficient Model Architectures
      * NAS -> Primer、EfficientNetV2
      * RL: https://ai.googleblog.com/2020/07/automl-zero-evolving-code-that-learns.html
    * Use of Sparsity
      * Switch Transformer
  * Trend 3: ML Is Becoming More Personally and Communally Beneficial





### 召回

* 索引方式
  * BF (BruteForce): 秒级到分钟级构建，十万到百万量级
  * IVF (Inverted File System): 分钟级到小时级构建，百万到亿级
    * GPU 对聚类进行加速
  * HNSW: 分钟级到天级构建，千万到百亿级实时性
    * 可能会 sharding

* 量化方式
  * Int8 
  * PQ

### 粗排

#### COLD : Towards the Next Generation of Pre-Ranking System

阿里定向广告最新突破：面向下一代的粗排排序系统COLD - 萧瑟的文章 - 知乎
https://zhuanlan.zhihu.com/p/186320100

Computing power cost-aware Online and Lightweight Deep pre-ranking system

小精排支持复杂算法探索

* SENet: 模型训练时获取特征重要性数据
* 并行化：在取PS之后做模型并行预估，能比论文中的实现更高效
* 列存：全链路列存
* fp16
  * mix precision: fp32 BN + fp16 fully-connected layers
  * parameter-free normalization

#### Towards a Better Tradeoff between Effectiveness and Efficiency in Pre-Ranking: A Learnable Feature Selection based Approach, SIGIR 2021

Feature Selection method based on feature Complexity and variational Dropout (FSCD)

2.1 FSCD for Pre-Ranking Model，核心思想是 variational dropout

* 特征选取概率与特征复杂度负相关
* 特征复杂度的因素：特征类型、embedding size、key size (能类比于候选数量)
* 数学手段：
  * 公式(3)，损失函数，参数z的正则化
    * 特征复杂度越大，正则项系数越大
    * 损失函数及正则化系数的推导（见附录）
  * 公式(5)，Bernoulli 分布的平滑化

2.2 Fine-Tuning the Pre-Ranking Model

* 用精排模型参数来初始化参数，fine-tune 加速训练
*  <img src="https://www.zhihu.com/equation?tex=%5Cgamma_3%3D10%5E%7B-7%7D" alt="\gamma_3=10^{-7}" class="ee_img tr_noresize" eeimg="1">  描述候选数量，也是一个衡量特征复杂度的参数



### Go+Torch

https://github.com/wangkuiyi/gotorch

Q: TensorFlow为什么需要引入图这个概念？

A: 

1.backward自动求导，需要定义前向的数据结构

2.python执行速度慢，决定执行效率的是图的解释器。图是python代码的另一种表示形式，开始包括前向计算过程，通过调用TensorFlow API，加入其它op包括反向计算过程和模型更新过程。构造图本质上是在编译。

* [TFRT](https://github.com/tensorflow/runtime)

调用libtorch内部的native function类比tf的op，但native function是函数，而不是一个class，每一个function可以用HLO（一种古老的适用于数值计算的语言）写一遍。gotorch调libtorch调pytorch XLA里的HLO程序，翻译成特定设备优化的代码

* native function有YAML描述，可自动生成Go Wrapper
* torchscripts：用到的python语法的子集 => python高层api可翻译成torchscripts再翻译

如果 Go+Torch 在未来一年里孕育成熟，有望优化以下核心 应用场景:

1. 统一训练和预测系统(目前训练用 Python 写，预测用 C++)
2. 统一云和端系统(目前云上用 TensorFlow，端上比如 xNN 调用 TensorFlow Lite)
3. 统一训练和预测时的数据处理流程(目前需要用 Python和C++分别做两套，开销大，而且容易出错)
4. 统一搜索、推荐、广告、金融核心、移动智能和端智能、无人驾驶等多个领域的基础架构
5. 能支持新的机器学习模式——online learning、GAN、reinforcement learning、imitation learning等。

### OneFlow: 大规模分布式深度学习框架

数据并行：allreduce + PS

模型并行：参数如何划分？复杂的通信模式

![platforms](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/MLSys/platforms.jpg)

横向拓展：片间高速互联，e.g. TPU

纵向拓展：单个芯片从通用到专用



静态调度与流式执行系统![layers](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/MLSys/layers.jpg)



OneFlow架构

* actor及流水线
  * 内存槽，用类似rust的ownership解决内存冲突问题，ownership随状态转移

![memory-pipeline](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/MLSys/memory-pipeline.jpg)

* node placement: consistent view
  * SBP, 在op层面实现数据和模型并行 
  ![SBP](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/MLSys/SBP.jpg)


### GPU 相关知识

共享卡 —— 如何实现算力和显存隔离
* 隔离方式：时间片 vs 空间
* 隔离级别：不隔离 vs 强隔离 vs 弹性
* 几种隔离技术对比：
  * vGPU(Grid)(Nvidia)：虚拟化；容器化支持不好，license
  * vCuda(腾讯)：cuda hook；性能损耗严重
  * cGPU(Alibaba)：ioctl；损耗小，硬隔离，侵入内核（机器容易坏）
  * MPS(Nvidia)：thread；显存隔离，故障隔离不好
  * MIG(~A100)：sm/global memory；硬件层面隔离

### MLOps

* 磁盘U形故障率 ~ GPU故障率建模



### AWS - SageMaker

#### Sagemaker Immersion Labs

https://sagemaker-immersionday.workshop.aws/

[Github Link](https://github.com/aws-samples/amazon-sagemaker-immersion-day)

* Lab 1. Feature Engineering
* Lab 2. Train, Tune and Deploy XGBoost
  * Hyperparameter tuner
* Lab 3. Bring your own model
  * Bring your own container
  * Bring your own script
* Lab 4. Autopilot, Debugger and Model Monitor
  * Autopilot: generates notebooks for you
  * debug hooks that listen to events
  * class activation maps with SageMaker Debugger
* Lab 5. Bias and Explainability
* Lab 6. SageMaker Pipelines

总体分析

* python SDK, IDE 式的开发体验
* instance per notebook
  * brings elastic, dedicated compute for each person, project, dataset, step in your ML lifecycle
* Train a model
  * Build-in algorithms
  * Script mode
  * Docker container
  * AWS ML marketplace
  * Notebook instance
* use_spot_instances=True

#### SageMaker Debugger

《AMAZON SAGEMAKER DEBUGGER: A SYSTEM FOR REAL-TIME INSIGHTS INTO MACHINE LEARNING MODEL TRAINING》

https://github.com/awslabs/sagemaker-debugger#run-debugger-locally

* 痛点：训练过程长、不透明（训练进程、底层资源情况）
  * e.g. 遇到过拟合，终止训练任务的机制
* 关键特性
  * 数据采集：零代码修改；持久化存储
  * 自动数据检测：检测训练过程、系统瓶颈；提前终止；自定义规则；与 Cloudwatch 事件集成
  * 实时监控：指标调试；通过训练的 step 或时间间隔进行资源利用率分析
    * 算法层面：特征重要性、layer weight/gradient 信息展现、帮助理解 serving/training 一致性 (data drift)
  * 节省时间和成本：原型验证；资源
  * 集成在 Studio 环境中
* 实现
  * 训练容器 ---> 存储 ---> Debugger 容器 ---> actions
    * Actions: [cloudwatch](https://aws.amazon.com/cn/cloudwatch/) + [lambda function](https://aws.amazon.com/cn/lambda/)
  * [smdebug](https://pypi.org/project/smdebug/#description)
  * Profiling
    * 原生框架分析：可能增加 GPU 内存消耗
    * 数据加载分析：调试器收集 DataLoader 事件信息，可能影响训练性能
    * python：cProfile (python operator), Pyinstrument (隔段时间记录堆栈情况)
  * Debugger Hook: 类似 Tensorflow 的 hook，传入 callback 对象，采集指标到存储
  * Rule 集成在 Hook 中: 系统层、模型层（过拟合、梯度消失）

```python
# record tensors
import smdebug.tensorflow as smd
hook = smd.KerasHook("/opt/ml/tensors")
model.fit(x, y, epochs=10, callbacks=[hook])

custom_collection=CollectionConfig(
	name="relu_ouput",
	parameters={
		"include_regex": ".*relu_output",
		"save_interval": "500"
  }
)

# access tensors
from smdebug.trials import create_trial
trial = create_trial("/opt/ml/tensors")
trial.tensor_names(regex=".*")
trial.tensor("conv0").value(step)

# monitor tensors
while not trial.loaded_all_steps:
  steps = trial.steps(mode=modes.EVAL)
	t = trial.tensor("conv1").value(steps[-1])
	plt.hist(t.flatten(), bins=100)
  
# analyze tensors
labels = "CrossEntropyLoss_input_0"
predictions = "CrossEntropyLoss_input_1"
inputs = "ResNet_input_0"
for step in trial.steps():
  l = trial.tensor(labels).value(step)
	p = trial.tensor(predictions).value(step)
	i = trial.tensor(inputs).value(step)
for prediction, label, img in zip(p,l,i):
	if prediction != label:
		plt.imshow(img)
```

* Challenges
  * Scale rule analysis by offloading into separate containers
  * Reduce overhead when recording and fetching tensors
    * optimize data retrieval with the help of index files that store metadata such as name, shape, and step along with the location of tensor objects
  * Separate compute and storage and minimize impact on training
* Rules
  * datasets
  * activation functions: sigmoid's vanishing gradients, dead ReLU
  * poor initialization: 随机 weights 是保证 independently learn

![debug-rule](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/MLSys/debug-rule.png)

* Deployment Results and Insights
  * latent space + [t-SNE]()
  * Using Debugger for iterative model pruning
    * Many types of pruning techniques are known, for example, structured versus unstructured prun-ing, randomly removing weights versus removing by size or rank, and iterative pruning versus one-shot pruning (Blalock et al., 2018). In case of CNNs, iterative filter pruning is known to achieve state of the art results



### Other MLSys

* [介绍 Facebook 推荐系统的文章](https://engineering.fb.com/2021/01/26/ml-applications/news-feed-ranking/)
  * 多目标 MMoE、分数融合
  * 相比抖音，社交属性更强
    * unread bumping logic: 未来得及展现的信息
    * action-bumping logic: 交互过的信息有再次交互
  * serving 流程
    * integrity processes
    * pass 0: lightweight model 选出 500 条
    * pass 1: 精排 500 条
      * 《Observational data for heterogeneous treatment effects with application to recommender systems》
      * People with higher correlation gain more value from that specific event, as long as we make this method incremental and control for potential confounding variables.
    * pass 2: 混排，contextual features, such as content-type diversity rules




### ToB

[Palantir](https://www.palantir.com/)

[C3.ai](https://c3.ai/#null)

[一篇分析 ToB 与 ToC 技术上区别的文章](https://zhuanlan.zhihu.com/p/341358485)





### 论文阅读

已读，待整理：

#### MLSys: The New Frontier of Machine Learning Systems

#### Deep Neural Networks for Youtube Recommendations, RecSys 16

#### Wide & Deep learning for Recommender Systems, RecSys 17

1.Introduction
* Wide ~ Memorization: 模型直接学习并利用历史数据中物品或者特征的“共现频率”的能力
* Deep ~ Generalization: 模型传递特征的相关性，以及发掘稀疏甚至从未出现过的稀有特征与最终标签相关性的能力
* Generalized linear models with nonlinear feature transformations
* cross-product transformations: 特征工程的概念，交叉积变换，缺点是无法 generalize 没出现过的 query-item feature pairs

问题：输入是稀疏高秩矩阵，缺少 interactions，难以利用它学到合适的低维 embedding

3.WIDE & DEEP Learning

3.1 The Wide Component

利用 cross-product transformation 提供多个特征的非线性

<->  对比：[Deep Neural Networks for YouTube Recommendations ] 用平方 和 平方根项 提供单个特征的非线性

3.3 Joint Training of Wide & Deep Model

注意辨析 joint training 和 ensemble 的区别
* 前者是共同训练，后者不是
* 后者模型可以更大
* 前者，Wide 只需要给 Deep 补少量 cross-product feature transformations

4.System Implementation

4.2 Model Training

* warm-starting system
* dry run
* sanity check

Appendix

* 概念：AUC：ROC曲线下方的面积，ROC横坐标FPR，纵坐标TPR
* 资源：
  * 这个大佬的专栏很实用，讲解tensorflow和推荐系统，https://zhuanlan.zhihu.com/learningdeep
* 思考：可否联系到IRLS方法，最优化稀疏矩阵的秩，用一个类似的矩阵学习秩的表示



**改进：Deep&Cross模型 (DCN)**

* 多层交叉层:  <img src="https://www.zhihu.com/equation?tex=x_%7Bl%2B1%7D%3Dx_0x_l%5ETw_l%2Bb_l%2Bx_l" alt="x_{l+1}=x_0x_l^Tw_l+b_l+x_l" class="ee_img tr_noresize" eeimg="1">  
  * 参数引入较为克制，增强模型的非线性学习能力
  * 解决了Wide&Deep模型人工组合特征的问题

#### A Hitchhiker's Guide On Distributed Training Of Deep Neural Networks, JPDC 18

#### TFX: A TensorFlow-based production-scale machine learning platform

#### TensorFlow: A system for large-scale machine learning, OSDI 16

#### Clipper: A Low-Latency Online Prediction Serving System, NSDI 17

low latencies, high throughputs, and improved accuracy

prediction cache, batching queue

##### Model abstraction layer

用object store存模型，减少初始化开销

prediction cache：本质上类似SHARED属性（同一batch内的某一特征用相同的预估结果）。两者的区别在于，前者的输入更简单，以模型和req id为标识，易于做cache操作；后者是feature层面，更精细。推荐系统入图的特征输入很难做到完全一致，因此做prediction cache操作难度较大。

batching：动态选batch size的方式
* additive-increase-multiplicative-decrease (AIMD) scheme 
* quantile regression
* delayed batching：按攒batch的timeout来delay，适合并行优化明显的模型

model container: 无状态服务
* Clipper performs adaptive batching independently for each replica

##### Model selection layer

动态调整选用模型的策略，推荐系统采用这类技术比CV/NLP难度更大

* Single Model Selection Policy
  * address the trade-off between exploring possible actions and exploiting the estimated best action. 
* Ensemble Model Selection Policies
  * Robust Prediction 
    * agreement衡量prediction confidence 
    * 有针对degraded模型的降级机制
  * Straggler Mitigation
* Contextualization: instantiate a unique model selection state for each user, context, or session.



#### Hidden Technical Debt in Machine Learning Systems, NIPS 15

boundary erosion, entanglement, hidden feedback loops, undeclared consumers, data dependencies, configuration issues, changes in the external world, and a variety of system-level anti-patterns.

2. Complex Models Erode Boundaries
* Entanglement: 即使多模型/超参的配置独立，效果也会互相影响
* Correction Cascade: 模型级联是 hidden debt
* Undeclared Consumers: 需要SLA (service-level agreement)

3. Data Dependencies Cost More than Code Dependencies
* Underutilized dependencies: legacy/bundled/epsilon/correlated, use exhaustive leave-one-feature-out evaluations to detect

4. Feedback Loops
* direct: related to bandit algorithms, costly
* hidden: two independent systems may interact

5. ML-System Anti-Patterns
* Glue Code: hard to achieve a domain-specific goal
* Pipeline Jungle: 特征工程的意义所在，thinking holistically about data collection and feature extraction
* Dead Experimental Codepaths
* Abstraction Debt
* Common Smells

6. Configuration Debts
* Feature A was incorrectly logged from 9/14 to 9/17
* Feature B is not available on data before 10/7
* The code used to compute feature C has to change for data before and after 11/1 because of changes to the logging format
* Feature D is not available in production, so a substitute features D′ and D′′ must be used when querying the model in a live setting
* If feature Z is used, then jobs for training must be given extra memory due to lookup tables or they will train inefficient
* Feature Q precludes the use of feature R because of latency constraints.

7. Dealing with Changes in the External World




#### Ad Click Prediction: a View from the Trenches, KDD 13
2. Brief System Overview：Google 场景是搜索广告

3.Online Learning and Sparsity

* FTRL-Proximal(Follow The Proximally Regularized Leader): get both the sparsity provided by RDA and the improved accuracy of OGD

* [在线学习（Online Learning）导读 - 吴海波的文章](https://zhuanlan.zhihu.com/p/36410780)
* FTRL的数学本质：SGD（梯度 + L2）+稀疏性（L1）

* 李亦锬大佬的机器学习答题集，很精彩，其中介绍了 FTRL 的实践意义
  https://zhuanlan.zhihu.com/p/20693546

4.Saving Memory at Massive Scale

进一步节省PS内存的方式

* Probabilistic Feature Inclusion
  * 出于效果、回溯性的考量，只考虑在 serving 时省内存
  * Poisson Inclusion, Bloom Filter Inclusion
* Encoding Values with Fewer Bits
  *  <img src="https://www.zhihu.com/equation?tex=%5Comega_%7Bi%2Crounded%7D%3D2%5E%7B-13%7D%5Clfloor%7B2%5E%7B13%7D%5Comega_i%2BR%7D%5Crfloor" alt="\omega_{i,rounded}=2^{-13}\lfloor{2^{13}\omega_i+R}\rfloor" class="ee_img tr_noresize" eeimg="1"> 
* Training Many Similar Models
  * savings from not representing the key and the counts per model
* A Single Value Structure
  * 动机是省内存，本质上有点像是对极其相似的 models 的公共参数做 cotrain
  * 用于特征淘汰、特征选择等实验场景 (Fast prediction of new feature utility. ICML, 2012)
* Computing Learning Rates with Counts

* Subsampling Training Data: 然后给负样本的 loss 增加权重，保证“期望上”目标函数的一致性

5.Evaluating Model Performance

* Progressive Validation: online loss, relative changes

6.Confidence Estimates

* 定义并估出了不确定度的 upper bound: 学习率向量点乘输入向量

7.Calibrating Predictions

* 有 Poisson regression、isotonic regression 等手段
* 系统的 inherent feedback loop 不保证理论准确性

8.Automated Feature Management

* 特征平台化

9.Unsuccessful Experiments

* Feature Hashing, Dropout, Feature Bagging, Feature Vector Normalization



机器学习框架易用性

* a high-dimensional visualization tool was used to allow researchers to quickly see effects across many dimensions and slicings
* enables data sources and features to be annotated. Automated checks can then be run to ensure that all dependencies have the appropriate annotations, and dependency trees can be fully resolved.



#### XDL: An industrial deep learning framework for high-dimensional sparse data, KDD 19

MPI (All Reduce) 和 PS，两种分布式计算的发展方向

Sparse + Dense

* SparseNet: Representa-tion learning which captures information from high-dimensional sparse input and embeds them into a low-dimensional space

* DenseNet: Function fitting which models the relationship between dense em- bedding representation and supervised label

In order to facilitate deployment on various computing platforms,
XDL can be scheduled by multiple resource management platform, like Yarn, and provides data I/O interfaces to various data storage systems, like HDFS and Kafka.

* I/O
  * Hierarchical sample compression: prefix tree

![prefix-tree](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/MLSys/prefix-tree.png)

* Workflow pipeline

  * I/O: read sample and group mini-batch -> prefetch (maybe cudaMemcpyAsync()) -> pull/forward/backward/push
  * SparseNet and DenseNet

* Optimization for Advanced Model Server

  * Network: [Seastar](https://github.com/scylladb/seastar) + zero-copy/CPU-binding

* Online Learning with XDL

  * Feature Entry Filter
  * Incremental Model Export
  * Feature Expire

#### Ethane: Taking control of the enterprise, SIGCOMM 2007

make networks more manageable and more secure，一种思路是全方位的增加控制，相当于新增一层，只是 hide 了复杂度；于是提出 ethane 解决这一问题

ethane的思想：
* The network should be governed by policies declared over high-
level names
* Policy should determine the path that packets follow
* The network should enforce a strong binding between a packet
and its origin.

Ethane的优势：
* Security follows management.

* Incremental deployability.

* Significant deployment experience.
  
  
#### Scaling distributed machine learning with the parameter server, OSDI 2014

PS架构的优势主要还是高可用(system efficiency)

2.2
* distributed subgradient descent

3.6 User-defined Filters
* signifi-cantly modified filter
* KKT(见5.1)：特征重要性筛选

4.Implementation

4.2 Messages
* key-caching and value-compression can be used jointly.
* key-cache让sender只需要传key lists的hash
* 用snappy压缩 zero value

4.3 Consistent Hashing
一致性hash和 key-range 的概念紧密相连，论文 Chord: A scalable peer-to-peer lookup protocol for Internet applications

4.5 Server Management
* 计算节点分为server node和worker node
* server共同维持全局共享的模型参数
* workers保留一部分的训练数据，并且执行计算
* worker只和server有通信，互相之间没有通信

examples
* CountMin Sketch Algo 有点像 bloom filter

PS运维：
* expectation - current_situation = operations
* 服务发现、数据发现

性能优化：
* 双buffer + RCU，读不被锁阻碍
* 简化版读写锁，优化系统态开销

#### Serving DNNs like Clockwork: Performance Predictability from the BottomUp, OSDI 2020

[presentation](https://www.usenix.org/conference/osdi20/presentation/gujarati) 挺有意思

model serving: ML system's "narrow waist"

这篇文章尝试解决服务化请求长尾问题

首先分析产生长尾的原因：out-of-order scheduling, interference from concurrency, power saving modes, and network queuing delays.
然后基于以下两个假设：
1) “DNN inference is predictable.”
2) 能限制系统到应用层面的决策能力（减少worker内部的并行）

提出解决方案：
分布式系统常用的思路，request打到worker之前，先过一个中心controller，中心controller掌握全局信息（模型是否load、worker是否pending等），预测latency是否会超过SLA，以决定将请求打到哪个worker

感觉这一系统难以直接应用于大公司的场景，因为：

1.需要和rpc框架做更深的结合

* 长尾问题本身有一部分是来自于服务化带来的网络传输开销，比如thrift worker负担，只有rpc框架能掌握更多信息
* 如果要落地到生产场景，自制的简陋 controller 不易推广

2.自身的优势不明显

* 分业务服务化部署、并且是online learning的场景，显存不是瓶颈，模型本身已经是preload了
* scalable能力未经过验证 (6.6)，controller成为瓶颈

有启发的地方
* 框架内的page cache可以借鉴一下 (https://gitlab.mpi-sws.org/cld/ml/clockwork/-/blob/master/src/clockwork/cache.h)

#### The Hardware Lottery, 2020

https://hardwarelottery.github.io/

* hardware, software and ML research communities evolve in isolation
  * Our own intelligence is both algorithm and machine.
  * Moore's Law ---> The predictable increases in compute and memory every two years meant hardware design became risk-averse.
  * machine learning researchers rationally began to treat hardware as a sunk cost to work around rather than something fluid that could be shaped
* The Hardware Lottery
  * "Happy families are all alike, every unhappy family is unhappy in it’s own way." (Tolstoy & Bartlett, 2016)
  * e.g. Babbage 的构想直到二战 electronic vacuum tubes 的使用才成为现实。"being too early is the same as being wrong."
  * von Neumann Bottleneck — the available compute is restricted by “the lone channel between the CPU and memory along which data has to travel sequentially” (Time, 1985).
  * GPU 并行能力 ---> 高 FLOPS ---> 能做矩阵乘 ---> 训得动深度神经网络
* The Persistence of the Hardware Lottery
  * sparsity ~ Ampere Architecture
  * 较为安全的硬件优化方向：matrix multiplication, unstructured sparsity, weight specific quantization
  * the difficulty of trying to train a new type of image classification architecture called capsule networks on domain specialized hardware
* The Likelyhood of Future Hardware Lotteries
  * how much future algorithms will differ from models like deep neural networks?
    * 许多子领域，参数量对效果提升的边际效应在下降（近似对数关系）
    * 100TB model (fp16) ~ 50T ~ 50万亿参数
    * Our own intelligence relies on decentralized local updates which surface a global signal in ways that are still not well understood
* The Way Forward
  * Producing a next generation chip typically costs $30-80 million dollars and takes 2-3 years to develop
  * A software evolution
    * one way is to focus on the development of domain-specific languages which cater to a narrow domain.
    * another way is to automatically auto-tune the algorithmic parameters of a program based upon the downstream choice of hardware.
* 另一篇强调 General Method + 算力 大力出奇迹的 blog: http://www.incompleteideas.net/IncIdeas/BitterLesson.html



#### Deep Learning Inference in Facebook Data Centers: Characterization, Performance Optimizations and Hardware Implications

1. Introduction

* general requirements for new DL hardware designs:
  * High memory bandwidth and capacity for embeddings 
  * Support for powerful matrix and vector engines 
  * Large on-chip memory for inference with small batches 
  * Support for half-precision floating-point computation

2. Characterization of DL Inference

* Ranking and Recommendation
  * embedding lookup 硬件层面分析
    * 特点是 low spatial locality，caching 难度高
    * High-bandwidth memory (HBM): 性能高，容量不够
    * Non-volatile memory (NVM): bandwidth低 不可行、成本低
* CV: 图像识别、目标检测、视频理解
  * number of operations per weight 高
  * number of operations per activation 不高
* NLP: NMT(Neural machine translation) uses seq2seq
  * parallelism: 针对 RNN-based approaches 的并行化方案，比如 stacked conv, transformer

* computation kernels 分析
  * 数据中心成本：fc > embedding lookup > tensor manipulation > conv
  * fc layer 分析：图内第一层运算抽象成矩阵乘（当batch size M 较小时，BLAS3 趋近于 BLAS2，matrix multiplication engine 效果削弱）
    * When an M×K matrix activation matrix is multiplied with a K×N weight matrix, we compute 2MKN operations while reading KN weights, leading to 2M operations per weight.
    * Similarly, the number of operations per activation is 2N.

3. Performance Optimizations

* bf16 sum pooling 是优化方向
* intel int8 multiplication with 16-bit accumulation 提升一倍吞吐

* FBGEMM, an algebra engine
  * outlier-aware quantization:  <img src="https://www.zhihu.com/equation?tex=W%20%3D%20W_%7Bmain%7D%2BW_%7Boutlier%7D" alt="W = W_{main}+W_{outlier}" class="ee_img tr_noresize" eeimg="1"> 
    * outlier uses 32-bit accumulation. We find that Woutlier becomes a sparse matrix, often with density less than 0.1%, especially when combined with sym-metric quantization [39].

* accuracy challenges
  * Fine-grain Quantization
  * Quantization-aware Training
  * Selective Quantization
  * Outlier-aware Quantization: 更精细、更窄地选取 quantize range
  * Net-aware Quantization: if an operator is only followed by ReLU, we can narrow down the range by excluding negative values

* HPC challenges
  * HPC 习惯 “pack” a block of input matrices into a format friendly for vectorization and cache locality, 但对于DL领域 tall-skinny matrices，pack 会带来 overhead
  * DL不完全是矩阵乘：比如conv op，转化为矩阵乘需要提前做 `im2col` 操作，有 overhead，因此需要专门做 kernel fusion 提供 conv interface
    * This will also enable algorithmic optimizations such as Winograd or FFT-based convolution as in cuDNN with automatic choice of the best algorithm for given tensor shapes.
    * reduced-precision 计算也需要专门的 fusion，一些库未能满足

```c++
template<typename T_PACK_A, typename T_PACK_B, typename T_C, typename OUT_FUNCTOR>
void gemmPacked(
  // packed inputs
  T_PACK_A& packA, T_PACK_B& packedB,
  // output
  T_C* C, uint32_t ldc,
  // post-processing functor, e.g. Relu
  OUT_FUNCTOR& outProcess);
```

* The packing of matrix A can be spe-cialized and fused with memory bandwidth bound operations such as `im2col`, row-wise sum for asymmetric quantization, or depth-wise convolution.

* whole graph optimization
  * 手动 fusion 仍有必要

4. Application Driven HW Co-design Directions

* Recommendation models not only require a huge memory capacity but also high bandwidth.
* 优化的副作用：比如 avx512 降频，见 Computer-Architecture.md
* 增加 tiers 的 trade-offs：传输、压缩解压开销，a hypothetical accelerator with 100 TOP/s compute throughput would require a few GB/s PCIe and/or network bandwidth

5. Related Work

* matrix-vector engine、FPGA、TPU

* ML benchmark



#### Practical Lessons from Predicting Clicks on Ads at Facebook, KDD 2014

2.指标

* Normalized Entropy: the average log loss per impression divided by what the average log loss per impression would be if a model predicted the background click through rate (CTR) for every impression. 
  * 用 background CTR 给 loss 做 normalize
* RIG (Relative Information Gain) = 1 - NE
* Calibration: the ratio of the average estimated CTR and empirical CTR
* AUC(Area-Under-ROC): 衡量排序， 忽略低估与高估

3.Prediction Model Structure

* BOPR (Bayesian online learning scheme for probit regression): 假定高斯分布，在线学习分布的参数
  * Both SGD-based LR and BOPR described above are stream learners as they adapt to training data one by one.
  * BOPR 相比 SGD-based LR 的区别在于，梯度下降的 step-size 由 belief uncertainty  <img src="https://www.zhihu.com/equation?tex=%5Csigma" alt="\sigma" class="ee_img tr_noresize" eeimg="1">  控制，也是在线更新的参数3
* 3.1 Decision tree feature transforms
  * bin the feature
  * build tuple input features
    *  joint binning, using for example a k-d tree
    * boosted decision trees
  * follow the Gradient Boosting Machine (GBM) [5], where the classic L2-TreeBoost algorithm is used
  * We can understand boosted decision tree based transformation as a supervised feature encoding that converts a real-valued vector into a compact binary-valued vector.

* 3.2 Data freshness
  * The boosted decision trees can be trained daily or every couple of days, but the linear classifier can be trained in near real-time by using some flavor of online learning.

* Experiment result for different learning rate schmeas for LR with SGD
  * NE: per weight > global > constant > per weight sqrt > per coordinate

* BOPR 与 LR 对比
  * LR's model size is half
  * BOPR provides a full predictive distribution over the probability of click. This can be used to compute percentiles of the predictive distribution, which can be used for explore/exploit learning schemes

4.Online Data Joiner

* length of waiting time window: 定义"no click"，需要 tune
  * 过长会增加buffer、影响"recency"
  * 过短会影响"click coverage" => the empirical CTR that is somewhat lower than the ground truth

* 数据结构：HashQueue
  * consisting of a First-In-First-Out queue as a buffer window and a hash map for fast random access to label impressions.
  * operations: enqueue, dequeue, lookup

* Anomaly detection mechanisms
  * 监测到数据剧烈变化，断流训练器

5.Containing Memory and Latency

* number of boosting trees: 500个比较折中
* boosting feature importance
  * the cumulative loss reduction attributable to a feature
  * 对多个 trees 的 reduction 相加
* features
  * contextual features: local time of day, day of week, device, current page
  * historical features: ctr of the ad in lask week, avg ctr of the user
  * historical features 明显比 contextual features 重要
  * contextual features 更需要 data freshness

6.Coping with Massive Training Data

* Uniform subsampling: sample rate 10% 

* Negative down sampling: sample rate 2.5%

* Model Re-Calibration:  <img src="https://www.zhihu.com/equation?tex=q%3D%5Cfrac%7Bp%7D%7Bp%2B%5Cfrac%7B1-p%7D%7Bw%7D%7D" alt="q=\frac{p}{p+\frac{1-p}{w}}" class="ee_img tr_noresize" eeimg="1"> 

#### DCAF: A Dynamic Computation Allocation Framework for Online Serving System, DLP-KDD 2020

* 加强 召回、粗排、精排 的联动，向统一分配算力的方向发展
* We formulate this resource allocation problem as a knapsack problem and propose a Dynamic Computation Allocation Framework (DCAF).

* 基于背包问题的机制，有限资源最大收益
  * 理论：https://en.wikipedia.org/wiki/Duality_(optimization)，凸优化，证明了在现实算力约束的条件下（有两个直觉的前提），用二分来找 global optimal lambda 即可获取最优解
    * construct the Lagrangian

* 系统有 control 能力，能动态响应流量波动
  * 理论：https://en.wikipedia.org/wiki/PID_controller

* Online Decision Maker
* Information Collection and Monitoring
* lambda 离线计算，Qij 在线预估
* Request Value Estimation.
* Policy Execution: assign j and PID control，我理解 PID controller 是为了给 lambda 更新慢的的情况来兜底
* Offline Estimator
* 感觉是个离线 batch 任务，模型预估不同算力下的ctr

* Experiments：控精排条数，增加条数有明显的边际效益
* TODO: fairness 问题、全链路算力分配

* 一些引用的论文
  * Deep Learning Inference in Facebook Data Centers: Characterization, Performance Optimizations and Hardware Implications
  * RobinHood: Tail latency aware caching–dynamic reallocation from cache-rich to cache-poor



#### A scalable pipeline for designing reconfigurable organisms, PNAS 2020

ML with bioengineering

如何探索更高效的器官组织

* 模拟(silico)：performant + conform to constraints
* 模拟(silico) ->现实(vivo)：noise resistance + build filter
* 目标：见 Object Manipulation 小节
