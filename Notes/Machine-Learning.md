[toc]

## Machine Learning

Materials

* http://neuralnetworksanddeeplearning.com/

### ML Basics

#### Algorithms

* crossentropy、KL散度、logistic regression、softmax
  * KL散度 ---> CE loss: [看得见的信息论-为什么用交叉熵作为逻辑回归的代价函数](https://zhuanlan.zhihu.com/p/31207556)
  * logistic regression ---> softmax
  * CE loss + softmax ---> 极其简洁的梯度形式
    * [求导推导](https://zhuanlan.zhihu.com/p/27223959)
    * $\frac{\partial l_{CE}}{\partial a_j}=y_j -t_j$

* XGBoost: gradient boosted trees works by combining predictions from many simple models, each of which tries to address the weaknesses of the previous models. By doing this the collection of simple models can actually outperform large, complex models.



* Feature Bagging

  * offering a potentially useful way of managing the bias-variance tradeoff

  * We were also interested in this as a potentially useful way to further parallelize training

  * 《An experimental comparison of three methods for constructing ensembles of decision trees: Bagging, boosting, and randomization》


* Dropout
  * 保证training/serving一致性：training或serving时scale
  * In the dense setting, dropout serves to separate effects from strongly correlated features, resulting in a more robust classifier. But in our sparse, noisy setting adding in dropout appears to simply reduce the amount of data available for learning. 《Ad Click Prediction: a View from the Trenches》
* Initialization
  
  * Xavier Initialization
  * ![image-20241219212615709](./Machine-Learning/image-20241219212615709.png)
  * Kaiming Initialization主要用于激活函数为ReLU（Rectified Linear Unit）的神经网络。
  
* 过拟合

  * 过拟合问题存在其他更深刻的原因。例如，将 28 × 28 的图片实施扁平化操作，将其变换为一个长度为 784 的一维向量，这将完全丢失了像素的空间排列信息

  * [为什么过多的特征（feature）导致过拟合（over-fitting)？ - Dr.Shiki的回答 - 知乎](https://www.zhihu.com/question/47375421/answer/306771331)

* 梯度相关

  * 梯度的意义：

    * 方向导数最大

    * ![image-20241219183022789](./Machine-Learning/image-20241219183022789.png)

  * [灾难遗忘现象](https://en.wikipedia.org/wiki/Catastrophic_interference)

  * 梯度消失和梯度爆炸：网络太深，网络权值更新不稳定造成的。本质上是因为梯度反向传播中的连乘效应




#### Optimizer

##### Optimization Problem

* Total Error = Optimization Error + Representation Error
* $F(w_{alg}) = F(w_{alg})-F(w_*) + F(w_*)$
* $F(w_*) \equiv \frac{1}{n} \sum_{i \in [n]} l(h_{w}(x_i), y_i) $
  * 模型预估误差均值，取决于模型结构

##### GD: 1st-order method

* 梯度下降：$w_t \leftarrow w_{t-1} - \eta \nabla F(w_{t-1})$
* Explanation: 一阶泰勒展开
* 性质：$$\mu \leq \frac{\|\nabla f(a) - \nabla f(b)\|}{\|a-b\|} \leq L, \forall a,b \in \R^d$$
  * 强凸：梯度变化率有下界
  * Lipchitz continuous gradient：梯度变化率有上界
* Note:
  * 令下标趋近，这个上下界本质是Hessian: f''(b) 的上下界
  * "linear convergence" means: $$F(w_{t+1}) - F(w_*) \leq (1-\frac{\mu}{L}) \left( F(w_t) - F(w_*) \right)^1$$
  * This convergence is for the function value $$F(w)$$ (there are other types of convergence)

##### Newton's method: 2nd-order method

* Gradient descent: $$w_t \leftarrow w_{t-t} - \eta (\nabla^2 F(w_{t-1}))^{-1} \nabla F(w_{t-1})$$

* $$ F(w) = \frac{1}{2} w^\top D w$$ , D is a diagonal matrix with all positive diagonal elements

  * 举一个例子，对比GD和Newton法的收敛速度
  * GD: 计算 $$F(w_{t+1})\le F(w_t)$$ 的恒成立条件是 $$\eta \lt \frac{2}{max_iD_{ii}}$$
  * Newton法，在例子中一步收敛了
  * 类似于在不同维度使用 Adaptive Learning Rate（D^-1，反映gradient的变化率）的效果
  * Quadratic Convergence

* Convergence

  * Strong convexity $$\mu$$ and Lipschtiz hessian $$H$$

  * Quadratic convergence: 

    * $$\|w_{t+1} - w_*\| \leq \frac{H}{\mu} \|w_t - w_*\|^2$$

    - Also needs a "good" initial value: $$\|w_0-w_*\| \leq \frac{\mu}{2H}$$

##### Polyak Momentum

* $$w_t \leftarrow w_{t-1} - \eta \nabla F(w_{t-1}) + \beta(w_{t-1} - w_{t-2})$$
* The formula above is equivalent to
  - $$v_t \leftarrow \eta \nabla F(w_{t-1}) + \beta v_{t-1}$$, $$w_t \leftarrow w_{t-1} - v_t$$
  - learning rate $$\eta$$ inside momentum variable $$v$$

- But we can also put learning rate outside the momentum:
  - $$v_t \leftarrow \nabla F(w_{t-1}) + \beta v_{t-1}$$, $$w_t \leftarrow w_{t-1} - \eta v_t$$
  - Caution: these 2 formulas will be different if the learning rate changes (warmup, decay)

##### Nesterov Momentum

- Concept: **lookahead** to get a better gradient estimation

- 理论上是两步，本方法基于最新model计算gradient，解决半步的staleness

* pytorch实际实现中，保留的是lookhead model

##### SGD: stochastic methods

* $$\min_{t} E\left[ \|\nabla F(w_{t-1})\|^2\right] \leq \frac{1}{T} \sum_{t=1}^T E\left[ \|\nabla F(w_{t-1})\|^2 \right] \leq \frac{2E[F(w_{0}) - F(w_*)]}{\eta T} + \frac{L\eta V_1}{b}$$
* 2 parts of error:
  - Escape from initial point to optimal
  - Variance (reduced by batch size)
* Typically, we take $$\eta\propto\frac{1}{\sqrt{T}}$$
  - so that $$\frac{1}{T} \sum_{t=1}^T E\left[ \|\nabla F(w_{t-1})\|^2 \right] \leq O(\frac{1}{\sqrt{T}})$$

- Implies learning rate **decay** for convergence: $$\eta_t \propto \frac{1}{\sqrt{t}}$$

- Converges to a point where $$\nabla F(w) = 0$$, could be a saddle point or local minimum, not necessarily a global minimum

##### Federated Averaging

《Advances and open problems in federated learning》p22


##### AdaGrad: a natural learning rate decay

- Algorithm:

  - In step $$t$$
  - Compute gradient: $$g_t \equiv \nabla f(w_{t-1})$$
    - Note：g可以保留一部分当前emb值
  - Update "2nd moment": $$v_t \leftarrow v_{t-1} + g_t \circ g_t$$
  - Update model: $$w_{t} \leftarrow w_{t-1} - \eta \frac{g_t}{\sqrt{v_t + \epsilon}} $$

- 本质：

  - using the average $$\frac{1}{t}\sum_t g_t \circ g_t$$ to estimate hessian 

  - a **naturally decay learning rate** $$\frac{\eta}{\sqrt{t}}$$

- Note:

  - 工程实现时，手动给 v 设置一个上界

##### FTRL: AdaGrad + L1 reg + L2 reg

* Related Paper: 《Ad Click Prediction: a View from the Trenches, KDD 13》

* Online Learning and Sparsity
  * FTRL-Proximal(Follow The Proximally Regularized Leader): get both the sparsity provided by RDA and the improved accuracy of OGD

  * [在线学习（Online Learning）导读 - 吴海波的文章](https://zhuanlan.zhihu.com/p/36410780)
  * FTRL的数学本质：SGD（梯度 + L2）+稀疏性（L1）

  * 李亦锬大佬的机器学习答题集，很精彩，其中介绍了 FTRL 的实践意义
    https://zhuanlan.zhihu.com/p/20693546

##### FTRL with Group Lasso

* Paper: https://dl.acm.org/doi/pdf/10.1145/3357384.3358114
  * 注意 Group Lasso 项是 L2 范数的一次幂
* Lasso: https://en.wikipedia.org/wiki/Lasso_(statistics)
* 应用：优化 sparse feature embedding layer (fid -> embedding vector layer) 的 model sparsity，将每个特征的 vector 当作一个 group

##### Adam

* Intro
  * adaptive moment estimation
  * Momentum 善于处理梯度的方向和大小，而 RMSProp 善于调整学习率以应对数据的稀疏性。Adam 的提出是为了结合这两种算法的优点，同时减少它们的缺点，提供一种更加鲁棒的优化解决方案。

- Algorithm:
  - In step $$t$$
  - Compute gradient: $$g_t \equiv \nabla f(w_{t-1})$$
  - Update 1st moment: $$m_t \leftarrow \beta_1 m_{t-1} + (1-\beta_1) g_t$$
  - Update 2nd moment: $$v_t \leftarrow \beta_2 v_{t-1} + (1-\beta_2) g_t \circ g_t$$
  - Bias-corrected 1st moment: $$\hat{m}_t \leftarrow \frac{m_t}{1-\beta_1^t}$$
    - 动机是没有 learning rate decay
    - 可尝试去掉，等价于learning rate warmup，会有点接近AdaGrad
  - Bias-corrected 2nd moment: $$\hat{v}_t \leftarrow \frac{v_t}{1-\beta_2^t}$$
  - Update model: $$w_{t} \leftarrow w_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} $$
  - Note: 
    - bias correction could be ignored
    - AdaGrad uses **uniformly weighted** average, while Adam assigns **larger weights for later** items
      - Intuition: 哪一种近似 Hessian 的方式在模型上更合适，需要考虑旧的 sparse item 的期望更新方式
    - AdaGrad has learning rate decay, while Adam doesn't have learning rate decay
      - Intuition: 结合同/异步训练方式思考，动机是训练后期减少lr让模型收敛更稳定（比如 W&D model 的 dense）
- Intuition
  - 1st momentum: 类似 Polyak Momentum
    - Also see SAG: https://arxiv.org/pdf/1309.2388v2.pdf
  - 2nd momentum
    - 用外积矩阵近似 Hessian 矩阵
- 不保证理论收敛
  - 2 ways to fix:
    - Use $$\max\{\hat{v}_t, \hat{v}_{t-1}, \ldots \hat{v}_1\}$$instead of $$\hat{v}_t$$to guarantee decreasing $$\frac{\eta_t}{\sqrt{\hat{v}_t} + \epsilon}$$: AMSGrad
    - Take $$\beta_2 \propto 1-\frac{1}{t}$$, approaches 1 when $$t$$ approaches infinity,  $$v_t$$barely changes at the end
- Note：
  - sparse 部分不适用 Adam：滑动平均用到了历史信息
  - 配合 slow start 技术，前期并发数缓慢增大

##### RMSProp

* Intro

  * RMSProp 善于调整学习率以应对数据的稀疏性

  * 本质：Adam with $$\beta_1=0$$, without any bias correction


##### Lookahead Optimizer: k steps forward, 1 step back, NIPS 2019

* 本文是SGD场景，slow weights + 主要提升训练稳定性、减小优化器的variance
* mini-batch 异步SGD场景也可以应用，提升模型效果
  * CV、NLP场景可以重复训同一批样本，这样的手段更有意义
  * 推荐、广告场景，假如照搬，感觉会丢失 fine-grained gradients 信息，但在异步训练场景，多worker更新参数天然构成了slow weights

* Method
  * Slow weights trajectory: We can characterize the trajectory of the slow weights as an exponential moving average (EMA) of the final fast weights within each inner-loop, regardless of the inner optimizer.
  * Proposition 1 (Optimal slow weights step size)

* 分析convergence
  * Proposition 2 (Lookahead steady-state risk): Lookahead has a variance fixed point that is strictly smaller than that of the SGD inner-loop optimizer for the same learning rate
  * Deterministic quadratic convergence: underdamped系统提升稳定性，overdamped系统略有损收敛

##### LAMB

- Algorithm:
  - In step $$t$$:
  - Compute update based on any optimizer: $$u_t$$
    - SGD: $$u_t = g_t$$
    - Adam: $$u_t = \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} $$
    - RMSProp: $$u_t=\frac{g_t}{\sqrt{v_t + \epsilon}} $$
  - Layer-wise normalization:
    - $$\hat{u}_t \leftarrow \frac{\|w_{t-1}\|}{\|u_t\|} u_t$$
  - Update model:
    - $$w_t \leftarrow w_{t-1} - \eta \hat{u}_t$$

- Intuition:
  - In large-batch training:
    - $$\|u_t\|$$ is unstable
  - Using large learning rate: diverge for large $$\|u_t\|$$
  - Using small learning rate: slow convergence for small $$\|u_t\|$$
  - LAMB: 
    - Adaptive learning rate: $$ \frac{\eta \|w_{t-1}\|}{\|u_t\|}$$
  - Smaller when $$\|u_t\|$$is large
  - Larger when $$\|u_t\|$$is small
  - Normalize $$\|u_t\|$$to the same scale of $$\|w\|$$

- Note:
  - We can apply LAMB normalization to any base optimizer
  - But the learning rate must be re-tuned

#### 激活函数

* Intro
  * 选激活函数 https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/
    * When using the ReLU function for hidden layers, it is a good practice to use a “*He Normal*” or “*He Uniform*” weight initialization and scale input data to the range 0-1 (normalize) prior to training.
  * 典型问题：XOR问题
  * ![image-20241221123336418](./Machine-Learning/image-20241221123336418.png)



* sigmoid函数
  * 1/(1+e^(-x))
  * 非常适合作为模型的输出函数用于输出一个0~1范围内的概率值
  * 已经不太受欢迎，实际中很少作为激活函数
    * 容易造成梯度消失。我们从导函数图像中了解到sigmoid的导数都是小于0.25的，那么在进行反向传播的时候，梯度相乘结果会慢慢的趋向于0。这样几乎就没有梯度信号通过神经元传递到前面层的梯度更新中，因此这时前面层的权值几乎没有更新，这就叫梯度消失。除此之外，为了防止饱和，必须对于权重矩阵的初始化特别留意。如果初始化权重过大，可能很多神经元得到一个比较小的梯度，致使神经元不能很好的更新权重提前饱和，神经网络就几乎不学习。
    * 函数输出不是以 0 为中心的，梯度可能就会向特定方向移动，从而降低权重更新的效率
    * 指数计算消耗资源
* Tanh(x)=2Sigmoid(2x)−1
  * 相比sigmoid，以0为中心
* ReLU
  * 优点：
    * ReLU解决了梯度消失的问题，当输入值为正时，神经元不会饱和
    * 由于ReLU线性、非饱和的性质，在SGD中能够快速收敛
    * 计算复杂度低，不需要进行指数运算
  * 缺点：
    * 输出不是以0为中心的
    * Dead ReLU 问题：要设置一个合适的较小的学习率

* Leaky ReLU：解决了ReLU输入值为负时神经元出现的死亡的问题
  * 函数中的α，需要通过先验知识人工赋值（一般设为0.01）
  * 有些近似线性，导致在复杂分类中效果不好。
* Parametric ReLU：alpha可学习
* ELU：
  * ![image-20241221141507094](./Machine-Learning/image-20241221141507094.png)

#### Tuning

https://github.com/google-research/tuning_playbook

#### Visualization

##### t-SNE：可视化高维向量

* 科普文章 https://medium.com/@sachinsoni600517/mastering-t-sne-t-distributed-stochastic-neighbor-embedding-0e365ee898ea



#### Evalutaion/Validation

* Metrics
  * **Mean Absolute Error** (MAE)
  * Normalized Discounted Cumulative Gain (NDCG)
  * Root Mean Square Error 

* holdout validation, cross-validation, leave-one-out validation, etc
  * “leave-one-out” 将数据分割为训练集、验证集和测试集。具体操作是对于每个用户，将其一个交互行为数据留出作为测试集，其余的作为训练集和验证集。例如，对于有个交互行为的用户，选择其中第个行为作为测试数据，其余个行为用于训练和验证。

```python
train_data, validation_data, test_data = np.split(model_data.sample(frac=1, random_state=1729), [int(0.7 * len(model_data)), int(0.9 * len(model_data))])   # Randomly sort the data then split out first 70%, second 20%, and last 10%
```

### 衡量相关性

* cosine similarity
* Pearson correlation
  * ![image-20241231004219620](./Machine-Learning/image-20241231004219620.png)



### 大 Batch 训练

分布式SGD在算法方面的挑战

* throughput ~ GPU num
  * 深度学习的大规模训练通常以线性增加的理想情况为基准，Horovod和NCCL库在保持高吞吐量方面做得很好，但是他们的性能与所使用的硬件有着千丝万缕的联系。高带宽和低延迟的要求导致了NVLink互连的开发，它是本课程所使用的服务器用来互连一个节点上的多个GPU的方法。 NVIDIA DGX-2通过NVSwitch将这种互连又推进一步，该互连结构可以300GB/s的峰值双向带宽连接多达16个GPU。

* critical batch size ~ gradient noise scale (openai)
* 对精度的影响：朴素的方法（比如不加data augmentation）会降低精度
  * ImageNet training in minutes. CoRR
  * [Train longer, generalize better: closing the generalization gap in large batch training of neural networks](https://arxiv.org/abs/1705.08741)
  * [On large-batch training for deep learning: Generalization gap and sharp minima](https://arxiv.org/abs/1609.04836)
  * [Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913)

* 应对策略

  * 提高学习率：One weird trick for parallelizing convolutional neural networks
  * 早期学习率热身： Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour.
* Batch Normalization
  * BN通过最小化每个层的输入分布中的漂移来改善学习过程
    * 缓解了深层网络中“梯度弥散”的问题（Internal Covariate Shift）
  * 提高学习速度并减少使用 Dropout 的需求
  * 想法是针对每批数据对**所有层**的输入 进行规一化（这比简单地只对输入数据集进行规一化更为复杂）
    * 为了保持模型的表达能力，引入可学习的参数，缩放因子和平移因子
* Ghost BN
  * 计算更小批量的统计数据（“ghost 批量”）
    * 引入其他噪声
  * 按 GPU 逐个单独执行批量归一化，解决同步 BN 通信开销问题
* 将噪声添加至梯度
  * 确保权重更新的协方差随着批量大小的变动保持不变 
  * 不会改变权重更新的平均值 
  * $$\hat{g}=\frac{1}{M}\sum^{N}_{n\in B}g_n z_n$$
* 更长的高学习率训练时间
* 增加批量大小代替学习率衰减
* LARS – 按层自适应学习率调整
  *  [LARS论文](https://arxiv.org/abs/1904.00962): 大LR -> LR warm-up -> LARS，只是能保证大batch训练能训，关于效果问题，作者认为“increasing the batch does not give much additional gradient information comparing to smaller batches.”
  *  [LARC](https://github.com/NVIDIA/apex/blob/master/apex/parallel/LARC.py): 带梯度裁剪的分层自适应学习率，以具有动力的SGD作为基础优化器
  *  [LAMB](https://arxiv.org/abs/1904.00962): 分层自适应学习率，以 Adam 作为基础优化器，在BERT等语言模型上比LARC更成功
  *  [NovoGrad](https://arxiv.org/abs/1905.11286): 按层计算的移动平均值，在几个不同的领域也有不错的表现

![training_result](Machine-Learning/training_result.png)



### ML Theory

* 2024 诺贝尔物理学奖授予人工神经网络机器学习，为什么会颁给 AI 领域？ - SIY.Z的回答 - 知乎
  https://www.zhihu.com/question/777943030/answer/4508673022
  * RBM 受限玻尔兹曼机
  * sigmoid 函数有了自然的解释：玻尔兹曼分布下隐含层神经元激活的条件概率的激活函数。
* 大模型是不是能够稀疏化? 
  * 从物理和能量密度/信息密度的角度来看, 似乎是可以的. 
  * 但是从范畴论的角度, 特别是预层范畴的角度来看待,Dense的Foundation Model训练是必须要做的, 因为只有在相对Dense的模型结构上才能更好的捕获所有的态射. 
  
* TOPOS理论 在ML的应用
  * on the diagram of thought https://github.com/diagram-of-thought/diagram-of-thought
  





### Learning To Rank

#### GBDT（Gradient Boosting Decision Tree）

* 通过拟合残差，生成第N颗子树

#### XGBoost

https://xgboost.readthedocs.io/en/stable/tutorials/model.html

XGBoost stands for “Extreme Gradient Boosting”, where the term “Gradient Boosting” originates from the paper *Greedy Function Approximation: A Gradient Boosting Machine*, by Friedman.

![image-20241210004602072](./Machine-Learning/image-20241210004602072.png)

![image-20241210004925060](./Machine-Learning/image-20241210004925060.png)

![image-20241210004943879](./Machine-Learning/image-20241210004943879.png)

![image-20241210005132323](./Machine-Learning/image-20241210005132323.png)

![illustration of structure score (fitness)](./Machine-Learning/struct_score.png)



#### LTR

https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html

* Intro
  * The default objective is `rank:ndcg` based on the `LambdaMART` [[2\]](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html#references) algorithm, which in turn is an adaptation of the `LambdaRank` [[3\]](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html#references) framework to gradient boosting trees. For a history and a summary of the algorithm, see [[5\]](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html#references)
  * 《Unbiased LambdaMART: An Unbiased Pairwise Learning-to-Rank Algorithm》
* 调参
  * lambdarank_num_pair_per_sample

### Position Bias

* Intro

  * Obtaining real relevance degrees for query results is an expensive and strenuous, requiring human labelers to label all results one by one. When such labeling task is infeasible, we might want to train the learning-to-rank model on user click data instead, as it is relatively easy to collect. Another advantage of using click data directly is that it can reflect the most up-to-date user preferences [[1\]](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html#references). However, user clicks are often biased, as users tend to choose results that are displayed in higher positions. User clicks are also noisy, where users might accidentally click on irrelevant documents. To ameliorate these issues, XGBoost implements the `Unbiased LambdaMART` [[4\]](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html#references) algorithm to debias the position-dependent click data. The feature can be enabled by the `lambdarank_unbiased` parameter; see [Parameters for learning to rank (rank:ndcg, rank:map, rank:pairwise)](https://xgboost.readthedocs.io/en/stable/parameter.html#ltr-param) for related options and [Getting started with learning to rank](https://xgboost.readthedocs.io/en/stable/python/examples/learning_to_rank.html#sphx-glr-python-examples-learning-to-rank-py) for a worked example with simulated user clicks.

  







### Quantization

#### 模型量化介绍

* 神经网络：多函数的嵌套表示
  * 越来越不规则

Serving 量化

* 用于存储的模型量化：
  * 传统问题局限性：求解量化误差最小，不面向loss函数，面向策略，不可解

* 用于计算的模型量化
  * 权重和输入都有delta（预估时认为权重delta为零）
  * 偏微分公式 -> 每层的输出到下一层的输入很重要
    * 同样的量化方式，相同量化精度给不同层的输入带来不同的误差
    * 存储量化 v.s 计算量化，后者更强调在存储约束下求解最优精度
  * 一种可求闭式解（分层量化模型）：量化标准排序、梯度排序，一一对应，排序不等式证明
    * e.g. HAWQ-v2

Training 量化

* 量化感知训练的原理：李沐的ps文章《communication efficient distributed machine learning with the parameter server》https://www.cs.cmu.edu/~muli/file/parameter_server_nips14.pdf
* 结论：控制梯度噪音的范数
  * 小结论：量化训练完后要恢复全精度进行计算，再用训练后量化手段进行量化
  * 实现上：量化的正传，量化/全精度的反传，量化的更新
    * 全精度反传，与自动求导模块的实现有关，可能存在

* 工具：https://github.com/NVIDIA/apex

总结：

* 量化问题本质是NP难问题，部分情况下可转换成指数规划问题
* 量化训练和预测是两个目标，训练结果应该恢复成全精度再用预测压缩的过程压缩一遍

### Deep Learning Basic

#### Intro

> 三要素

* 特征
* 结构
  * 多层：提取深层特征
  * 非线性变换
* 参数
  * Optimizer
  * 人工设计：
    * 差分
    * 空洞卷积
    * 非线性变换
  * e.g. 图片边缘提取

* 现代算法是策略和模型的融合：人工设计参数、特征提取、模型结构，有相似之处
  * 特征提取：图像处理中的一阶、二阶特征
    * 一阶特征主要关注图像的亮度和颜色信息，二阶特征则侧重于图像的纹理结构信息
  * 人工设计参数：
    * 比如MNIST768维像素，压缩到28维（每一行多少个点 / 每一列多少个点）
  * 模型结构：
    * ResNet





### Bert

* Transformer 具有 field reduce 能力，将 N 个 token reduce 成 M 个 token
* [GELU](https://paperswithcode.com/method/gelu)
  * GELUs are used in [GPT-3](https://paperswithcode.com/method/gpt-3), [BERT](https://paperswithcode.com/method/bert), and most other Transformers.


#### model finetune

* model finetune是基于BERT预训练模型强大的通用语义能力，使用具体业务场景的训练数据做finetune，从而针对性地修正网络参数，是典型的双阶段方法。（[BERT在美团搜索核心排序的探索和实践](https://zhuanlan.zhihu.com/p/158181085)）
* 在BERT预训练模型结构相对稳定的情况下，算法工程师做文章的是模型的输入和输出。首先需要了解BERT预训练时输入和输出的特点，BERT的输入是词向量、段向量、位置向量的特征融合（embedding相加或拼接），并且有[CLS]开头符和[SEP]结尾符表示句间关系；输出是各个位置的表示向量。finetune的主要方法有双句分类、单句分类、问答QA、单句标注，区别在于输入是单句/双句；需要监督的输出是 开头符表示向量作为分类信息 或 结合分割符截取部分输出做自然语言预测。
* 搜索中finetune的应用：model finetune应用于query-doc语义匹配任务，即搜索相关性问题和embedding服务。在召回and粗排之后，需要用BERT精排返回一个相关性分数，这一问题和语句分类任务有相似性。搜索finetune的手法有以下特点：
  * 广泛挖掘有收益的finetune素材：有效的包括发布号embedding、文章摘要、作者名，训练手段包括直接输入、预处理。model finetune方法能在标注数据的基础上，利用更多的挖掘数据优化模型。
  * 改造模型输入or输出
    * 模型输入
      * 简单的title+summary+username+query拼接
      * 多域分隔：“考虑到title和summary对于query的相关性是类似的分布，username和query的相关性关联是潜在的。所以给user_name单独设了一个域，用sep分隔”
    * 模型输出
      * 门过滤机制，用某些表示向量的相应分数加权CLS的语句类型输出分
      * 引入UE，直接和CLS输出向量concat
  * 素材的进一步处理，引入无监督学习
    * 在model finetune的有监督训练之前，利用text rank算法处理finetune素材，相当于利用无监督学习提升了挖掘数据 —— 喂入BERT的数据的质量。
    * 截断摘要，实测有效
  * Bert训练任务的设计方式对模型效果影响大
    * 将finetune进一步分为两阶段，把质量较低、挖掘的数据放在第一阶段finetune，质量高的标注数据放在第二阶段finetune，优化finetune的整体效果。
    * 这种递进的训练技巧在BERT中较常见，论文中也有将长度较短的向量放在第一阶段训练的方法。

#### 向量降维

* 向量白化
  * https://arxiv.org/pdf/2103.15316

### Contrastive Learning

#### Intro

[Constrastive Learning: MoCo and SimCLR](https://mp.weixin.qq.com/s/v5p9QA3vDl-WTF3-7shp4g)

#### 训练 Dense Retriever

* Query2Doc paper
  * For training dense retrievers, several factors can influence the final performance, such as hard negative mining (Xiong et al., 2021), intermediate pretraining (Gao and Callan, 2021), and knowledge distillation from a cross-encoder based re-ranker (Qu et al., 2021). In this paper, we investigate two settings to gain a more comprehensive understand- ing of our method. The first setting is training DPR (Karpukhin et al., 2020) models initialized from BERTbase with BM25 hard negatives only
  * ![image-20241117211622999](Machine-Learning/image-20241117211622999.png)



### NLP

#### Intro

概念：语言模型

* [Bag-of-words(BoW) model](https://en.wikipedia.org/wiki/Bag-of-words_model) 可作为一种信息模型，表示句子或图片，用于衡量相似度或别的用途
* stop words: 停用词
* [Tokenization and text normalization](https://www.analyticsvidhya.com/blog/2021/03/tokenization-and-text-normalization/)

#### Embedding

* one-hot的缺点：
  * 过于稀疏
  * 无法体现距离
  * 没有数学或逻辑关系
    * e.g. 国王 - 男人 + 女人 = 女王

* NLP Embedding
  * Word2Vec
  * CLIP
  * OpenAI Embedding
* 每个Token对应一个embedding
  * GPT-3: 每个token **12228维**
  * 经典的transformer，每个向量只有512维
* 向量空间模型（Vector Space Model，以下简称VSM），主要基于两个假说：
  * 词袋假说（Bag of Words Hypothesis）和分布假说（Distributional Hypothesis）。
  * 前者是说，一篇文档的词频（而不是词序）代表了文档的主题；
  * 后者是说，上下文环境相似的两个词有着相近的语义。

#### Word2Vec: Efﬁcient Estimation of Word Representations in
Vector Space

> https://www.cnblogs.com/sandwichnlp/p/11596848.html
>
> * CBOW和NNLM的区别：
>   * 移除NNLM模型中的Hidden layer结构；
>   * 直接将Embedding layer的查表之后累加求和（NNLM是将输出结果拼接）
>   * 将下文单词纳入上、下文环境，真正考虑了Context（NNLM的输入严格来说为上文文本）

* Intro
  * 为什么之前流行 N-gram model
    * simplicity, robustness and the observation that simple models trained on huge amounts of data outperform complex systems trained on less data
  * 改进路线：distributed representations of words [10]
  * datasets with billions of words, and with millions of words in the vocabulary
    * multiple degrees of similarity
    * vector(”King”) - vector(”Man”) + vector(”Woman”)   -- **syntactic/semantic relationships**

* 模型

  * Feedforward Neural Net Language Model (NNLM)
    * 效率优化：Huffman tree based **hierarchical softmax**
  * Recurrent Neural Net Language Model (RNNLM)
  * Continuous Bag-of-Words Model
    * 去除hidden layer
    * ![image-20241230025152492](./Machine-Learning/image-20241230025152492.png)
  * Continuous Skip-gram Model
    * Since the more distant words are usually less related to the current
      word than those close to it, we give less weight to the distant words by sampling less from those words in our training examples.
  * ![image-20241230025140727](./Machine-Learning/image-20241230025140727.png)

* 训练：

  * trained in two steps:
    * ﬁrst, continuous word vectors are learned using simple model
    * and then the N-gram NNLM is trained on top of these distributed representations of words.
  * Noise-Contrastive Estimation
    * https://www.cnblogs.com/sandwichnlp/p/11596848.html

* 结论：

  * Skip - gram 在语义任务表现出色，CBOW 在句法任务较好
  * **微软研究句子完成挑战**：Skip-gram 模型与 RNNLMs 结合在该任务中取得新的最优结果。

* MLSys

  * DistBelief：异步训练 + adagrad

* [word2vec的局限性](https://www.cnblogs.com/sandwichnlp/p/11596848.html#4144593371)

  总的来说，word2vec通过嵌入一个线性的投影矩阵（projection matrix），将原始的one-hot向量映射为一个稠密的连续向量，并通过一个语言模型的任务去学习这个向量的权重，而这个过程可以看作是无监督或称为自监督的，其词向量的训练结果与语料库是紧密相关的，因此通常不同的应用场景需要用该场景下的语料库去训练词向量才能在下游任务中获得最好的效果。这一思想后来被广泛应用于包括word2vec在内的各种NLP模型中，从此之后不单单是词向量，我们也有了句向量、文档向量，从Word Embedding走向了World Embedding的新时代。word2vec非常经典，但也有其明显的局限性，其主要在以下几个方面：

  1. 在模型训练的过程中仅仅考虑context中的局部语料，没有考虑到全局信息；
  2. 对于英文语料，对于什么是词，怎样分词并不是问题（单个词就是独立的个体）。而对于中文而言，我们在训练词向量之前首先要解决分词的问题，而分词的效果在很多情况下将会严重影响词向量的质量（如分词粒度等），因此，从某些方面来说word2vec对中文不是那么的友好；
  3. 在2018年以前，对于word2vec及其一系列其他的词向量模型都有一个相同的特点：其embedding矩阵在训练完成后便已经是固定了的，这样我们可以轻易从网上获取到大量预训练好的词向量并快速应用到我们自己任务中。但从另一个角度来说，对于同一个词，在任意一个句子，任意一个环境下的词向量都是固定的，这对于一些歧义词来说是存在较大问题的，这也是限制类似word2vec、Glove等词向量工具性能的一个很重要的问题。

#### GloVe(Globel Vectors)算法

* Intro
  * SVD分解与Word2Vec的结合。
  * 现矩阵X，该矩阵中的Xij表示第j个单词出现在以第i个单词为中心，长度为n的窗口中的次数。将长度为n的窗口遍历整个语料库，则得到了共现矩阵X。
  * 考虑全局信息
* 推导
  * ![image-20241231003216044](./Machine-Learning/image-20241231003216044.png)



### Position Encoding

* 绝对位置编码：
  * Convolutional Sequence to Sequence Learning
  * 正弦-余弦编码
* 相对位置编码：
  * 作用于自注意力机制





#### 词之间的相似度

* Embedding-based
  * SentenceEmbedding

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('...')

field_embeddings = model.encode(fields)

def find_most_similar(input_word):
    input_embedding = model.encode(input_word)
    similarities = util.cos_sim(input_embedding, field_embeddings)
    best_index = similarities.argmax()
    return fields[best_index]
```



* 编辑距离
  * Levenshtein
  * fuzzywuzzy

```python
from fuzzywuzzy import process
def find_most_similar(input_word):
    match, score = process.extractOne(input_word, fields)
    return match
```

#### 句子之间的相似度

* 编辑距离

* BLEU（Bilingual Evaluation Understudy）

  - **定义**：BLEU 是一种用于评估机器翻译质量的指标。它通过比较机器生成的翻译文本与参考（人工翻译）文本之间的 n - gram（n 个连续单词的序列）重叠情况来衡量翻译的准确性。
    - 工作原理
      - 计算生成文本和参考文本之间的 n - gram（如 1 - gram、2 - gram、3 - gram 和 4 - gram）的匹配数量。例如，对于一个句子，1 - gram 是单个单词的匹配，2 - gram 是两个连续单词的匹配，以此类推。
      - BLEU 得分是这些 n - gram 匹配率的几何平均值，并且会**考虑简短回答的惩罚因子**。如果生成的句子过短，会受到惩罚，以避免系统总是生成非常简短但可能部分匹配的句子来获取高分。
      - 在整个测试集上平均下述值
      - 完整计算公式：$\mathrm{BLEU}_4=\min\left(1,\frac{output-length}{reference-length}\right)\left(\prod_{i=1}^4 precision_i\right)^{\frac{1}{4}}$

  - **应用场景**：广泛应用于机器翻译系统的评估，帮助比较不同翻译模型或算法的性能，确定哪种模型能够生成更接近人工翻译质量的译文。

* ROUGE（Recall-Oriented Understudy for Gisting Evaluation）

  - **定义**：ROUGE 是一组用于评估自动文本摘要和机器翻译质量的指标，主要侧重于召回率（Recall），即衡量系统生成的文本能够包含参考文本中多少重要信息。
    - Rouge-N：将模型生成的结果和标准结果按 N-gram 拆分后，只计算召回率；
    - Rouge-L: 利用了最长公共子序列（Longest Common Sequence），计算：$P=\frac{LCS(c,r)}{len(c)}$, $R=\frac{LCS(c,r)}{len(r)}$, $F=\frac{(1+\beta^2)PR}{R+\beta^2P}$
    - 函数库：https://pypi.org/project/rouge-score/
    - 对比 BLEU 与 ROUGE：
      - BLEU 能评估流畅度，但指标偏向于较短的翻译结果（brevity penalty 没有想象中那么强）
      - ROUGE 不管流畅度，所以只适合深度学习的生成模型：结果都是流畅的前提下，ROUGE 反应参照句中多少内容被生成的句子包含（召回）

  - **应用场景**：在文本摘要领域，用于评估自动生成的摘要是否能够准确地捕捉原始文本的主要内容；在机器翻译中，也可以帮助评估翻译后的文本是否完整地传达了原文的关键信息。

* METEOR（Metric for Evaluation of Translation with Explicit ORdering）

  - **定义**：METEOR 是一种综合考虑了精确率（Precision）、召回率（Recall）和词序（Word Order）的文本生成质量评估指标，用于评估机器翻译和其他文本生成任务。
    - 工作原理
      - 首先计算生成文本和参考文本之间的精确率和召回率，类似于 BLEU 和 ROUGE 的部分计算。
      - 然后，METEOR 引入了一个基于词序的 F-measure（调和平均数）来综合评估匹配质量。它通过考虑单词的匹配、同义词匹配（通过预定义的词表）和词序的连贯性来计算得分。
      - 还包括一个对齐模块，用于找到生成文本和参考文本之间单词的最佳匹配，考虑了多种匹配类型，如完全匹配、同义词匹配和词根匹配等。
    - 对语言学和语义词表有依赖，所以对语言依赖强。

  - **应用场景**：在机器翻译和文本生成任务中，METEOR 能够提供一个更全面的评估，因为它不仅仅关注单词或短语的匹配，还考虑了词序和语义相似性，能够更好地反映生成文本的自然度和准确性。

#### 指标

* 困惑度(perplexity)的基本概念及多种模型下的计算（N-gram, 主题模型, 神经网络）https://zhuanlan.zhihu.com/p/114432097






### CV

#### 卷积网络

* Intro

  * 过滤器常使用一个 4 维的张量表示，前两维表示过滤器的大小，第三维表示输入的通道数，第四维表示输出的通道数
    * 卷积核的深度或层数与输入图片的通道数目应保持一致。这是pytorch底层实现的，开发人员不需要考虑。
    * 卷积操作之后输出的通道数目是由卷积核个数所决定的。
  * 理解卷积
    * 函数拟合
    * 模式匹配：卷积核定义了某种模式，卷积运算是在计算每个位置与该模式的相似程度，或者说每个位置具有该模式的分量有多少，当前位置与该模式越像，响应越强。
  * 感受野：第一层卷积层的输出特征图像素的感受野的大小等于卷积核的大小。
  * Feature Map大小：
    * ![image-20241221015354226](./Machine-Learning/image-20241221015354226.png)

* 特点

  * 局部连接

  * 权值共享：所有神经元共享filter的参数，filters可以有多个

  * 下采样
    * L2 pooling
    * Local Contrast Normalization

![image-20221209232135219](Machine-Learning/conv-downsampling.png)

* Local feature map
  * CNN（卷积神经网络）中的局部特征图（Local Feature Maps）
    - 生成原理
      - 在 CNN 中，卷积层通过卷积核（filter）在输入图像（或上一层的特征图）上滑动进行卷积操作来生成局部特征图。例如，对于一个大小为的卷积核在一个的图像上滑动，卷积核每次覆盖的区域就是一个局部区域。卷积操作会对这个局部区域内的像素值进行加权求和等计算，得到新的特征值，这些特征值组成了输出特征图中的一个元素。当卷积核遍历完整个输入图像后，就生成了一个完整的局部特征图。
    - 特征表示意义
      - 这些局部特征图能够捕捉图像中不同位置的局部特征，如边缘、纹理等。例如，在一个用于图像分类的 CNN 中，较低层的局部特征图可能主要捕捉简单的边缘和纹理信息。随着网络层数的增加，局部特征图会逐渐组合这些简单特征，形成更复杂的局部特征，如物体的局部形状等，最终用于识别图像中的物体类别。
    - 应用场景
      - **图像分类**：通过提取局部特征图中的特征来区分不同类别的图像。例如，在识别猫和狗的图像时，CNN 可以通过局部特征图捕捉猫和狗的不同面部特征、毛发纹理等局部信息，进而判断图像类别。
      - **目标检测**：用于定位图像中的目标物体。局部特征图可以帮助确定目标物体的位置和大致形状，例如在检测汽车时，局部特征图可以捕捉汽车的轮廓、车窗等局部特征，从而找到汽车在图像中的位置。
  * ViT（Vision Transformer）中的局部特征图
    - 与 CNN 的区别
      - ViT 主要基于 Transformer 架构，其处理图像的方式与 CNN 有所不同。它将图像分割成一系列的图像块（patches），这些图像块可以看作是一种局部特征的表示，但与 CNN 的局部特征图在概念和生成方式上有差异。在 ViT 中，每个图像块经过线性投影等操作后，类似于序列中的一个元素，而不是像 CNN 那样通过卷积操作生成特征图。
    - 类似的局部特征提取方式（以图像块为基础）
      - ViT 把图像划分为固定大小的图像块，例如的图像块。这些图像块可以被视为局部特征的载体。每个图像块经过线性嵌入（linear embedding）后，被转化为一个向量，所有这些向量组成一个序列输入到 Transformer 的编码器（encoder）中。在这个过程中，虽然没有像 CNN 那样传统意义上的局部特征图，但这些图像块在某种程度上起到了局部特征提取的作用，并且通过 Transformer 的自注意力机制（self - attention mechanism），能够学习到图像块之间的全局关系，从而综合考虑局部和全局特征进行图像理解。
    - 应用场景
      - **图像分类与 CNN 类似**：在图像分类任务中，ViT 通过对图像块组成的序列进行处理，学习到图像的整体特征表示，用于区分不同的图像类别。由于 Transformer 架构的自注意力机制能够捕捉长距离的依赖关系，ViT 在一些大规模图像分类任务中表现出色。
      - **多模态融合等新兴场景**：在与其他模态（如文本）结合的多模态任务中，ViT 的图像块表示方式可以更容易地与其他模态的特征进行融合。例如，在图像 - 文本匹配任务中，图像块的特征可以和文本的词向量特征通过合适的方式结合，用于判断图像和文本是否相关。



* group convolution
  * only the input channels in the same group are used for computing a given output channel. A group convolution with total Ci input, Co output channels and G groups is essentially G independent convolutions each with d=Ci/G input and Co/G output channels. 
  * depth-wise convolution: Ci=Co=G and consequently group size d=1
* 1x1卷积核
  * 《Network-in-Network》论文中首次介绍了1x1卷积层用于“跨信道下采样”或“跨信道池池化”。
  * 1x1卷积用于减少信道数量
    * 伴随ReLU引入非线性

  * ![image-20241221012403964](./Machine-Learning/image-20241221012403964.png)


* 反卷积
  * 上采样
  * DCGAN
  * 转置卷积先将卷积核转为稀疏矩阵的形式，然后正向传播的时候左乘这个稀疏矩阵的转置，反向传播的时候左乘这个稀疏矩阵

#### 空洞卷积

* Dilated Convolution
  * dilation rate
  * 扩大了感受野，而不增加参数量
    * 降低过拟合
* Multi-astrous Convolution
  * 多尺度提取

#### 池化层

* Intro
  * ![image-20241221015412266](./Machine-Learning/image-20241221015412266.png)

* 降维
  * 对卷积层输出的特征图进行特征选择和信息的过滤，提取主要特征
  * 能够实现对特征图的下采样，从而减少下一层的参数和计算量
  * 通过减小特征图的维度，池化层有助于减少模型的参数数量，从而减小了过拟合的风险，提高了模型的泛化能力
  * 保持特征的不变性（平移、旋转、尺度）
* 常见分类
  * 平均池化：背景信息
  * max pooling：纹理特征信息

#### 注意力

* 通常将CV领域中注意力机制中的模型结构分为三大注意力域来分析，主要是：空间域(spatial domain)，通道域(channel domain)，混合域(mixed domain)。

  - 空间域——将图片中的的空间域信息做对应的空间变换，从而能将关键的信息提取出来。对空间进行掩码的生成，进行打分，代表是Spatial Attention Module。

  - 通道域——类似于给每个通道上的信号都增加一个权重，来代表该通道与关键信息的相关度的话，这个权重越大，则表示相关度越高。对通道生成掩码mask，进行打分，代表是Channel Attention Module。

  - 混合域——空间域的注意力是忽略了通道域中的信息，将每个通道中的图片特征同等处理，这种做法会将空间域变换方法局限在原始图片特征提取阶段，应用在神经网络层其他层的可解释性不强。

#### 传统图像处理

* [LBP (local binary patterns)](https://en.wikipedia.org/wiki/Local_binary_patterns)
  * resize到固定大小：大小越大则越准但有噪声，大小越小则误召回率高
  * hamming 距离度量
* NMS算法
  * 非极大值抑制
  * 1、对所有的框，通过一个置信度阈值将置信度低的框滤除。
  * 2、接着，选出置信度最高的框，将其保存进输出列表中。
  * 3、依次计算该框与其他剩余的框的IOU值。然后通过一个IOU阈值将和这个置信度最高的框拥有较大IOU的框（即和这个框相近的框）去除。
  * 4、 继续对剩余的框进行2，3操作，直到遍历完毕

### RL

* [AI挑战黑神话！死亡1000次，我训练的AI终于击败了首个BOSS【图灵计划10】](https://www.bilibili.com/video/BV1qE421c7mU)
* [【DQN只狼实战教程】手把手带你实现用强化学习DQN打只狼里的boss（第一期）](https://www.bilibili.com/video/BV1by4y1n7pe)



* 豆包 veRL https://arxiv.org/abs/2409.19256



[深度强化学习（一）强化学习概述 - iker peng的文章 - 知乎](https://zhuanlan.zhihu.com/p/22542101)

[深度强化学习系列（二）强化学习基础 - iker peng的文章 - 知乎](https://zhuanlan.zhihu.com/p/23436744)


### AutoML

* [为什么基于Bayesian Optimization的自动搜参没有大规模运用？](https://www.zhihu.com/question/33711002)
  * 无法并行化
  * automl的收益问题
  * 不太适合离散输入
* [AutoTVM 探秘（二）](https://reku1997.gitee.io/2019/12/31/autotvm-2/)

### 特征压缩、降维

* According to [JL-lemma](https://en.wikipedia.org/wiki/Johnson–Lindenstrauss_lemma), [random projection](https://en.wikipedia.org/wiki/Random_projection) reduces the dimensionality of data while approximately preserving the pairwise distances between data points.

  * 压缩 feature num 而非压缩 embedding size (PCA, SVD)

  * 输入 feature，输出 concated_compressed_embedding + dot_products

  * 变种：instance-wise (Y adaptive to X), implicit-order, GroupCDot, CDotFlex

  * 对比：比[AutoInt](https://arxiv.org/pdf/1810.11921.pdf)计算量小；比[DCN-V2](https://arxiv.org/pdf/2008.13535.pdf)线上效果好

* Q-Former

* Random projection
  * Random projection in dimensionality reduction: Applications to image and text data


### 特征交叉

##### CAN: Revisiting Feature Co-Action for Click-Through Rate Prediction

[想为特征交互走一条新的路 - 周国睿的文章 - 知乎](https://zhuanlan.zhihu.com/p/287898562)

feature co-action

1. sum-pooling: DIN 系列
2. graph-based
3. combinatorial embedding methods: DCN、PNN

1+2: the edges are only used for information aggregation but not information augmentation；edge weight 的维度不高，可能信息不足以刻画好 feature co-action

3: 同时进行 representation learning and co-action modeling，可能有冲突

CAN网络结构：

* 核心思路是有限度地扩充交叉特征参数，CAN独立参数
* Pitem做MLP参数，这样选取是考虑到 candidate ads 的量级比 user history少
* CAN本质上感觉是一种更“深度”的“朴素特征交叉”，既直接由 Pitem + Puser 输出 embedding，又不引入新的 dense MLP 参数，保证“穿越性”

相比笛卡尔积方案的优势是：1）参数解耦；2）参数量折中；3）冷启动

Multi-level Independence

* parameter independence
  * 我认为这篇文章核心思路在这里，稀疏特征的场景将 表征学习 与 特征交叉 解耦，这一思想与CV领域解决长尾分布问题，[表征学习 与 分类器 解耦](https://arxiv.org/abs/1910.09217)的思路异曲同工（本质还是“有限地”增加参数）
* combinations independence
* orders independence



##### DCN-V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems

1.Introduction

* DNN比较难学好二阶、三阶特征交叉 ---> implicit 转 explicit 的思路

* DCN的问题：Cross网络的参数量 O(input size) is overwhelmed by Deep网络

2.Related Work

* Parallel Structure: Wide & Deep, DeepFM, DCN, xDeepFM, AutoInt, InterHAt

* Stacked Structure: PNN(IPNN, OPNN), NFM, DLRM, AFN

* 一些对比的要点：特征交叉的方式、高阶特征交叉、定长/变长特征交叉

3.Proposed Architecture: DCN-V2

stacked and parallel structure

* cross network(DCN-M) 在W为对角阵时退化为DCN
* Cost-Effective Mixture of Low-Rank DCN：本质上是矩阵分解减参数，insights:
  * learn feature crosses in a subspace -> Mixture-of-Experts(MoE)
  * 利用低秩特性，先降维再升维 -> 在低维空间做非线性

6.Emprical Understanding

* DCN-V2的交叉能力优于DNN

* CrossNet ~ ReLu 学习非线性

* 朴素情况下的类比：rank threshold ~ feature num
  * 第8小节声明 rank=input_size/4 时无效果损失

9.Conclusion

DCN-V2: to model explicit crosses in an expressive yet simple manner. 

DCN-Mix: Observing the low-rank nature of the weight matrix in the cross network, to propose a mixture of low-rank DCN，是效果和延时的折中

### AI4Science

* [AlphaFold开发者获2024诺贝尔化学奖，AI抢夺科学家的最重要荣誉](https://mp.weixin.qq.com/s/BqO1-UN3hQ4Bagcp206_uw)
  * 从围棋到蛋白质结构预测
  * AlphaFold2 成功解决蛋白质折叠问题
  * AlphaFold 3 具备了药物设计的能力，可以预测药物中常用的分子（如配体和抗体），这些分子可与蛋白质结合，改变蛋白质在人类健康和疾病中的相互作用方式。
  * 方向：构建虚拟细胞
  * 基于AlphaFold 3 推出了免费平台AlphaFold Server，供全世界的科学家利用它进行非商业性研究，预测蛋白质如何与细胞中的其他分子相互作用。
  * 鲜为人知的是，AlphaFold一直存在诸多竞争者，其中最为知名的莫过于华盛顿大学的David Baker团队。
    * Baker是预测和设计蛋白质三维结构方法的开创者
  * “癌症、气候变迁、能源、基因组学、宏观经济学、金融系统、物理学等，太多我们想掌握的系统知识正变得极其复杂,如此巨大的信息量让最聪明的人穷其一生也无法完全掌握。如何才能从如此庞大的数据量中筛选出正确的见解呢？” 未来超级智能机器将与人类专家合作解决一切问题，一种通用人工智能可以自动将非结构化信息转换为可使用知识，这是一种针对任何问题的元解决方法（meta-solution）。
* ChemCrow


### Fundamentals of Deep Learning -- nvidia

[MNIST](http://yann.lecun.com/exdb/mnist/)

```python
from tensorflow.keras.datasets import mnist
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

import matplotlib.pyplot as plt
image = x_train[0]
plt.imshow(image, cmap='gray')

# Flattening the Image Data
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# Normalization
x_train = x_train / 255
x_test = x_test / 255 

import tensorflow.keras as keras
num_categories = 10
y_train = keras.utils.to_categorical(y_train, num_categories)
y_test = keras.utils.to_categorical(y_test, num_categories)

# instantiating the model
from tensorflow.keras.models import Sequential
model = Sequential()
from tensorflow.keras.layers import Dense
model.add(Dense(units=512, activation='relu', input_shape=(784,)))
model.add(Dense(units = 512, activation='relu'))
model.add(Dense(units = 10, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train,
                    epochs=5,
                    verbose=1,
                    validation_data=(x_test, y_test))
```

One-hot编码

```python
def label2OH(y, D_out):
  N = y.shape[0]
  OH = np.zeros((N, D_out))
  OH[np.arange(N), y] = 1
  return OH

def OH2label(OH):
  if(torch.is_tensor(OH)):
  	y = OH.argmax(dim=1)
  else:
  	y = OH.argmax(axis=1)
  return y
```


Image Classification of an American Sign Language Dataset

```python
import pandas as pd
train_df = pd.read_csv("asl_data/sign_mnist_train.csv")
test_df = pd.read_csv("asl_data/sign_mnist_test.csv")
train_df.head()

y_train = train_df['label']
y_test = test_df['label']
del train_df['label']
del test_df['label']

x_train = train_df.values
x_test = test_df.values

import matplotlib.pyplot as plt
plt.figure(figsize=(40,40))

num_images = 20
for i in range(num_images):
    row = x_train[i]
    label = y_train[i]
    
    image = row.reshape(28,28)
    plt.subplot(1, num_images, i+1)
    plt.title(label, fontdict={'fontsize': 30})
    plt.axis('off')
    plt.imshow(image, cmap='gray')
    
x_train = x_train / 255
x_test = x_test / 255

import tensorflow.keras as keras
num_classes = 25
```

CNN

```python
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization

num_classes = 25

model = Sequential()
model.add(Conv2D(75 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (28,28,1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(50 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(25 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Flatten())
model.add(Dense(units = 512 , activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(units = num_classes , activation = 'softmax'))
```

data augmentation

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images horizontally
        vertical_flip=False)  # Don't randomly flip images vertically

datagen.fit(x_train)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(datagen.flow(x_train,y_train, batch_size=32), # Default batch_size is 32. We set it here for clarity.
          epochs=20,
          steps_per_epoch=len(x_train)/32, # Run same number of steps we would if we were not using a generator.
          validation_data=(x_test, y_test))

model.save('asl_model')
model = keras.models.load_model('asl_model')

from tensorflow.keras.preprocessing import image as image_utils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_image(image_path):
    image = mpimg.imread(image_path)
    plt.imshow(image)
def predict_letter(file_path):
    show_image(file_path)
    image = load_and_scale_image(file_path)
    image = image_utils.img_to_array(image)
    image = image.reshape(1,28,28,1) 
    image = image/255
    prediction = model.predict(image)
    # convert prediction to letter
    predicted_letter = dictionary[np.argmax(prediction)]
    return predicted_letter

from tensorflow.keras.applications.vgg16 import preprocess_input
image = preprocess_input(image)
from tensorflow.keras.applications.vgg16 import decode_predictions
print('Predicted:', decode_predictions(predictions, top=3))
```



**Transfer Learning**

[NGC](https://ngc.nvidia.com/catalog/models?orderBy=modifiedDESC&pageNumber=0&query=&quickFilter=models&filters=)

[Keras Application](https://keras.io/api/applications/#available-models)

```python
from tensorflow import keras
base_model = keras.applications.VGG16(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(224, 224, 3),
    include_top=False)
base_model.trainable = False
inputs = keras.Input(shape=(224, 224, 3))
# Separately from setting trainable on the model, we set training to False 
x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
# A Dense classifier with a single unit (binary classification)
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
# Important to use binary crossentropy and binary accuracy as we now have a binary classification problem
model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True), metrics=[keras.metrics.BinaryAccuracy()])

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# create a data generator
datagen = ImageDataGenerator(
        samplewise_center=True,  # set each sample mean to 0
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False) # we don't expect Bo to be upside-down so we will not flip vertically

# load and iterate training dataset
train_it = datagen.flow_from_directory('presidential_doggy_door/train/', 
                                       target_size=(224, 224), 
                                       color_mode='rgb', 
                                       class_mode='binary', 
                                       batch_size=8)
# load and iterate test dataset
test_it = datagen.flow_from_directory('presidential_doggy_door/test/', 
                                      target_size=(224, 224), 
                                      color_mode='rgb', 
                                      class_mode='binary', 
                                      batch_size=8)

model.fit(train_it, steps_per_epoch=12, validation_data=test_it, validation_steps=4, epochs=20)
```

finetune

```python
# Unfreeze the base model
base_model.trainable = True

# It's important to recompile your model after you make any changes
# to the `trainable` attribute of any inner layer, so that your changes
# are taken into account
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate = .00001),  # Very low learning rate
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])
model.fit(train_it, steps_per_epoch=12, validation_data=test_it, validation_steps=4, epochs=10)
```



**headline generator**

[embedding layer](https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/)

[LSTM](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

[Adam optimizer](https://medium.com/datadriveninvestor/overview-of-different-optimizers-for-neural-networks-e0ed119440c3)

[pretrained word embedding](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html), [GPT2](https://openai.com/blog/better-language-models/), [BERT](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html)

```python
import os 
import pandas as pd

nyt_dir = 'nyt_dataset/articles/'

all_headlines = []
for filename in os.listdir(nyt_dir):
    if 'Articles' in filename:
        # Read in all of the data from the CSV file
        headlines_df = pd.read_csv(nyt_dir + filename)
        # Add all of the headlines to our list
        all_headlines.extend(list(headlines_df.headline.values))
# Remove all headlines with the value of "Unknown"
all_headlines = [h for h in all_headlines if h != "Unknown"]
len(all_headlines)

from tensorflow.keras.preprocessing.text import Tokenizer
# Tokenize the words in our headlines
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_headlines)
total_words = len(tokenizer.word_index) + 1
print('Total words: ', total_words)

# Print a subset of the word_index dictionary created by Tokenizer
subset_dict = {key: value for key, value in tokenizer.word_index.items() \
               if key in ['a','man','a','plan','a','canal','panama']}
print(subset_dict)
tokenizer.texts_to_sequences(['a','man','a','plan','a','canal','panama'])

# Convert data to sequence of tokens 
input_sequences = []
for line in all_headlines:
    # Convert our headline into a sequence of tokens
    token_list = tokenizer.texts_to_sequences([line])[0]
    
    # Create a series of sequences for each headline
    for i in range(1, len(token_list)):
        partial_sequence = token_list[:i+1]
        input_sequences.append(partial_sequence)
print(tokenizer.sequences_to_texts(input_sequences[:5]))
input_sequences[:5]

# Convert data to sequence of tokens 
input_sequences = []
for line in all_headlines:
    # Convert our headline into a sequence of tokens
    token_list = tokenizer.texts_to_sequences([line])[0]
    
    # Create a series of sequences for each headline
    for i in range(1, len(token_list)):
        partial_sequence = token_list[:i+1]
        input_sequences.append(partial_sequence)

print(tokenizer.sequences_to_texts(input_sequences[:5]))
input_sequences[:5]
input_sequences

# padding sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
# Determine max sequence length
max_sequence_len = max([len(x) for x in input_sequences])
# Pad all sequences with zeros at the beginning to make them all max length
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
input_sequences[0]

from tensorflow.keras import utils
# Predictors are every word except the last
predictors = input_sequences[:,:-1]
# Labels are the last word
labels = input_sequences[:,-1]
labels = utils.to_categorical(labels, num_classes=total_words)


from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

# Input is max sequence length - 1, as we've removed the last word for the label
input_len = max_sequence_len - 1 
model = Sequential()
# Add input embedding layer
model.add(Embedding(total_words, 10, input_length=input_len))
# Add LSTM layer with 100 units
model.add(LSTM(100))
model.add(Dropout(0.1))
# Add output layer
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

```

```python
tf.keras.preprocessing.text.Tokenizer(
    num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True,
    split=' ', char_level=False, oov_token=None, document_count=0, **kwargs
)
```

```python
def predict_next_token(seed_text):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    prediction = model.predict_classes(token_list, verbose=0)
    return prediction

prediction = predict_next_token("today in new york")
prediction
tokenizer.sequences_to_texts([prediction])

def generate_headline(seed_text, next_words=1):
    for _ in range(next_words):
        # Predict next token
        prediction = predict_next_token(seed_text)
        # Convert token to word
        next_word = tokenizer.sequences_to_texts([prediction])[0]
        # Add next word to the headline. This headline will be used in the next pass of the loop.
        seed_text += " " + next_word
    # Return headline as title-case
    return seed_text.title()
  
seed_texts = [
    'washington dc is',
    'today in new york',
    'the school district has',
    'crime has become']
for seed in seed_texts:
    print(generate_headline(seed, next_words=5))
```



### Fundamentals of Deep Learning for MultiGPUs -- Nvidia

* 与梯度下降法不同，随机梯度下降法并不使用整个数据集而是使用较小的数据子集（称为一个批次，即batch；其大小称为 batch size）来计算损失函数。这对我们算法的性能有着深远的影响。由于每个批次里的数据是从数据集里随机抽取的，所以每个批次的数据集都不相同。即使对于同一组权重，这些批次的数据集也会提供不同的梯度，引入一定程度的噪声
* 这种噪声实际上是非常有益的，因为它所产生的极小值的数学特性与梯度下降大相径庭。这在多 GPU 训练问题中之所以重要，是因为通过增加参与训练过程的 GPU 数量，我们实际上加大了批量（batch size），而这会导致减少有益的噪声

```python
# This section generates the training dataset as defined by the variables in the section above.
x = np.random.uniform(0, 10, n_samples)
y = np.array([w_gen * (x + np.random.normal(loc=mean_gen, scale=std_gen, size=None)) + b_gen for x in x])

# Create the placeholders for the data to be used.
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

# Create our model variables w (weights; this is intended to map to the slope, w_gen) and b (bias; this maps to the intercept, b_gen).
# For simplicity, we initialize the data to zero.
w = tf.Variable(0.0, name="weights")
b = tf.Variable(0.0, name="bias")

# Define our model. We are implementing a simple linear neuron as per the diagram shown above.
Y_predicted = w * X + b

# Define a gradient descent optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

# Define the maximum number of times we want to process the entire dataset (the number of epochs).
# In practice we won't run this many because we'll implement an early stopping condition that
# detects when the training process has converged.
max_number_of_epochs = 1000

# We still store information about the optimization process here.
loss_array = []
b_array = []
w_array = []
    
with tf.Session() as sess:
    # Initialize the necessary variables
    sess.run(tf.global_variables_initializer())
    # Print out the parameters and loss before we do any training
    w_value, b_value, loss_value = sess.run([w, b, loss], feed_dict={X: x, Y: y})
    print("Before training: w = {:4.3f}, b = {:4.3f}, loss = {:7.3f}".format(w_value, b_value, loss_value))
    print("")
    print("Starting training")
    print("")
    # Start the training process
    for i in range(max_number_of_epochs):
        # Use the entire dataset to calculate the gradient and update the parameters
        sess.run(optimizer, feed_dict={X: x, Y: y})
        # Capture the data that we will use in our visualization
        w_value, b_value, loss_value = sess.run([w, b, loss], feed_dict={X: x, Y: y})
        w_array.append(w_value)
        b_array.append(b_value)
        loss_array.append(loss_value)
        # At the end of every few epochs print out the learned weights
        if (i + 1) % 5 == 0:
            print("Epoch = {:2d}: w = {:4.3f}, b = {:4.3f}, loss = {:7.3f}".format(i+1, w_value, b_value, loss_value))
        # Implement your convergence check here, and exit the training loop if
        # you detect that we are converged:
        if FIXME: # TODO
            break
    print("")
    print("Training finished after {} epochs".format(i+1))
    print("")
    
    print("After training: w = {:4.3f}, b = {:4.3f}, loss = {:7.3f}".format(w_value, b_value, loss_value))
```

```python
# adjust batch size
batch_size = 32
num_batches_in_epoch = (n_samples + batch_size - 1) // batch_size
```



研究训练速度和 batch_size 的关系

* 非常小或非常大的批量对于模型训练的收敛来说可能不是的最佳选择（非常小的批量带来的噪声往往过于嘈杂而无法使模型充分收敛到损失函数的最小值，而非常大的批量则往往造成训练的早期阶段就发散）
* 观察到大batch size的val_acc和acc很接近，不容易过拟合，但后期准确度效果提升缓慢
* Machine-Learning/GPU_training_batch_size.py 



多GPU训练

```shell
# CPU training
CUDA_VISIBLE_DEVICES= python fashion_mnist.py --epochs 3 --batch-size 512
# GPU training
horovodrun -np $num_gpus python fashion_mnist.py --epochs 3 --batch-size 512
```

* [Horovod](https://github.com/horovod/horovod)是一种最初由[Uber开发](https://eng.uber.com/horovod/)的开源工具，旨在满足他们许多工程团队对更快的深度学习模型训练的需求。它是跨框架的分布式深度学习库，支持多种框架、高性能算法、高性能网络（RDMA、GPUDirect），也是分布式训练方法不断发展的生态系统（包括[Distributed TensorFlow](https://github.com/tensorflow/examples/blob/master/community/en/docs/deploy/distributed.md)) 的一部分。Uber开发的这种解决方案利用[MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface)进行分布式进程间通信，并利用[NVIDIA联合通信库（NCCL）](https://developer.nvidia.com/nccl)，以高度优化的方式实现跨分布式进程和节点的平均值计算。 由此产生的Horovod软件包实现了它的目标：仅需进行少量代码修改和直观的调试即可在多个GPU和多个节点上扩展深度学习模型的训练。

  自2017年开始实施以来，Horovod已显著成熟，将其支持范围从TensorFlow扩展到了Keras，PyTorch和Apache MXNet。 Horovod经过了广泛的测试，迄今已用于一些最大的深度学习训练当中。例如，在[Summit系统上支持 **exascale** 深度学习，可扩展到 **27,000多个V100 GPU**](https://arxiv.org/pdf/1810.01993.pdf)

  * 支持多种框架

```python
import horovod.tensorflow as hvd
import horovod.keras as hvd
import horovod.tensorflow.keras as hvd
import horovod.torch as hvd
import horovod.mxnet as hvd
```

Horovod与MPI的渊源

* Horovod与MPI具有非常深厚的联系。对于熟悉MPI编程的程序员来说，您对通过Horovod实现的分布式模型训练会感到非常熟悉。对于那些不熟悉MPI编程的人来说，简短地讨论一下Horovod或MPI分布式进程所需的一些约定和注意事项是值得的。
* 与MPI一样，Horovod严格遵循[单程序多数据（SPMD）范例](https://en.wikipedia.org/wiki/SPMD)，即在同一文件或程序中实现多个进程的指令流。由于多个进程并行执行代码，因此我们必须注意[竞赛条件](https://en.wikipedia.org/wiki/Race_condition)以及这些进程间的同步。
  * [Horovod and Model Parallelism](https://github.com/horovod/horovod/issues/96)
* Horovod为执行程序的每个进程分配一个唯一的数字ID或**rank**（来自MPI的概念）。rank是可以通过编程的方式获得的。通过以编程方式在代码中标识进程的rank，我们可以进一步采取以下步骤：

  * 将该进程固定到自己的专属GPU上。
  * 使用单个rank来广播需要所有ranks统一使用的值。
  * 利用单个rank收集所有ranks产生的值和/或计算它们的均值。
  * 利用一个rank来记录或写入磁盘。

![horovod-rank](Machine-Learning/horovod-rank.png)

```python
# 同步初始状态的几种方式
# Method 1
callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
model.fit_generator(train_iter,
                    steps_per_epoch=len(train_iter) // hvd.size(),
                    callbacks=callbacks, ...)
# Method 2
hooks = [hvd.BroadcastGlobalVariablesHook(0)]
with tf.train.MonitoredTrainingSession(hooks=hooks, …) as sess:
# Method 3
bcast_op = hvd.broadcast_global_variables(0) sess.run(bcast_op)

# 只由一个worker保留检查点
ckpt_dir = "/tmp/train_logs" if hvd.rank() == 0 else None
with tf.train.MonitoredTrainingSession(checkpoint_dir=ckpt_dir, …) as sess:
```

* 数据分区的方式：先洗牌再分区，workers按分区顺序读取；先洗牌，单worker从整个数据集随机读取

```shell
在 4 个有 4 块 GPU 卡的节点上运行:
$ mpirun -np 16 -H server1:4,server2:4,server3:4,server4:4 -bind-to none -map-by slot -mca pml ob1 -mca btl openib -mca btl_tcp_if_include eth0 \
-x NCCL_DEBUG=INFO -x NCCL_SOCKET_IFNAME=eth0 -x LD_LIBRARY_PATH -x ...\
python train.py
```



### 术语

NLU: Natural Language Understanding

### TODO

* 调参
  * https://github.com/google-research/tuning_playbook

* 传统关键词检索
  * https://www.elastic.co/cn/blog/implementing-academic-papers-lessons-learned-from-elasticsearch-and-lucene
* 对比学习
  * [Constrastive Learning: MoCo and SimCLR](https://mp.weixin.qq.com/s/v5p9QA3vDl-WTF3-7shp4g)