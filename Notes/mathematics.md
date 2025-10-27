# Mathematics

## 基础运算

### 进制

* 关于10进制有效数字位数可用下面这个公式来估算：
  * $$n \approx log_{10}(2^m) = m \times log_{10}(2) \approx m \times 0.3010$$

### 位运算

* 对于一个数 X 和一个2的幂 P ，$$X \% P = X \& (P - 1)$$





## 概率论与数理统计

### 正态分布

- 约 68.27% 的数值落在 μ ± 1σ 的范围内。
- 约 95.45% 的数值落在 μ ± 2σ 的范围内。
- 约 99.73% 的数值落在 μ ± 3σ 的范围内。

#### 霍夫丁不等式 Hoeffding Inequality

* UCB的公式是基于霍夫丁不等式（Hoeffding Inequality）推导而来的。

  * 假设有N个范围在0到1的独立有界随机变量，$$X_1, X_2, \cdots, X_N$$，那么这N个随机变量的经验期望为$$ \bar{X} = \frac{X_1 + \cdots + X_N}{N} $$
  * 满足如（式5 - 4）所示的不等式：$$ P(|\bar{X}-E[\bar{X}]| \geq \epsilon) \leq e^{-2N\epsilon^2} \quad (式5 - 4) $$

  * 那么，霍夫丁不等式和UCB的上界有什么关系呢？令 $$\epsilon = \sqrt{\frac{2\log t}{N}}$$，并代入（式5 - 4），可将霍夫丁不等式转换成如（式5 - 5）所示的形式：
    $$ P\left(|\bar{X}-E[\bar{X}]| \geq \sqrt{\frac{2\log t}{N}}\right) \leq t^{-4} \quad (式5 - 5) $$

  * 从（式5 - 5）中可以看出，如果选择UCB的上界是 $$\sqrt{\frac{2\log t}{N}}$$ 的形式，那么$$\bar{X}$$ 的均值与$$\bar{X}$$ 的实际期望值的差距在上界之外的概率非常小，小于$$t^{-4}$$，这就说明采用UCB的上界形式是严格且合理的。 --> UCB根据变量的上界进行exploration

### 切比雪夫不等式

* 核心思想：即使我们对数据的分布一无所知（它完全可以不是正态分布），切比雪夫不等式也为我们提供了一个关于数据偏离均值的概率的“下限保证”。

* 切比雪夫不等式是统计学中一个非常强大的定理。它指出，对于任何一个具有有限期望 μ 和有限非零方差 σ² 的随机变量 X ，对于任何实数 k > 0 ，以下不等式成立：
  * P(|X - μ| ≥ kσ) ≤ 1/k²
    * 这个公式描述的是 数据落在 k 个标准差之外 的概率。我们通常更关心它落在 范围之内 的概率，所以我们使用它的等价形式：
  * P(|X - μ| < kσ) ≥ 1 - 1/k²

* 这个公式的强大之处在于，它对分布形状没有任何要求。

* 让我们来看几个 k （对应代码中的 std_scale ）的例子：

  - 当 k = 2 ( std_scale = 2 )： P(|X - μ| < 2σ) ≥ 1 - 1/4 = 0.75 。
    - 数学含义 ：无论数据是什么鬼样子， 至少有 75% 的数据点落在均值的 2 个标准差范围内。

  - 当 k = 3 ( std_scale = 3 )： P(|X - μ| < 3σ) ≥ 1 - 1/9 ≈ 0.889 。
    - 数学含义 ： 至少有 88.9% 的数据点落在均值的 3 个标准差范围内。

  - 当 k = 4 ( std_scale = 4 )： P(|X - μ| < 4σ) ≥ 1 - 1/16 ≈ 0.9375 。
    - 数学含义 ： 至少有 93.75% 的数据点落在均值的 4 个标准差范围内。

### beta分布

* beta分布是伯努利分布的共轭先验分布
  * **共轭先验分布的概念**：根据先验分布计算后验分布，后验分布和先验分布属于同一个分布族
  * 应用：Thompson sampling

### 聚类 & 相似度

* Silhouette coefficient（轮廓系数）是一种用于评估聚类效果的指标，它综合考虑了聚类的紧密性和分离性，能有效衡量一个样本与其所属聚类以及相邻聚类之间的关系，其具体信息如下：
  * **定义与计算方式**    - 对于样本 $$i$$，设 $$a(i)$$ 为样本 $$i$$ 到其所属聚类 $$C$$ 中其他样本的平均距离，反映了样本在其所属聚类内的紧密程度。计算方式为 $$a(i)=\frac{1}{|C|-1}\sum_{j\in C,j\neq i}d(i,j)$$，其中 $$d(i,j)$$ 是样本 $$i$$ 和 $$j$$ 之间的距离度量（如欧几里得距离），$$|C|$$ 是聚类 $$C$$ 中的样本数量。    - 设 $$b(i)$$ 为样本 $$i$$ 到与其相邻最近聚类 $$C'$$ 中所有样本的平均距离，衡量了样本与其他聚类的分离程度。通过计算样本 $$i$$ 到所有其他聚类的平均距离，并取最小值作为 $$b(i)$$。    - 样本 $$i$$ 的轮廓系数 $$s(i)$$ 计算公式为 $$s(i)=\frac{b(i)-a(i)}{\max\{a(i),b(i)\}}$$，其取值范围在 $$[-1,1]$$ 之间。_
  * _ - **取值意义**    - 当 $$s(i)$$ 接近 1 时，表示样本 $$i$$ 与所属聚类内的样本距离较近，且与相邻聚类的样本距离较远，说明聚类效果较好，样本被合理地划分到了当前聚类中。    - 当 $$s(i)$$ 接近 0 时，表明样本 $$i$$ 处于两个聚类的边界附近，其所属聚类可能不太明确，聚类效果有待提高。    - 当 $$s(i)$$ 接近 -1 时，意味着样本 $$i$$ 可能被错误地划分到了当前聚类，它与相邻聚类的样本更为相似，聚类效果较差。 
  * - **整体评估**：通常会计算数据集中所有样本的轮廓系数，并取平均值作为整个数据集的轮廓系数。该平均值越高，说明聚类的整体质量越好；反之，则表示聚类效果不佳，可能需要调整聚类算法的参数或尝试其他聚类方法。例如，在一个包含 $$n$$ 个样本的数据集上，整体轮廓系数 $$S=\frac{1}{n}\sum_{i = 1}^{n}s(i)$$。 

* [Jaccard相似度](https://zh.wikipedia.org/wiki/%E9%9B%85%E5%8D%A1%E5%B0%94%E6%8C%87%E6%95%B0)：是用于比较[样本](https://zh.wikipedia.org/wiki/样本)集的相似性与[多样性](https://zh.wikipedia.org/wiki/多样性指数)的统计量，雅卡尔系数能够量度有限样本集合的相似度，其定义为两个集合[交集](https://zh.wikipedia.org/wiki/交集)大小与[并集](https://zh.wikipedia.org/wiki/并集)大小之间的比例
  * 文本相似度
* KL散度
  * $$D_{KL}(P||Q)=\sum_{i}P(i)\log\frac{P(i)}{Q(i)}$$
    * 衡量的是用分布Q来近似分布P时所损失的信息,有向的度量
  

## 博弈论

### 博弈论与机器学习

* Milgrom, Paul R., and Steven Tadelis.How artificial intelligence and machine learning can impact market design. No. w24282. National Bureau of Economic Research, 2018.
* https://cloud.tencent.com/developer/article/1530214



## A Course in Game Theory - Martin J. Osborne and Ariel Rubinstein

![image-20240720183436637](./Mathematics/game-theory-structure-1506147.png)

### Chpt 1: Introduction

* Game Theory

  * The basic assumptions that underlie the theory are that decision-makers pursue well-defined exogenous objectives (they are **rational**) and take into account their knowledge or expectations of other decision-makers’ behavior (they **reason strategically**).
  * 对现实的指导
    * the theory of Nash equilibrium (Chapter 2) has been used to study oligopolistic and politi- cal competition. 
    * The theory of mixed strategy equilibrium (Chapter 3) has been used to explain the distributions of tongue length in bees and tube length in flowers. 
    * The theory of repeated games (Chapter 8) has been used to illuminate social phenomena like threats and promises. 
    * The theory of the core (Chapter 13) reveals a sense in which the outcome of trading under a price system is stable in an economy that contains many agents.

* Games and Solutions

  * Noncooperative and Cooperative Games
    * Noncooperative：Part 1、2、3； Cooperative：Part 4
  * Strategic Games and Extensive Games
    * 1； 2、3
  * Games with Perfect and Imperfect Information
    * 2；3

* Game Theory and the Theory of Competitive Equilibrium

  * competitive reasoning（经济学中的概念）只关注外部环境变量来做决策，不关注其他agent的决策

* Rational Behavior

  * A set A of actions from which the decision-maker makes a choice.
  * A set C of possible consequences of these actions.
  * A consequence function g:A → C that associates a consequence with each action.
    *  if the consequence function is **stochastic** and known to the decision-maker
      * **maximizes the expected value**
    * If not known: 引入状态空间，he is assumed to choose an action a that maximizes the expected value of **u(g(a,ω))**
  * A preference relation (a complete transitive reflexive binary relation) 􏰵 on the set C.
    * utility function

* The Steady State and Deductive Interpretations

  * **The Steady State** treats a game as a model designed to explain some regularity observed in a family of similar situations.
    * Each participant “knows” the equilibrium and tests the optimality of his behavior given this knowledge, which he has acquired from his long experience.
  * **Deductive Interpretations** treats a game in isolation, as **a “one-shot” event**, and attempts to infer the restrictions that rationality imposes on the out- come; it assumes that each player deduces how the other players will behave **simply from principles of rationality**

* Bounded Rationality

  * asymmetry between individuals in their abilities
  * Modeling asymmetries in abilities and in perceptions of a situation by different players is a fascinating challenge for future research

* Terminology and Notation

  *  increasing、nondecreasing function

  * concave

  * arg maxx∈X f (x) the set of maximizers of f

  * Throughout we use N to denote the set of players. We refer to a collection of values of some variable, one for each player, as a profile       (xi)

    * ![image-20240720191434381](./Mathematics/game-theory-xi-1506147.png)

  * A binary relation on set的可能的性质: complete、reflexive、transitive

    * **A preference relation is a complete reflexive transitive binary relation.**

  * A preference relation的性质

    * 连续性：序列收敛

    * quasi-concave

      * ![image-20240720191747144](./Mathematics/game-theory-quasi-concave-1506147.png)

      * 在某条线上的点的连续性
      * https://en.wikipedia.org/wiki/Quasiconvex_function
      * ![image-20240720192146282](./Mathematics/quasi-convex-1506147.png)

  *  Pareto efficient 和 strongly Pareto efficient 
    *  Pareto efficient：没被碾压
    * strongly Pareto efficient ：没人有任何方面对我有优势
  * **A probability measure μ** on a finite (or countable) set X is an additive function that associates a nonnegative real number with every subset of X

### Part I: Strategic Games

* strategic game = game in normal form
  * This model specifies for each player a set of possible actions and a preference ordering over the set of possible action profiles.

### Chpt 2: Nash Equilibrium

* Strategic Games
  * A strategic game is a model of interactive decision-making in which each decision-maker chooses his plan of action **once and for all,** and these **choices are made simultaneously**
  * ![image-20240720225501522](./Mathematics/game-theory-strategic-game-1506147.png)
  * finite/非有限
  * 分析：
    * the range of application of the model is limited by the requirement that we associate with each player a preference relation
    * 限定过于宽松，难以得出重大结论
    * a player can form his expectation of the other players’ behavior
      * A sequence of plays of the game can be modeled by a strategic game **only if there are no strategic links between the plays.**
      * The model of a repeated game discussed in Chapter 8 deals with series of strategic interactions in which such intertemporal links do exist.
  * 拓展：
    * To do so we introduce **a set C of consequences**, a function g:A → C that associates consequences with action profiles, and a profile (􏰵∗i ) of preference rela- tions over C.
    * a function g: A × Ω → C with the interpretation that g(a, ω) is the consequence when the action profile is a ∈ A and the realization of the random variable is ω ∈ Ω
      * 引入之后，**the lottery over C induced by g(a,·) is at least as good** according ...，似乎更严格了
  * payoff function
    * denote：⟨N, (Ai), (ui)⟩
    * 2 player时的矩阵表示：the convention is that the row player is player 1 and the column player is player 2

* Nash Equilibrium
  * ![image-20240721000029501](./Mathematics/game-theory-nash-equil-1506147.png)
  * 分析：
    * This notion captures a steady state of the play of a strategic game in which each player holds the correct expectation about the other players’ behavior and acts rationally.
    * It does not attempt to examine the process by which a steady state is reached.
  * ![image-20240721000209969](./Mathematics/game-theory-nash2-1506147.png)
  * 延伸：一种可能的求解纳什均衡的方式，N个best-response function(如果是单值)

* Examples

  * Example 15.3 (Bach or Stravinsky? (BoS))

    * 元素1在列中最大、元素2在行中最大

  * 囚徒困境

  * 鹰鸽博弈

    * 是一种Symmetric games，但只存在 asymmetric equilibria

  * Matching Pennies

    * Such a game, in which the interests of the players are diametrically opposed, is called **“strictly competitive”**. The game Matching Pennies has no Nash equilibrium.

  * Exercise 18.2 first-price auction

    * 注意lowest index的约束

  * **in a second price auction** the bid vi of any player i is a **weakly dominant action**

    * An equilibrium in which player j obtains the good is that in which b1 < vj, bj > v1, and

      bi = 0 for all players i ∈/ {1,j}.

  *  Example 18.4 (A war of attrition)

  * Example 18.6 (A location game)

    * There is no equilibrium in which all three players become candidates

* Note：

  * Every SNE is weakly Pareto-efficient https://en.wikipedia.org/wiki/Strong_Nash_equilibrium

* Existence of a Nash Equilibrium

  * 表述为 a∗ ∈ B(a∗)
  * Lemma 20.1 (Kakutani’s fixed point theorem角谷静夫不动点定理)
  * ![image-20240721011951283](./Mathematics/game-theory-kakutani-1506147.png)

  * ![image-20240721013351172](./Mathematics/game-theory-nash-existence-1506147.png)
    * 注意不适用于有限actions集合的情况

* Strictly Competitive Games

  * 定义strictly competitive

  * for a strictly competitive game that possesses a Nash equilibrium, a pair of actions is a Nash equilibrium if and only if the action of each player is a maxminimizer

  * for strictly competitive games that possess Nash equilibria all equilibria yield the same payoffs.

  * ![image-20240721020542300](./Mathematics/game-theory-strictly-1506147.png)

    * part c提供一种场景的解法
    * by parts (a) and (c), **the Nash equilibria of a strictly competitive game are interchangeable**: if (x, y) and (x′, y′) are equilibria then so are (x, y′) and (x′, y).
    * Thus in any game (whether or not it is strictly competitive) the payoff that player 1 can guarantee herself is at most the amount that player 2 can hold her down to
      * ![image-20240721022026163](./Mathematics/image-20240721022026163-1506147.png)
      * 需要额外条件来确保存在纳什均衡

    * Part b: this payoff, the equilibrium payoff of player 1, is **the value of the game**
    * strictly competitive game中，增加payoff一定不会让value有损失；减少操作空间只可能让value有损失

* Bayesian Games: Strategic Games with Imperfect Information

  * 定义：Definition 25.1 A Bayesian game
    * We model the players’ uncertainty about each other by introducing a set Ω of possible “states of nature”, each of which is a description of all the players’ relevant characteristics. For convenience we assume that Ω is finite. Each player i has a prior belief about the state of nature given by a probability measure pi on Ω.
    * signal functions
    * 先验信号 ![image-20240721050519922](./Mathematics/image-20240721050519922-1506147.png)
  * 分析：
    * player has imperfect information about the state of nature
  * 用处：
    * a state of nature is a profile of parameters of the players’ preferences (for example, profiles of their valuations of an ob- ject)
    * player is uncertain about what the others know.（section 2.6.3
  * bayesian game的纳什均衡
    * ![image-20240721051935708](./Mathematics/image-20240721051935708-1506147.png)
  * Exercise 28.1 (An exchange game）
    * in any Nash equilibrium the highest prize that either player is willing to exchange is the smallest possible prize.
  * 贝叶斯纳什均衡的习题例子：https://www.cnblogs.com/haohai9309/p/17753112.html
  * A Bayesian game can be used to model not only situations in which each player is uncertain about the other players’ payoffs, as in Exam- ple 27.1, but also situations in which each player is **uncertain about the other players’ knowledge.**
    * we can let Ω = Θ × (×i∈N Xi) be the state space and use the model of a Bayesian game to capture any situation in which players are uncertain not only about each other’s payoffs but also about each other’s beliefs.

### Chpt 3: Mixed, Correlated, and Evolutionary Equilibrium

* Mixed Strategy Nash Equilibrium



### Intro

* Game theory provides a framework based on the construction of rigorous models that describe situations of conflict and cooperation between *rational* decision makers.
* 大家有没有比较推荐的博弈论的书？ - 子不语D的回答 - 知乎
  https://www.zhihu.com/question/446554214/answer/1751901215
* 学习博弈论，从入门、进阶到精通，如何列书单？ - 食其的回答 - 知乎
  https://www.zhihu.com/question/20266302/answer/76445562

## ML Theory

* 神经网络的万能逼近公式
  * ![image-20250408202259016](./Mathematics/image-20250408202259016-1506147.png)（来源LHUC paper)
  * ![image-20251004022110555](./Mathematics/image-20251004022110555-1506147.png)
    * 每条边代表一个参数
  
* 神经网络扰动
  * [Hessian-based Analysis of Large Batch Training and Robustness to Adversaries](https://arxiv.org/pdf/1802.08241) 分析了Hessian特征值和神经网络扰动的关系
    * Q-Bert 计算最大特征值：通过两次反向传播计算Hv，[幂迭代法 Power Iteration](https://en.wikipedia.org/wiki/Power_iteration)
    * [Training Deep and Recurrent Networks with Hessian-Free Optimization](https://www.cs.toronto.edu/~jmartens/docs/HF_book_chapter.pdf) 完整综述
  * 应用：量化分析，Q-Bert
* 2024 诺贝尔物理学奖授予人工神经网络机器学习，为什么会颁给 AI 领域？ - SIY.Z的回答 - 知乎
  https://www.zhihu.com/question/777943030/answer/4508673022
  * RBM 受限玻尔兹曼机
  * sigmoid 函数有了自然的解释：玻尔兹曼分布下隐含层神经元激活的条件概率的激活函数。
* 大模型是不是能够稀疏化? 
  * 从物理和能量密度/信息密度的角度来看, 似乎是可以的. 
  * 但是从范畴论的角度, 特别是预层范畴的角度来看待,Dense的Foundation Model训练是必须要做的, 因为只有在相对Dense的模型结构上才能更好的捕获所有的态射. 

* TOPOS理论 在ML的应用
  * on the diagram of thought https://github.com/diagram-of-thought/diagram-of-thought



## ML Application —— Exercises in Machine Learning

https://arxiv.org/pdf/2206.13446

### Backpropagating计算

https://web.eecs.umich.edu/~justincj/teaching/eecs442/notes/linear-backprop.html

![image-20250331145615951](./Mathematics/image-20250331145615951-1506147.png)




好的，这是一个非常深刻的问题。要从数学上理解 VQ-VAE 中的 "variational"，我们必须先回到它的“父辈”——标准的 VAE（Variational Autoencoder），理解 "variational" 在那里意味着什么，然后再看 VQ-VAE 是如何继承、修改和“绕过”这个概念的。

### 解构 VQ-VAE：一个关于变分贝叶斯方法应用的特例

#### Part 1: 标准 VAE 中的 "Variational" 是什么？

一个标准的生成模型的目标是学习数据的真实分布 `p(x)`。我们通常引入一个隐变量 `z` 来实现这一点，假设数据 `x` 是由 `z` 生成的。我们希望最大化数据的边际似然 `p(x)`：

`p(x) = ∫ p(x|z) p(z) dz`

- `p(z)` 是先验分布，通常我们假设它是一个简单的分布，比如标准正态分布 `N(0, I)`。
- `p(x|z)` 是由解码器（Decoder）建模的似然，表示给定一个隐变量 `z`，生成数据 `x` 的概率。

这个积分通常是**无法直接计算的 (intractable)**，因为它需要在整个隐空间上进行积分。

**Variational Inference (变分推断) 的登场**:
既然无法直接优化 `p(x)`，我们就需要一个近似的方法。变分推断的核心思想是：我们引入一个**近似的后验分布 `q(z|x)`** 来模拟真实的、但无法计算的后验分布 `p(z|x)`。

- `p(z|x)`: 真实的后验。给定一个数据点 `x`，它对应的隐变量 `z` 的真实分布是怎样的？这个我们算不出来，因为它依赖于 `p(x)`。
- `q_φ(z|x)`: 近似的后验。我们用一个简单的、可参数化的分布（例如高斯分布）来近似它。编码器（Encoder）的工作就是学习这个分布的参数 `φ`（例如，给定 `x`，输出对应的高斯分布的均值 `μ` 和方差 `σ²`）。

我们的目标是让 `q_φ(z|x)` 尽可能地接近 `p(z|x)`。衡量两个分布之间距离的工具是 **KL 散度**。我们希望最小化 `KL(q_φ(z|x) || p(z|x))`。

经过一系列数学推导（这是变分推断的核心），可以证明，最小化这个 KL 散度等价于**最大化证据下界 (Evidence Lower Bound, ELBO)**：

`log p(x) ≥ L_ELBO = E_{z~q_φ(z|x)}[log p_θ(x|z)] - KL(q_φ(z|x) || p(z))`

这个 ELBO 就是 VAE 的损失函数（取负号后最小化）：
1.  **`E_{z~q_φ(z|x)}[log p_θ(x|z)]`**: **重构项 (Reconstruction Term)**。从编码器得到的分布 `q_φ` 中采样一个 `z`，然后让解码器 `p_θ` 尽力去重构出原始的 `x`。
    * 理解`p_θ` ：模型的参数化
2.  **`KL(q_φ(z|x) || p(z))`**: **正则化项 (Regularization Term)**。它是一个 KL 散度，惩罚近似后验 `q_φ` 与先验 `p(z)` 之间的差异。这迫使编码器产生的隐空间结构化，使其接近于一个标准正态分布，从而使得我们可以在生成新样本时方便地从 `N(0, I)` 中采样。

**小结**：标准 VAE 的 "Variational" 来自于**变分推断**，即使用一个参数化的分布 `q_φ(z|x)` 去近似真实的后验 `p(z|x)`，并通过最大化 ELBO 来同时优化模型的重构能力和隐空间的结构。

#### Part 2: VQ-VAE 如何“打破”了这个框架？

VQ-VAE 的核心创新是引入了一个**离散的**隐空间，通过一个码本（Codebook）`E = {e_1, e_2, ..., e_K}` 来实现。

其过程如下：
1.  编码器 `Encoder(x)` 输出一个连续的向量 `z_e(x)`。
2.  通过**最近邻查找**，找到码本 `E` 中与 `z_e(x)` 最接近的码本向量 `e_k`。
3.  将这个被**量化 (Quantized)** 的向量 `e_k` 送入解码器 `Decoder(e_k)` 来重构 `x`。

这里的关键在于第二步的“最近邻查找”：`k = argmin_j ||z_e(x) - e_j||²`。这个 `argmin` 操作是**不可微分的**。

这导致了标准 VAE 框架的两个关键部分失效了：
1.  **无法重参数化采样**: 我们不能再从一个连续的分布 `q_φ(z|x)` 中平滑地采样。选择是“硬性”的，要么是 `e_1`，要么是 `e_2`，没有中间状态。
2.  **KL 散度项消失**: 标准的 `KL(q_φ(z|x) || p(z))` 公式是为连续分布（如高斯分布）设计的，现在隐空间是离散的，这个公式不再适用。

那么，如果 VAE 的核心数学工具 ELBO 看起来被打破了，为什么它还叫 VQ-**VAE** 呢？

#### Part 3: 数学上，VQ-VAE 的 "Variational" 在哪里？

这是最精妙的部分。VQ-VAE 的作者在论文中给出了一个数学解释，将它重新纳入了变分推断的框架，但形式非常特殊。

> 感觉不难想到，只要把数学意义对上即可

1.  **后验分布 `q(z|x)` 的新定义**:
    在 VQ-VAE 中，后验分布 `q(z|x)` 变成了一个**确定性的离散分布**。对于给定的 `x`，编码和量化过程会唯一确定一个码本向量 `e_k`。所以，后验分布可以看作是一个 one-hot 分布：
    `q(z=e_k | x) = 1`  (如果 `e_k` 是 `z_e(x)` 的最近邻)
    `q(z=e_j | x) = 0`  (对于所有 `j ≠ k`)

2.  **先验分布 `p(z)` 的新定义**:
    先验分布 `p(z)` 不再是高斯分布，而是在 `K` 个码本向量上的一个**均匀离散分布**。也就是说，我们先验地认为任何一个码本向量被选中的概率都是相等的：
    `p(z=e_j) = 1/K`  (对于所有 `j = 1, ..., K`)

3.  **KL 散度项的坍缩**:
    现在我们来计算这个新的离散框架下的 KL 散度 `KL(q(z|x) || p(z))`：
    `KL = Σ_{j=1 to K} q(z=e_j | x) * log( q(z=e_j | x) / p(z=e_j) )`

    由于 `q(z|x)` 是一个 one-hot 分布，这个求和中只有一项（当 `j=k` 时）不为零：
    `KL = 1 * log( 1 / (1/K) ) = log(K)`

**结论：KL 散度项变成了一个常数 `log(K)`！**

在模型优化中，一个常数项的梯度为零，对参数更新没有任何贡献。因此，在 VQ-VAE 的最终损失函数中，**这个 KL 散度项被直接忽略了**。

#### Part 4: VQ-VAE 的实际损失函数

既然 KL 正则项在数学上“消失”了，那模型如何学习呢？VQ-VAE 设计了新的损失函数来替代 ELBO 的功能：

最大化`L = - log p(x | z_q(x)) + ||sg[z_e(x)] - e_k||² + β * ||z_e(x) - sg[e_k]||²`

1.  **`log p(x | z_q(x))`**: **重构损失**。本质是decoder，这和 VAE 中的重构项目的一样。
    * 比如 `C * ||x - x_hat||²`
2.  **`||sg[z_e(x)] - e_k||²`**: **码本损失 (Codebook Loss)**。`sg` 是 `stop_gradient` 算子。这个损失项用于更新码本向量 `e_k`，让它向编码器的输出 `z_e(x)` 靠拢。
3.  **`β * ||z_e(x) - sg[e_k]||²`**: **承诺损失 (Commitment Loss)**。这个损失项用于更新编码器。它鼓励编码器的输出 `z_e(x)` “承诺”于它所映射到的那个码本向量 `e_k`，防止编码器的输出空间无限增长而码本更新跟不上。**这可以被看作是 VAE 中 KL 散度项的替代品，起到了正则化编码器的作用。**
    * β = 0.25，比较鲁棒

同时，为了解决 `argmin` 不可导的问题，VQ-VAE 使用了**直通估计器 (Straight-Through Estimator, STE)**，在反向传播时，直接将解码器输入端 `e_k` 的梯度复制给编码器输出端 `z_e(x)`，从而让编码器可以被训练。

#### 关于 Diagonal Covariance

对角协方差 (Diagonal Covariance) : 这同样是一个为了 简化计算 而做出的关键假设。我们假设，给定数据 x ，其对应的隐变量 z 的各个维度是条件独立的。
- 数学意义 : q_φ(z|x) 的协方差矩阵是一个对角矩阵，非对角线元素为 0。这意味着 z 的联合概率密度可以分解为各个维度边缘概率密度的乘积： q(z_1, z_2, ...|x) = q(z_1|x) * q(z_2|x) * ... 。
- 实践意义 :
  - 编码器输出简化 : 编码器只需要为 z 的 d 个维度输出 d 个均值和 d 个方差，总共 2d 个数值，而不需要输出一个包含 d*(d+1)/2 个独立数值的完整协方差矩阵。
  - KL 散度可解析计算 : 这是最重要的一个好处。当 q_φ(z|x) 和 p(z) 都是对角协方差的高斯分布时，它们之间的 KL 散度有一个 解析解 (closed-form solution) ，可以直接用公式计算出来，而无需进行复杂的蒙特卡洛采样估计。
     - $$KL( N(μ, σ²) || N(0, I) ) = 0.5 * Σ (σ² + μ² - 1 - log(σ²))$$
     - 这个公式可以直接嵌入到损失函数中，并且是完全可微分的。如果没有这个解析解，VAE 的训练将变得极其困难。

#### 关于 Gaussian Reparametrisation Trick

这个技巧的核心思想是： **将随机性与模型的参数分离开**，从而打通梯度路径，可以通过构造源源不断的随机数据，去学习模型参数

我们不直接从 z ~ N(μ_φ(x), σ²_φ(x)) 中采样，而是通过以下方式构造 z ：

* 从一个 固定的、与模型参数无关的 标准正态分布中采样一个随机噪声 ε 。即 ε ~ N(0, I) 。

* 通过一个确定性的函数来生成 z ： z = μ_φ(x) + σ_φ(x) * ε



#### 总结

- **数学上的理解**: VQ-VAE 中的 "variational" 是一种**概念上的继承**。它在理论上可以被解释为一个特殊的 VAE，其后验是一个确定性的离散分布，先验是一个均匀的离散分布。在这种特殊情况下，ELBO 中的 KL 散度项坍缩成一个常数，因此在最终的优化目标中被省略。
- **实践上的理解**: VQ-VAE **用一个码本和承诺损失取代了标准 VAE 中的 KL 散度正则化**。它不再强制隐空间服从高斯分布，而是强制编码器的输出被“绑定”到一个共享的、离散的码本上，通过这种方式来正则化隐空间。

所以，称其为 "Variational" 是为了表明它与 VAE 的师承关系和作为生成模型的共同目标，尽管它为了引入离散隐空间，在数学实现上走了另一条巧妙的道路。
        





## 线性代数

*  $$E_{S}=E_CE_U$$    -> $$e_s^x={E_U}^Te_c^x$$
* 最小二乘法：
  * 考虑线性方程组 $$Ax = b$$，当该方程组无解（即 $$b$$ 不在 $$A$$ 的列空间中）时，我们希望找到一个 $$\hat{x}$$ 使得 $$\|Ax - b\|^2$$ 最小。
  * 此时，$$\hat{x}=(A^TA)^{-1}A^Tb$$，$$A\hat{x}=A(A^TA)^{-1}A^Tb$$
  * 即 $$A\hat{x}$$ 是 $$b$$ 在 $$A$$ 的列空间上的投影，$$A\hat{x}$$ 与 $$b$$ 的误差在所有可能的 $$Ax$$ 中是最小的。 

* $$E_U^T E_U=\left(U_1,U_2,U_3, ... U_u\right)\left(\begin{array}{c}{U_1}^T\\{U_2}^T\\{U_3}^T\\...\\{U_u}^T\end{array}\right) = \sum_{i=1}^uU_i{U_i}^T$$

### SVD、矩阵分解

#### Theory

* 求解SVD

  * $$A = U\Sigma V^T \Rightarrow A^T = V\Sigma U^T \Rightarrow A^T A = V\Sigma U^T U\Sigma V^T = V\Sigma^2 V^T$$
  * $$A^{-1}=(U \Sigma V^T)^{-1} = (V^T)^{-1} \Sigma^{-1} U^{-1} = V \Sigma^{-1} U^T$$

* [SVD分解(一)：自编码器与人工智能](https://spaces.ac.cn/archives/4208) —— 苏剑林

  * **不带激活函数的三层自编码器，跟传统的SVD分解是等价的。**

    * 我们降维，并不是纯粹地为了减少储存量或者减少计算量，而是**“智能”的初步体现**

  * SVD

    * (m+n)k < mn

  * 自编码器

    * 无视激活函数，只看线性结构。自编码器是希望训练一个f(x)=x的恒等函数，但是中间层节点做了压缩
    * ![image-20241229231515019](./Mathematics/image-20241229231515019-1506147.png)

  * 压缩即智能

    * 通过压缩后进行重建，能够挖掘数据的共性，即我们所说的规律，然后得到更泛化的结果

    * ![image-20241229231727074](./Mathematics/image-20241229231727074-1506147.png)

    * 我们通过SVD分解，原始的目的可能是压缩、降维，但复原后反而衍生出了更丰富、词汇量更多的结果。

    * ```
      将结巴分词的词表中所有频数不小于100的二字词选出来，然后以第一字为行，第二字为列，频数为值（没有这个词就默认记为0），得到一个矩阵。对这个矩阵做SVD分解，得到两个矩阵，然后把这两个矩阵乘起来（复原），把复原后的矩阵当作新的频数，然后对比原始矩阵，看看多出了哪些频数较高的词语。
      ```

  * 激活函数在这里的物理意义

    * 是对无关紧要元素的一种舍弃（截断），这种操作，我们在很久之前的统计就已经用到了（扔掉负数、扔掉小统计结果等）。
    * 当然，在神经网络中，激活函数有更深刻的意义，但在浅层的网络（矩阵分解）中，它给我们的直观感受，就是截断罢了。

* SVD的聚类含义

  * https://spaces.ac.cn/archives/4216
  * 转化为三个矩阵相乘
  * ![image-20241229233643160](./Mathematics/image-20241229233643160-1506147.png)

* word2vec是SVD

  * https://spaces.ac.cn/archives/4233

  * 词向量Embedding层 = one hot的全连接层

  * 在做情感分类问题时，如果有了词向量，想要得到句向量，最简单的一个方案就是直接对句子中的词语的词向量求和或者求平均，这约能达到85%的准确率。

    * FastText，多引入了ngram特征，来缓解词序问题，但总的来说，依旧是把特征向量求平均来得到句向量
    * 本质上是 **词袋模型**

  * word embedding和SVD的联系

    * distributed representation，即分布式表示，N维embedding
      * 设想开一个窗口（前后若干个词加上当前词，作为一个窗口），然后统计当前词的前后若干个词的分布情况，就用这个分布情况来表示当前词，而这个分布也可以用相应的NN维的向量来表示
    * SVD将distributed representation降维到k维
    * Word2Vec的一个CBOW （continuous bag-of-words）方案是，将前后若干个词的词向量求和，然后接一个NN维的全连接层，并做一个softmax来预测当前词的概率

  * word2vec和svd的区别

    * 1、Word2Vec的这种方案，可以看作是通过前后词来预测当前词，而自编码器或者SVD则是通过前后词来预测前后词；

      2、Word2Vec最后接的是softmax来预测概率，也就是说实现了一个非线性变换，而自编码器或者SVD并没有。



#### Application

* [矩阵分解在协同过滤推荐算法中的应用 ](https://www.cnblogs.com/pinard/p/6351319.html)
  * SVD，计算量大
    * 稀疏SVD
      * 平均值补全
      * 随机投影或Lanczos迭代

  * FunkSVD
    * ![image-20241229023812815](./Mathematics/image-20241229023812815-1506147.png)

  * BiasSVD
    * 考虑 评分系统平均分μ,第i个用户的用户偏置项,第j个物品的物品偏置项
  * SVD++算法在BiasSVD算法上进一步做了增强，这里它增加考虑用户的隐式反馈
    * ![image-20241229024446412](./Mathematics/image-20241229024446412-1506147.png)

* 文档矩阵、主题模型
  *  Aij表示第 i 个文档中第 j 个词的出现情况
     * 词频/ TF-IDF

### 特征值

* 特征值求解
  * [幂迭代法 Power Iteration](https://en.wikipedia.org/wiki/Power_iteration) 求解最大特征值
  * QR算法

#### 谱范数、次可乘性 --> 误差传播

* 矩阵的谱范数 (Spectral Norm)
  * 对于一个矩阵 $$A \in \mathbb{R}^{m \times n}$$，其 谱范数 （也称为2-范数，$$||A||_2$$）被定义为：
  * $$||A||_2 = \sup_{x \neq 0} \frac{||Ax||_2}{||x||_2}$$
  * 最大“拉伸”比例
* 谱范数也等于矩阵 $$A$$ 的 最大奇异值 (Largest Singular Value) ，记为 $$\sigma_{max}(A)$$。
  $$||A||_2 = \sigma_{max}(A) = \sqrt{\lambda_{max}(A^T A)}$$
  其中，$$\lambda_{max}(A^T A)$$ 表示矩阵 $$A^T A$$ 的最大特征值。
* 范数的次可乘性 (Submultiplicativity)
  * 如果对于任意两个可相乘的矩阵 $$A$$ 和 $$B$$，都满足不等式$$||AB|| \leq ||A|| \cdot ||B||$$

* 证明 :
  * 利用谱范数的定义（$$||Az||_2 \leq ||A||_2 \cdot ||z||_2$$）：$$||A(Bx)||_2 \leq ||A||_2 \cdot ||Bx||_2$$
  * ----> $$||AB||_2 \leq ||A||_2 \cdot ||B||_2$$

#### 层条件数 (Condition Number)  -> 误差扰动

* **定义**: 矩阵 $J$ 的条件数 $\kappa(J) = ||J||_2 \cdot ||J^{-1}||_2 = \frac{\sigma_{max}(J)}{\sigma_{min}(J)}$。它衡量了矩阵对输入扰动的敏感度，即最大拉伸与最大压缩的比率。

* **分析**:

  * $\kappa(J_l) \approx 1$ 表示该层是“良态的 (well-conditioned)”，对不同方向的误差分量处理方式相似。
  * $\kappa(J_l) \gg 1$ 表示该层是“病态的 (ill-conditioned)”，变换是高度“各向异性 (anisotropic)”的。它会沿某些方向（最大奇异向量方向）急剧放大误差，同时沿另一些方向（最小奇异向量方向）急剧缩小误差。
  * 在深度网络中，多个病态层的累积效应会导致训练极其不稳定，部分梯度爆炸而另一部分梯度消失同时发生。

* 证明：

  *  $$||J||_2 = \sigma {max}(J)$$。
    * $$||J|| 2^2 = \sup {x \neq 0} \frac{(Jx)^T(Jx)}{x^T x} = \sup {x \neq 0} \frac{x^T J^T J x}{x^T x}$$
    * 这个表达式 $$\frac{x^T (J^T J) x}{x^T x}$$ 是矩阵 $$J^T J$$ 的 瑞利商 (Rayleigh Quotient) 。根据瑞利商的性质，其最大值等于该矩阵的最大特征值。令 $$\lambda_{max}(J^T J)$$ 为矩阵 $$J^T J$$ 的最大特征值。$$||J||_2^2 = \lambda {max}(J^T J)$$
    * 根据奇异值的定义，矩阵 $$J$$ 的奇异值 $$\sigma_i(J)$$ 是矩阵 $$J^T J$$ 的特征值的平方根。因此，$$J$$ 的最大奇异值 $$\sigma_{max}(J)$$ 等于 $$\sqrt{\lambda_{max}(J^T J)}$$。

  * 证明 $$||J^{-1}||_2 = \frac{1}{\sigma {min}(J)}$$。 
    * 利用奇异值分解



### matrix vector derivatives for machine learning

> matrix vector derivatives for machine learning.pdf

### Cholesky decomposition

https://en.wikipedia.org/wiki/Cholesky_decomposition

* 分解 a [Hermitian](https://en.wikipedia.org/wiki/Hermitian_matrix) [positive-definite matrix](https://en.wikipedia.org/wiki/Positive-definite_matrix) A
* e.g.
  * 解方程：提升数值计算稳定性
  * non linear optimization
    * GPTQ

## 最优化、凸优化

### 算法

* hill climbing https://en.wikipedia.org/wiki/Hill_climbing
  * 多元函数，每次只改变一个输入
  * 容易陷入local maxima

### 应用

#### 飞行器精准制导

【NASA公开课：凸优化如何实现飞行器精准制导 | Behçet Açıkmeşe讲座【中英双语】-哔哩哔哩】 [NASA公开课：凸优化如何实现飞行器精准制导 | Behçet Açıkmeşe讲座【中英双语】_哔哩哔哩_bilibili](https://b23.tv/aMLplAp)



## 数学分析

### 一些定义

* 完备的度量空间是泛函分析中的重要概念，其核心性质是**所有柯西序列都收敛到该空间中的某个点**。
  * 核心在于确保 “所有潜在的极限点都存在于空间内”

### 压缩映射原理

![image-20250402221637163](./Mathematics/image-20250402221637163-1506147.png)

* [压缩映射与皮卡迭代的简单比较](https://www.bilibili.com/video/BV1pjAte2EwV )



## 微分方程 ODE

### 基础技巧

* [用特征线方法与傅里叶变换求解偏微分方程](https://www.bilibili.com/video/BV16JFTeVEGj)
* 皮卡迭代
  * ![image-20250402222144436](./Mathematics/image-20250402222144436-1506147.png)

### 应用

* [玻色爱因斯坦凝聚与Boltzmann-Nordheim方程](https://www.bilibili.com/video/BV1YwNHeSEbp)
  * 利用测试函数，求弱解

## 抽象代数

* **环（ring）**，是抽象代数的基本结构，指非空集合 $$ R $$ 配以加法与乘法两种二元运算，满足：   
  * $$ (R, +) $$ 是阿贝尔群（加法交换、有逆元等）；   
  * 乘法结合律 $$ (ab)c = a(bc) $$；   
  * 乘法对加法的左右分配律 $$ a(b + c) = ab + ac $$、$$ (a + b)c = ac + bc $$。  
  * 环在数学与工程领域应用广泛，为众多数学分支及实际应用提供了统一的代数框架，是理解和解决复杂问题的重要工具：
    * 在密码学中，其结构用于设计加密算法；
    * 代数几何里，环帮助刻画空间与函数性质；
    * 编码理论中，借助环的特性构造高效编码，提升数据传输可靠性与效率。

## 泛函分析

[Hanner不等式与Lp空间的一致凸性](https://www.bilibili.com/video/BV18VZcYKEsz)

* Lp空间

![image-20250405000218965](./Mathematics/image-20250405000218965-1506147.png)

* 一致凸空间
  * ![image-20250405000623658](./Mathematics/image-20250405000623658-1506147.png)

* Lp空间的一致凸性

![image-20250405000829187](./Mathematics/image-20250405000829187-1506147.png)



![image-20250405000910294](./Mathematics/image-20250405000910294-1506147.png)

## 数字信号处理

### DFT

![image-20250605195600286](./Mathematics/image-20250605195600286-1506147.png)







## 数学在工科的应用

* [为什么这么多年直到马斯克出来才想到做可复用火箭？](https://www.zhihu.com/question/597238433/answer/3080541702)
  * ![image-20241110022322621](./Mathematics/image-20241110022322621-1506147.png)

