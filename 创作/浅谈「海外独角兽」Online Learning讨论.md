# 浅谈「海外独角兽」 Online Learning 讨论

- 推荐一篇文章，该文章**认为** **LLM** **下一个核心范式会是 Online Learning**，并结合 LLM 演进与 Machine Learning 领域多种任务场景（包括推荐系统等）的算法分析，从多维度进行了讨论，信息密度、知识广度和深度较大。
- 下面我结合自己的理解，讨论文章中一些观点：

![image-20251101202838760](./%E6%B5%85%E8%B0%88%E3%80%8C%E6%B5%B7%E5%A4%96%E7%8B%AC%E8%A7%92%E5%85%BD%E3%80%8DOnline%20Learning%E8%AE%A8%E8%AE%BA/image-20251101202838760.png)

## 为什么需要研究 Online Learning

* 第23条提到“Model 的数据利用效率问题”，强调了LLM领域在当下的弱点，数据利用效率低，从而 online rl 无法有效达成 in-context learning 的能力并替代它，从而在AI应用中仍大量依赖 in-context learning，即最近火热的 context engineering 的概念。这些概念一定程度是由于模型能力、范式做不到 online learning，从而衍生出的工程手段。
* 第33条提到“数据分布差异越大，Online Learning 价值越突出”，真实应用场景的数据分布差异大（如涉及人类偏好、实时新闻或个性化需求的任务），本条强调了Online Learning对AI Application领域的价值



## 和推荐系统的对比

* 第11条提到，“真正的 Online Learning 系统应能够随着数据的不断收集持续提升性能，而不是在短期内很快收敛。”，推荐系统通过稀疏emb做到了这一点，类比相当于给 LLM 增加了 doc id 而非 token 级别的 embedding。
* 第72条提到，“系统层面的 Online Learning 正是通过外部知识的存储与利用来实现能力提升。”，可类比于推荐系统中的实时特征，均是通过提升[系统的记忆特征的完备性](https://zhuanlan.zhihu.com/p/1930155262179807978)，来提升能力。可结合《推荐系统里的“七伤拳”：那些高ROI但长期有损的优化》（搜索可搜到）这篇文章理解。
* 第34条，对五类AI系统进行了对比：
  * ![image-20251101212617347](./%E6%B5%85%E8%B0%88%E3%80%8C%E6%B5%B7%E5%A4%96%E7%8B%AC%E8%A7%92%E5%85%BD%E3%80%8DOnline%20Learning%E8%AE%A8%E8%AE%BA/image-20251101212617347-2003586.png)

## Online Learning 的几种路线

* meta-learning：能力上的变革，核心是【快速】影响模型表现。
  * 模型实现（Parametric Learning）
    * in-weights learning
      * 模型结构中可学习的参数（比如MoE-CL的路线）
    * soft prompt
    * 特例：推荐系统sparse embedding实现
  * 系统实现（Non-parametric Learning）
    * in-context learning：文本prompt
    * 假如模型有 in-context RL 能力，能够理解 reward 代表的意思，就不需要 weights 更新。但如果模型不懂，就需要把 reward 更新到模型中。
* Lifelong learning：先做work，再在此基础上探索meta-learning更强的能力
  * 思路1：更充分的语义化，才能在稠密的模型中共享信息增益
  * 思路2：通过系统赋予模型学习能力，如AlphaGo

* 关于实现的路线的讨论：
  * 第13条，“从实现路径上，做好 meta learning 之后再做 lifelong learning 会更轻松。”，我认为存疑，lifelong learning的执行路径更清晰，系统更简单；meta learning的探索性更强，上限更高。  路径上可能应该先做lifelong learning。
  * 第86条指出，考虑到 Online Learning 的算力挑战，倾向于 in-weight learning。
  * 第88条指出，考虑到快速跑通流程，倾向于优先 in-context learning。

## Online Learning 的未来

* 第49条，指明Online Learning的研究方向：“通过交互、探索（exploration）和奖励的自我收集（reward self-collection），让模型能够不断改进自身能力。”，这句总结较为精要。
* 第84条分析Agent系统未来是否会走向端到端模型，判断仍会先模块化，优化关键组件。此处比较容易切入的地方是将记忆作为参数而非明文存储，即soft prompt。
* 第85条指出，“将 context 中的关键信息注入到模型的参数（weights）中，通过 learning 来改变分布。”是解决LLM的个性化问题的重要路线，我感觉context类似于RNN中的状态，该范式类似于“用RNN包住transformer”，也许该方向需要Linear Attention相关的算法积累。