# AIGC

[toc]

## 概要

* 术语
  * Large Language Model(LLM)
  * ChatGPT
  * PaLM/Bard(Google)
  * Llama(Meta)
  * Github Copilot
* 大模型的发展
  * https://arxiv.org/pdf/2304.13712
  * BERT pretrain的概念

### 大模型简要介绍

* 表面上做什么事情：不断根据前文生成“下一个”词
* 大模型的输入
  * 编码：word embedding、one-hot、文字、整数
* 关键要素
  * 数据
    * 微调数据如何大量获得

  * 算力
  * 训练技术：RLHF、prefix tuning、hard/soft prompt tuning、SFT、retrieval augment
  * 模型结构

* 影响要素
  * 信任
  * 安全
  * 隐私
  * 认知


### 参考大纲

![img_v2_a03e4de2-1cd1-498e-a8b8-fdab0c33371g](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/AIGC/course.png)

## 历史发展

* emergent ability
  * [How much bigger can/should LLMs become?](https://cmte.ieee.org/futuredirections/2023/04/24/how-much-bigger-can-should-llms-become/)
  * https://arxiv.org/abs/2206.07682
  * 100TB=50000Billion

![截屏2023-11-19 05.00.35](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/AIGC/emergent-1.png)

![image-20231119050100619](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/AIGC/emergent-2.png)

![image-20231119050435217](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/AIGC/history-1.png)

![Compute required for training LLMs](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/AIGC/Compute-for-Training-LLMs-GPT3-paper-672x385.jpg)



* Note
  * GPT-3.5相比于GPT-3，参数量变化不大，效果差距很大，这是由于微调技术

## 算法

### Attention Is All You Need

* The best performing models also connect the encoder and decoder through an attention mechanism. 
  * Encoder: 映射到另一个语义空间
* Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence.
* 模型结构是什么？
  * 过N个注意力层，再过一个full connection
  * Attention(Q,K, V) = softmax(QK^T/sqrt(d_k))V

* 模型参数是什么？
  * 词嵌入向量
    * learnable?
  * 将词嵌入向量转化为q、k、v向量的三个矩阵和bias
* 模型输出是什么？
  * 全连接层的结果，一个长度为全部词汇数量的向量
  * 如何增强随机性：
    * top-k采样

* The Transformer follows this overall architecture using **stacked self-attention and point-wise**, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1
  * 左边encoder，右边decoder
    * Encoder: 自注意力
    * Decoder：Q用outputs embedding做masked attention后的结果，K、V用encoder结果
    * 表征向量512维
  * masked multi-head attention保证输出对输入的感知序列不会超出长度
  * 自注意力机制：Q（输入矩阵）、K（字典）、V
    * 用1/(dk)^(1/2) scale了一下QK的乘法，可能是为了防止gradient太小
    * Dot product的结果方差比additive attention的方差大

![image-20231025202735456](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/AIGC/transformer.png)

* Multi-head attention

![image-20231025203852956](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/AIGC/multi-head-attention.png)

#### Implementation

* TODO2: https://tensorflow.org/text/tutorials/transformer

### GPT-2

* TODO1: https://jalammar.github.io/illustrated-gpt2/

### ChatGPT

* 对话式大型语言模型：https://openai.com/blog/chatgpt/
  * 自回归语言模型：帮助背下来事件知识
  * 大语言模型：百亿参数以上
    * 不好做finetune，成本高
    * 用prompt作为输入，generated text作为输出
    * 语言知识 + 事件知识，事件知识更需要大模型

  * 未来：AGI(Artificial General Intelligence)；教会它使用工具

* 三个关键技术：
  * In-Context Learning 情景学习
    * 在前向中学习
    * 涌现能力：百亿参数规模之后，能力突然提升，改变传统学习范式
    * 大幅降低下游任务开发成本
    * 《Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?》 --> 随机label仍可能提升效果
  * Chain-of-Thought, CoT 思维链
    * 《PAL: Program-aided Language Models》，让语言模型生成能由计算模型执行的描述代码
    * 在大模型中打破scaling law
  * Learning from Natural Instructions 自然指令学习
    * 很像情景学习，样本逐渐简化（负例不需要suggestion；不需要负例）
    * https://instructions.apps.allenai.org/
    * OpenAI: 通过人类反馈对齐人类指令
* RLHF
  * 见【算法-finetune-RLHF】部分
* limitations
  * Correctness: 模型不是全知的，一本正经地胡说八道
  * sensitive to rephrase
  * verbose
  * No asking clarifying questions，而是猜
  * it will sometimes respond to harmful instructions or exhibit biased behavior

* [Iterative deployment](https://openai.com/blog/language-model-safety-and-misuse/)
* Evaluation
  * Holistic Evaluation of Language Models


* Note
  * 科技部部长王志刚表示，ChatGPT有很好的计算方法，同样一种原理，在于做得好不好；就像踢足球，都是盘带、射门，但是要做到像梅西那么好也不容易。
  * 客观题高考515分水平
* [专访Altman](https://www.pingwest.com/a/285835)

  * **感想**：有几个点值得关注：ai自运行的能力、ai隐藏意图的能力、ai与真实物质世界接口的能力、ai认识到自己的现实处境并差异化处理的能力

    * 当这些能力完全具备，可能AGI确实可以毁灭人类

  * 当他观察模型的隐藏层时，发现它有一个专门的神经元用于分析评论的情感。神经网络以前也做过情感分析，但必须有人告诉它们这样做，而且必须使用根据情感标记的数据对它们进行专门的训练。而这个神经网络已经自行开发出了这种能力。
  * 语言是一种特殊的输入，信息量极为密集
  * "假设我们真的造出了这个人工智能，其他一些人也造出了"。他认为，随之而来的变革将是历史性的。他描述了一个异常乌托邦的愿景，包括重塑钢筋水泥的世界。他说："使用太阳能发电的机器人可以去开采和提炼它们需要的所有矿物，可以完美地建造东西，不需要人类劳动。"你可以与 17 版 DALL-E 共同设计你想要的家的样子，"Altman说。"每个人都将拥有美丽的家园。在与我的交谈中，以及在巡回演讲期间的舞台上，他说他预见到人类生活的几乎所有其他领域都将得到巨大的改善。音乐将得到提升（"艺术家们将拥有更好的工具"），人际关系（人工智能可以帮助我们更好地 "相互对待"）和地缘政治也将如此（"我们现在非常不擅长找出双赢的妥协方案"）。
  * GPT-4学会了“说谎”：验证码

    * -> 让GPT-4讲解自己做事情的目的，将不再可靠
    * Sutskever 说，他们可能会在弱小的时候采取一种行动，而在强大的时候采取另一种行动。我们甚至不会意识到，我们创造的东西已经决定性地超越了我们，我们也不知道它打算用自己的超能力做些什么。


### GPT-4

* GPT-4幕后的研发团队大致可分为七个部分：预训练（Pretraining）、长上下文（Long context）、视觉（Vision）、强化学习和对齐（RL & alignment）、评估和分析（Evaluation & analysis）、部署（Deployment）以及其他贡献者（Additional contributions）
* [GPT-4技术报告](https://mp.weixin.qq.com/s?__biz=Mzk0NzQzOTczOA==&mid=2247484155&idx=1&sn=5ef0fcf20d4b87366269d3c0cf4312c0&scene=21#wechat_redirect)
  * 32k对应50页的context
* [Language models can explain neurons in language models](https://openai.com/research/language-models-can-explain-neurons-in-language-models)
  * 步骤：
    * GPT-4解释某个GPT-2神经元的行为
    * 用GPT-4模拟这一行为
    * 比较并打分

  * OpenAI 共让 GPT-4 解释了 GPT-2 中的 307200 个神经元，其中大多数解释的得分很低，只有超过 1000 个神经元的解释得分高于 0.8。
  * 三种提高解释得分的方法：
    - 对解释进行迭代，通过让 GPT-4 想出可能的反例，根据其激活情况修改解释来提高分数。
    - 使用更大的模型来进行解释，平均得分也会上升。
    - 调整被解释模型的结构，用不同的激活函数训练模型。
  * https://github.com/openai/automated-interpretability
  * 传统的视觉解释方法不能scale well
    * https://openai.com/research/microscope
    * https://distill.pub/2020/circuits/curve-detectors/

### Sora

* 技术报告：https://openai.com/research/video-generation-models-as-world-simulators

* [一锤降维！解密OpenAI超级视频模型Sora技术报告，虚拟世界涌现了](https://mp.weixin.qq.com/s/ODsebK3fEc-adRDwRVDhQA?poc_token=HMxd12WjhN3a1nz74MaIrMjep8dIn2Cj_NTdFwef)

  * 扩展视频生成模型的规模，是构建模拟物理世界通用模拟器的非常有希望的方向
  * patch
    * 从宏观角度来看，研究者首先将视频压缩到一个低维潜空间中，随后把这种表征分解为时空patch，这样就实现了从视频到patch的转换。
    * 在推理时，可以通过在一个合适大小的网格中适当排列随机初始化的patch，从而控制生成视频的大小。
  * 训练技巧
    * 直接在视频原始比例上训练
    * 研究者采用了DALL·E 3中的重新标注技术，应用在了视频上。
      * 首先，研究者训练了一个能生成详细描述的标注模型，然后用它为训练集中的所有视频，生成文本说明。
      * 他们发现，使用详细的视频说明进行训练，不仅能提高文本的准确性，还能提升视频的整体质量。
      * 类似于DALL·E 3，研究者也使用了GPT，把用户的简短提示转化为详细的说明，然后这些说明会被输入到视频模型中。

  * 生成的视频特点：
    * 多种输入形式、多视频间过渡、人和物的特征

### finetune

* finetune v.s. from scratch
* 如何做finetune
  * 基座模型选型
* 全参数finetune和小参数量finetune
  * 小参数量finetune
    * Adapters
    * Prompt-tuning v1/v2
    * LoRA

* finetune需求
  * OpenAI: 1.3w条SFT prompt
* 很厉害的alpaca

![image-20231025213448602](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/AIGC/alpaca.png)

#### 指令微调

* 指令微调是什么? - superpeng的回答 - 知乎
  https://www.zhihu.com/question/603488576/answer/3178990801
  * 指令微调是一种特定的微调方式，在不同的论文中以不同的方式引入。我们在一个新的语言建模任务上对模型进行微调，其中的示例具有额外的结构，嵌入到模型提示中。
    * 先无监督训练，再用有监督的“指令-回答“预料
    * 指令调整模型接收一对输入和输出，描述引导模型的任务。
  * 核心思路：解决“回答问题”与“接话”的差异
  * Note：
    * 数据获取昂贵（RLHF人工打分的成本比人工写故事要低）
    * 对开放性问题效果不好（write a story about ...）

#### RLHF

* Reinforcement Learning from Human Feedback (RLHF), using the same methods as [InstructGPT](https://openai.com/blog/instruction-following/), but with slight differences in the data collection setup
  * RLHF的blog介绍：https://huggingface.co/blog/rlhf
    * supervised fine-tuning: human AI trainers provided conversations in which they played both sides—the user and an AI assistant
  * 步骤：
    * 预训练一个语言模型 (LM) ；
    * 聚合问答数据并训练一个奖励模型 (Reward Model，RM) ；
    * 用强化学习 (RL) 方式微调语言模型（LM）。
      * 长期以来，出于工程和算法原因，人们认为用强化学习训练 LM 是不可能的。而目前多个组织找到的可行方案是使用策略梯度强化学习 (Policy Gradient RL) 算法、近端策略优化 (Proximal Policy Optimization，PPO) 微调初始 LM 的部分或全部参数。因为微调整个 10B～100B+ 参数的成本过高 (相关工作参考低秩适应 LoRA 和 DeepMind 的 Sparrow LM)
  * reward model: 人工打分
    * 人工写答案 -> 人工选答案 -> 机器选答案
    * prompt dataset
    * fine-tune the model using [Proximal Policy Optimization](https://openai.com/blog/openai-baselines-ppo/)
    * 一些巧妙的打分方式：
      * 客服点按钮，选取ai答案，也是finetune过程
      * reddit帖子中的最高分

![img](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/AIGC/ChatGPT_Diagram.svg)

* 

#### LoRA

![image-20231026212212239](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/AIGC/LoRA.png)



https://github.com/huggingface/peft

## 框架

* AI系统：记录数据、与人交互、机器学习分析、预测、干预人的决策

### 训练成本

* LLaMA：2048 A100 21d
  * a100一个月几十刀，训一个几十万
* Note
  * 和芯片的对比：This “growth” is strikingly similar to the one involved in chip evolution where as the number of transistors increases (higher density on a chip) the cost for plants manufacturing  those chips skyrocket.  In  the case of chip manufacturing  the economics remained viable because new plants did cost more but they also produced many more chips so that till the middle lf the last decade the cost per chip was actually  decreasing generation over generation (one effect captured in the Moore’s law).
  * As with chips one may  wonder if there is a limit to the economic affordability (there sure is, it is just difficult  to pinpoint!).
  * TODO: https://www.wired.com/story/openai-ceo-sam-altman-the-age-of-giant-ai-models-is-already-over/

## 应用

### 变现逻辑

* [陆奇对话高科技营销之父：从技术到市场，ChatGPT还需跨越“鸿沟”](https://mp.weixin.qq.com/s/xvWzQ73Dg0XzJ5LxwmyWsA)
  * 近期出现的真正具有颠覆性的技术，我认为一个是基因编辑，另一个就是OpenAI的ChatGPT
  * 如果我们想打造出ChatGPT这样高科技产品的市场，技术成熟远远不够，还需**将这种颠覆性创新产品社交化**，这中间还有巨大的“鸿沟”需要跨越。
  * 技术生命周期一般分为4个阶段：
    * 第一阶段是有一些技术的狂热者以及有远见者，相信技术能够成功，希望成为第一个尝试新技术的人；
      * 早期阶段的策略：等对的人主动找你
    * 第二阶段是早期大众会觉得这项技术可能是对的，但是还处于早期，需要观望一下，他们可能会看看别人会不会采用这项新技术，会跟风去使用或者拒绝使用。
    * 当一家初创公司积累了足够多的使用案例后，大家就会认为这其实是行业的基础设施建设，是我们每个人都需要的，比如云计算和Wi-Fi，人工智能等，那些观望者就会想要进入市场，追赶潮流。瞬间所有预算涌入市场，购买你的产品，我们把这种现象称为“龙卷风”。
  * 跨越“鸿沟”的关键所在就是如何让早期大众能够开始采用颠覆性的新技术，你必须要瞄准一个很小的利基市场，他们有很多痛点，需要更多新的解决方案来解决当下的问题。如果你能解决他们的问题，他们就会采用你的新技术。
  * 在早期市场，人们买的不是产品，而是**项目**。早期市场，作为一个初创企业，你的客户其实把你当成了一个咨询公司，他们可能会给你很多资金，让你按照他的想法去打造一个产品。
    * 与ToB“项目制”的联系
  * 早期市场的这些客户，我们称之为旗舰型的客户，他们一定是一群知名度很高的客户。比如美国银行、福特汽车、迪士尼或者微软，一定是大家都知道的企业。
    * 一定要找那些大型的知名企业作为你的客户，做两个项目就够了，在这之后就不要继续再做项目，而是开始重复地做相同的解决方案。
  * 我还有另外一个问题，如何去辨别一个非常小众的需求和一个有远见的需求之间的区别？
    * **摩尔：**我觉得利基市场的需求存在一个实用案例，同时也会有一个预算，**这个预算不是为你的产品，而是为了解决他们的问题**。你可能会在做项目的时候遇到这样一个问题，有远见者说这是非常重要的问题，我希望能够改变整个世界。但是在你的头脑里，你应该想到，如果别的客户也有同样的问题，我们如何解决。因为我们实际上解决不了太多问题，但是实用主义者是希望你的产品一定能解决他们的问题。
    * 核心是计算给新客户做定制化需求的代价
    * 更进一步，形成生态，寻找加盟合作。当市场越来越大时，首先是基础服务的提供商们赚钱，然后就轮到后端的软件提供商。
  * 现在可以用AI去管理数据，AI可以从海量数据中精准地找到你想要的信息，这一点比人做得更好。

* 关于开源
  * 开源可以非常迅速地渗透市场，这就像免费增值一样，如果坚持这一点，战略就会把握在更强有力的人手中。如果你卖出你模型中的一部分，你的客户竞争将升级到一定的水平，而你会继续前进，这是一种批量运营模式。
  * 我对于一家公司中的context（场景上下文）来说，开源是最佳选择，但对core（核心）而言则不是这样。核心指的是让你的产品脱颖而出，为你提供别人不具有的比较优势的东西，也就是你应该保护的有产权的知识，大家都想得到它并希望它正常运行，如果它正常运行，你不会得到任何奖励，但如果它运行故障，你却会受到惩罚，所以开源会带来网络安全和产品质量的双重风险。
  * 作为对比，关于PingCap激进的开源：
    * 这是一个典型的开源模式，他们给你SaaS或者给你分发，本质上就跟Red Hat一样定制，他们非常相信开源，相信这会让创新更快，长期客户获取的成本会降低。
    * 规模化和货币化会比较困难

* 企业业务增长的可能性
  * 现在业务规模小、赛道窄的互联网公司，有可能做起来了

* 自动做ppt
  * https://gamma.app/

* 自动画结构化的脑图
* 数据库+NLP
* ToB场景示例
  * 大模型相关的ToB场景研发成本下降

![image-20231025201548103](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/AIGC/tob.png)



### Prompting

* https://learnprompting.org/docs/category/-basics TODO
* [23 prompt rules](https://lifearchitect.ai/sparrow/)

### Agent

* HuggingGPT： 缝合怪
  * https://beebom.com/how-use-microsoft-jarvis-hugginggpt/

### 更多方向

* 决策大模型
* 对ToG的影响
  * Geoffrey Moore：我觉得中国的模型可能跟美国的模型完全不一样。就美国来说，我觉得政府需要去为一些研究提供资金，他们就像风投者一样。我们公共服务的三个方面，如社会的安全网、医疗和教育都陷入了困境，他们都想去提供下一代的服务，但是一来没有合适的人才，二来用人成本太高，所以他们真正需要的是合适的软件来解决他们的问题（数字转型），这就带来了跨越“鸿沟”的机会。（但很难做）
* 游戏 AI Npc
  * https://foresightnews.pro/article/detail/30224
  

## Prompt Engineering

* 基于openai api
  * https://platform.openai.com/docs/guides/gpt
  * https://platform.openai.com/docs/api-reference/chat
  * model
  * role
    * user
    * assistant
    * system: 大环境
  * temperature: 0~2
* 多轮交互：系统回复也加入上下文
* 安全性：OpenAI内容审核，薄弱；防止机制被洗掉



## LangChain

* 介绍
  * 面向大模型的开发框架
  * 简单实现复杂功能的AI应用
  * 多组件封装
* 向大模型输入知识块，大模型结合外部数据

![image-20231026203340859](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/AIGC/langchain.png)

* I/O模块：
  * Format: PromptTemplate
  * 模型
    * LLM: from langchain.llms import OpenAI
    * ChatModel
  * Output parsers

* data connection
  * source
  * load
    *  from langchain.document_loaders import PyPDFLoader
  * transform
    * Splitter
    * Translate
  * embed: 模型只认识feature
  * vector store
    * FAISS
  * retrieve
* memory
  * 针对多轮对话强相关
  * Note: load_memory_variables()需要填参数{}
  * ConversationBufferWindowMemory
  * ConversationSummaryMemory

## 安全

* 去中心化的思想构建未来的AI安全：https://mp.weixin.qq.com/s/K1gbW1aIkwl8aLzkD9nYnQ
  * 比特币：攻击收益远小于攻击成本
  * 以生态著称的公链以太坊：虽然秘钥也是几十位，但是系统就太复杂了，各种二层技术、跨链桥等带来了很多漏洞，以至于网络攻击不断，就是因为攻击收益大于攻击成本
  * 方案：确权，实名，竞争

## Potpourri

### Llya访谈系列

* [访谈系列·E01S01｜AI大神Ilya访谈揭秘GPT-4成功背后的关键，20年如一日的AGI坚守终惊艳世界](https://mp.weixin.qq.com/s?__biz=Mzk0NzQzOTczOA==&mid=2247496274&idx=2&sn=4450fee63ae1cff6449e8f2f97784224&chksm=c3746347f403ea51db05dd2a9a14721340f68fb9af2c9ecdda8ac99d5e4ae7cbed888dc20a4c&scene=178&cur_album_id=3096596465520590849#rd)

* [访谈系列·E02S01｜llya的AGI信念：为伊消得人憔悴，十年终迎GPT震撼崛起](https://mp.weixin.qq.com/s?__biz=Mzk0NzQzOTczOA==&mid=2247496316&idx=1&sn=538c876f4341e77d9d8e9cac1dd37b03&chksm=c3746369f403ea7f1e56ab4d58b5fbe47c21febb8398084340bec75250d6504e0d8e31dc2386&scene=178&cur_album_id=3096596465520590849#rd)
  * OpenAI的核心理念
    * 无监督学习的一种路径是通过数据压缩实现（unsupervised learning through compression）
      * 2017年论文，发现神经元能学到情感的解耦特征
    * 强化学习（reinforcement learning）包括游戏对抗学习和人类反馈学习
  * transformer为什么成功？
    * gpu易于计算attention
    * 非RNN结构
  * 双下降现象
    * https://zhuanlan.zhihu.com/p/96739930
* [访谈系列·E03S01｜GPT-4成功背后灵魂人物Ilya访谈解读——从预训练模型到可靠可用AGI](https://mp.weixin.qq.com/s?__biz=Mzk0NzQzOTczOA==&mid=2247496355&idx=1&sn=6e997afffa78af2404d0e67661bb9281&chksm=c37463b6f403eaa083f14c6355bb13813dcd94494cef798e81730b823d11047b8bcce6d81113&scene=178&cur_album_id=2921262939645526019#rd)
  * AGI是否有意识？
    * 你无法分辨出这是机器智能自己的意识还是它学习了有关意识的人类文本内容后模仿的意识
  * 当我们训练大型神经网络以准确预测互联网上大量不同文本的下一个词时，实际上我们正在学习一个世界模型。从表面上看，神经网络只是在学习文本中的统计相关性，但实际上，学习统计相关性就可以将知识压缩得很好。神经网络学习的是在生成文本过程中的某些表达，因为文本只是这个世界的一种映射，所以神经网络学习了这个世界的许多方面的知识。
  * 这就是它在准确预测下一个词的任务中学到的内容。对下一个词的预测越准确，还原度就越高，所以你看到的文本的准确度就越高。这就是ChatGPT模型在预训练阶段所做的，它尽可能多地从世界的映射（也就是文本）中学习关于世界的知识。
  * 但这并不能说明（预训练后的）神经网络会表现出人类希望它表现的行为，这需要第二阶段的微调、人类反馈的强化学习（RLHF）以及其他形式的AI系统的协助。这个阶段做得越好，神经网络就越有用、可靠。
  * 多模态非必要，但绝对有用
  * 预测具有不确定性的高维向量：给定某书中的一页，预测下一页

![图片](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/AIGC/gpt-3.png)
