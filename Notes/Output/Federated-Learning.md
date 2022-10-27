[toc]

### Federated Learning

* 概念
  * Cross-Silo 纵向联邦
  * Cross-Device 横向联邦
* [Google Research: Themes from 2021 and Beyond](https://ai.googleblog.com/2022/01/google-research-themes-from-2021-and.html)
  * Trend 3: ML Is Becoming More Personally and Communally Beneficial
    * 从 ML+产品（Pixel手机） 到 联邦学习 
    * phone cameras
    * live translate/caption
    * [federated analytics](https://ai.googleblog.com/2020/05/federated-analytics-collaborative-data.html) and federated learning
      * 复用 FL 的 evaluating 能力 (without the learning part)
      * Now Playing: on-device database
      * https://arxiv.org/pdf/1902.01046.pdf
      * secure aggregation protocol

#### IOS14

* 概念
  * opt-in：IDFA授权弹窗允许的用户
  * opt-out：弹窗后不允许的用户
  * IDFA：唯一设备标识符，同一设备间不同App可共享
  * mmp：mobile measurement partner，ios14之前，用于APP精确归因convert
    * 可能不稳定。。。需要考虑容灾
    * [iOS 14 and MMPs: Where we stand right now](https://www.singular.net/blog/ios14-mmp/)
      * pros and cons of SKAdNetwork
      * No [view-through attribution](https://www.singular.net/glossary/view-through-attribution/)
        * vta占比高，会使得preconvert低估，需要纠偏萃取label，乘以 (vta+cta)/cta
  * post-click（萃取）：https://instapage.com/blog/post-click-optimization
* 之前：user唯一IDFA，用户设置 Limit Ads Tracking (LAT)
  * LAT主要存在于14.5-的用户中
* 之后：App Tracking Transparency (ATT)，只能获取 IPUA/email
  * 官方 SKAdNetwork 方案 https://developer.apple.com/documentation/storekit/skadnetwork
    * *Ad networks* that sign ads and receive install-validation postbacks after ads result in conversions
    * *Source apps* that display ads provided by the ad networks
    * *Advertised apps* that appear in the signed ads
    * Supports measurement of all ad networks, including self-attributing networks or SANs (Google, Facebook, Twitter, Snap, etc.)
    * SKAd [Code](https://www.singular.net/blog/skadnetwork-code/)
    * 24-72h的兑回延迟
  * [SKAN compaign id 限定100个 —— AppsFlyer](https://support.appsflyer.com/hc/zh-cn/articles/360011502657-SKAdNetwork%E5%B9%BF%E5%91%8A%E5%B9%B3%E5%8F%B0%E5%AF%B9%E6%8E%A5%E6%8C%87%E5%8D%97)
    * 实验越多，分配到每个SKAN id上的convert越少，可能触发隐私阈值
  * [Apple User Data Privacy and Data Use](https://developer.apple.com/app-store/user-privacy-and-data-use/)
  * Fingerprinting and probabilistic matching
    * [关于 IPUA](https://www.ichdata.com/ios-app-tracks-attribution-challenges.html)，ichdata这个网站有许多跟踪匹配方向的文章
      * User Agent https://whatmyuseragent.com/
    * IPUA 局限性
      * IPUA在用户维度不具有唯一性
      * IPUA具有时效性
    * probabilistic matching: IPUA -- PM model --> user
      * FL下的PM归因，和普通PM归因有明显差异
  * Privacy-preserving attribution with supply-side consent
    * All these suggestions rely on Apple accepting the concept of single-side consent and assume that supply-side consent is sufficient for attributing user-level installs on the demand-side. 
* 归因服务
  * [Cryptographically Secure Bloom-Filters](https://dzlp.mk/sites/default/files/255.pdf)

#### Privacy

* [Safari Privacy Overview by Apple](https://www.apple.com/safari/docs/Safari_White_Paper_Nov_2019.pdf)
  * Intelligent Tracking Prevention (ITP) , Protection from cross-site tracking
    * on device学习涉及隐私的domains
    * 阻止tracking social widgets 
      * 2005年起，默认阻止利用 third-party cookies做cross-site tracking
      * ITP开始，阻止利用其它形式的data做cross-site tracking
  * Private Click Measurement：端上完成归因
  * Minimizing data sharing with the Smart Search field: 使用浏览器搜索，而不使用网页搜索栏
  * Secure payments on the Web
    * When a purchase is made on a website using Apple Pay, **a device-specific number and a unique per transaction security code** are sent to the merchant rather than the actual credit card number and security code, so the user’s real credit card information can’t be compromised.
  * Sync and sign-in features that keep the user in control
    * 只会利用icloud keychains帮助登录网站，但切换浏览器不会自动登录它们
* [iCloud Private Relay Overview](https://www.apple.com/privacy/docs/iCloud_Private_Relay_Overview_Dec2021.PDF)
  * for IOS15+
  * sending their requests through two separate internet relays so that no single entity can combine IP address, location, and browsing activity into detailed profile information
  * Private Relay protects all web browsing in Safari and unencrypted activity in apps
  * Private Relay’s dual-hop architecture protects the privacy of users by separating who can observe their IP addresses from who can see the websites they visit. 
  * Transport and Security Protocols
    * These include protocols to proxy internet connections, protect DNS name lookups, and authenticate users when connecting to Private Relay in order to prevent fraud.
    * QUIC (RFC 9000) is a general-purpose transport layer network, standardized by the IETF in May 2021. Connections using QUIC can achieve great performance even in poor network environments by taking advantage of improved loss recovery. QUIC also allows connections to easily switch between network interfaces, allowing connectivity to be maintained as users move between Wi-Fi and cellular networks.
      * TLS 1.3
    * Oblivious DNS over HTTPS (ODoH) ODoH adds a layer of public key encryption, as well as a network proxy between devices and DNS servers. The combination of these two added elements is designed such that only the user has access to both the DNS messages and their original IP address at the same time.
    * Relay access and fraud prevention
      *  blind signature
      *  the server performs device and account attestation using the Basic Attestation Authority (BAA) server prior to vending out tokens
    * Coverage and Compatibility 
      * Cellular services, such as Multimedia Messaging Service (MMS), telephony services (XCAP), Entitlement Server access, tethering traffic, and Visual Voicemail, do not use Private Relay

![private-relay](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/Federated-Learning/private-relay.png)

* 一个苹果隐私相关的例子：IPhone内存满白苹果问题
  * [描述](https://www.zhihu.com/question/432520118/answer/1601795843): 隐私相关的数据校验任务无法执行，导致卡白苹果
  * [民间解决方案](https://zhuanlan.zhihu.com/p/493774509): 怀疑是ROM连带的RAM问题导致卡屏，通过改变电量（用户少有的可操作空间），使校验任务可执行
* user embedding
  * model training outputs might sufficiently constitute "user profiles" to violate PA restrictions
  

#### 《Advances and open problems in federated learning》

略读：Chapter 4、5

##### 1.Introduction

* Federated learning is a machine learning setting where multiple entities (clients) collaborate in solving a machine learning problem, under the coordination of a central server or service provider. Each client’s raw data is stored locally and not exchanged or transferred; instead, focused updates intended for immediate aggregation are used to achieve the learning objective.
  * Focused updates are updates narrowly scoped to contain the minimum information necessary for the specific learning task at hand;
  * aggregation is performed as early as possible in the service of data minimization.
* Learning with privacy
  * https://machinelearning.apple.com/research/learning-with-privacy-at-scale
* Notes
  * Typically a hub-and-spoke topology, with the hub representing a coordi- nating service provider (typically without data) and the spokes connecting to clients.
  * FL 特点
    * Data is generated locally and remains decentralized.
    * A central orchestration server/service organizes the training, but never sees raw data.
    * However, the approach presented above has a substantial advantage in affording a separation of concerns between different lines of research: advances in compression, differential privacy, and secure multi-party computation can be devel-oped for standard primitives like computing sums or means over decentralized updates, and then composed with arbitrary optimization or analytics algorithms, so long as those algorithms are expressed in terms of aggregation primitives.
  * Client 特点
    * Stateless
    * Clients cannot be indexed directly (i.e., no use of client identifiers)
    * Selectable 带来的问题：semi-cyclic data availability and the potential for bias in client selection
  * FL Research
    * be unambiguous about where computation happens as well as what is communicated.

##### 2. Relaxing the Core FL Assumptions: Applications to Emerging Settings and Scenarios

* Fully Decentralized / Peer-to-Peer Distributed Learning
  * graph
    * be sparse with small maximum degree so that each node only needs to send/receive messages to/from a small number of peers
    * that the notion of a round does not need to even make sense in this setting. See for instance the discussion on clock models in [85].
    * multi-agent optimization
      * undirected/directed graph
  * Algorithmic challenges
    * Effect of network topology and asynchrony on decentralized SGD
      * for the special case of generalized linear models, schemes using the duality structure could enable some of these desired robustness properties [231]
    * Local-update decentralized SGD
      * understanding the convergence under non-IID data distributions, not just for the specific scheme based on matching decomposition sampling described above
    * Personalization, and trust mechanisms
    * Gradient compression and quantization methods
    * Privacy
      * 差分隐私
      * local加噪声，通常有损
      * FL的 secure aggregation or secure shuffling 不易 apply
  * practical challenges: 基于 blockchain (a distributed ledger) 讲解
    * A blockchain is a distributed ledger shared among disparate users, making possible digital transactions, including transactions of cryptocurrency, without a central authority. In particular, smart contracts allow execution of arbitrary code on top of the blockchain, essentially a massively replicated eventually-consistent state machine. In terms of federated learning, use of the technology could enable decentralization of the global server by using smart contracts to do model aggregation, where the participating clients executing the smart contracts could be different companies or cloud services.
    * secure aggregation 要 handle dropping out participants [80]，可以利用区块链，用户 stake
* Cross-Silo FL
  * 纵向联邦 (partitioned by features)
    * 数据求交、中心服务；中间信息的传递
  * 横向联邦 (partiioned by examples)
  * Application: 搜 FATE
  * Incentive mechanisms
    * The option to deliver models with performance commensurate to the contributions of each client is especially relevant in collaborative learning situations in which competitions exist among FL participants.
    * Free-rider problem
  * DP
  * Tensor factorization: 中心服务只拿中间结果
    * [272] used an alternating direction method of multipli- ers (ADMM) based approach and [325] improved the efficiency with the elastic averaging SGD (EASGD) algorithm and further ensures differential privacy for the intermediate factors.
* Split Learning
  * Cut layer, smashed data
  * 《Training neural networks using features replay》并行化backward部分
  * One other engineering driven approach to minimize the amount of information communicated in split learning has been via a specifically learnt pruning of channels present in the client side activations [422].

##### 3. Improving Effiency and Effectiveness

* Non-IID Data in Federated Learning
  * non-IIDness 体现在：1）不同client的数据分布不一样；2）client分布及其数据分布随时间变化
  * Non-identical client distributions
    * Feature distribution skew (covariate shift): P(x)
      * 思路是学习 P(y|x)，学习不变的量
    * Label distribution skew (prior probability shift): P(y)
    * Same label, different features (concept drift): P(x|y)
    * Same features, different label (concept shift): P(y|x)
      * some form of personalization (Section 3.3) maybe essential to learning the true labeling function
    * Quantity skew or unbalancedness
  * Violations of indenpendence
    * cross-device FL: a strong geographic bias in the source of the data [171]
  * Dataset shift
  * Strategies for Dealing with Non-IID Data
    * 3.2.2, 一个路子是augment data，构造通用数据来缓解
    * The heterogeneity of client objective functions gives additional importance to the question of how to craft the objective function — it is no-longer clear that treating all examples equally makes sense. Alterna- tives include limiting the contributions of the data from any one user (which is also important for privacy, see Section 4) and introducing other notions of fairness among the clients; see discussion in Section 6.
    * 3.3 multi-model，由client侧的独立model来处理label distribution skew
    * [171] 有一种思路，对不同timezone的人部署不一样的模型iterate版本
* Optimization Algorithms for Federated Learning
  * client 引入的特点：client-sampling, stateless
  * 与其它技术的组合 -> 排除了纯异步更新的技术
    * Optimization algorithms do not run in isolation in a production deployment, but need to be combined with other techniques like cryptographic secure aggregation protocols (Section 4.2.1), differential privacy (DP) (Section 4.2.2), and model and update compression (Section 3.5).
    * Federated Averaging algorithm
  * Performing local updates and communicating less frequently with the central server addresses the core challenges of respecting data locality constraints and of the limited communication capabilities of mobile device clients. However, this family of algorithms also poses several new algorithmic challenges from an optimization theory point of view.
  * Optimization Algorithms and Convergence Rates for IID Datasets
    * 一些假设：H-smooth, gradient约束
    * Comparing these two results, we see that minibatch SGD attains the optimal **‘statistical’ term**, whilst SGD on a single device (ignoring the updates of the other devices) achieves the optimal **‘optimization’ term**
    * These result show that if the number of local steps K is smaller than **T/M^3** then the (optimal) statistical term is dominating the rate. However, for typical cross-device applications we might have T = 10^6 and M = 100 (Table 2), implying K = 1.
      * K: local steps, T: total communication rounds, M: clients per round
      * T*K = constant
    * 调参思路：
      * The costs (at least in wall-clock time) are small for increasing M, and so it may be more natural to increase M sufficiently to match the optimization term, and then tune K to maximize wall-clock optimization performance.
      * How then to choose K? Performing more local updates at the clients will increase the divergence between the resulting local models at the clients, before they are averaged. As a result, the error convergence in terms of training loss versus the total number of sequential SGD steps TK is slower. However, performing more local updates saves significant communication cost and reduces the time spent per iteration.
    * local-update SGD 分析
      * over-provisioning clients [81]
      * fix the number of local updates
        * EASGD
      * A larger challenge in the FL setting, however, is that as discussed at the beginning of Section 3.2, asynchronous approaches may be difficult to combine with complimentary techniques like differential privacy or secure aggregation.
  * Optimization Algorithms and Convergence Rates for Non-IID Datasets
    * the analysis technique can be extended to the non-IID case by **adding an assumption on data dissimilarities**, for example by constraining the difference between client gradients and the global gradient [305, 300, 304, 469, 471] or the difference between client and global optimum values [303, 268].   相应有更差的error bound结果
    * Clarifying the regimes where K > 1 may hurt or help convergence is an important open problem. [303]
* Multi-Task Learning, Personalization, and Meta-Learning
  * Personalization via Featurization
  * Multi-Task Learning
    * 适合 cross-silo 场景的 MOCHA 算法
    * task 对应 client subset 的思路：4.4.4
  * Local Fine Tuning and Meta-Learning
    * In the standard learning-to-learn (LTL) setup [56], one has a meta-distribution over tasks, samples from which are used to learn a learning algorithm, for example by finding a good restriction of the hypothesis space.
    * FL and MAML(model-agnostic meta-learning)
    * LML (lifelong ML)
  * When is a Global FL-trained Model Better?
    * Loss factorization tricks can be used in supervised learning to alleviate up to the vertical partition assumption itself, but the practical benefits depend on the distribution of data and the number of parties [373].
* Adapting ML Workflows for Federated Learning
  * Hyperparameter Tuning
    * Federated learning adds potentially more hyperparameters — separate tuning of the aggregation / global model update rule and local client optimizer, number of clients selected per round, number of local steps per round, configuration of update compression algorithms, and more.
  
  * Neural Architecture Design
    * Neural architecture search (NAS) in the federated learning setting
      * Among these, the gradient-based method leverages efficient gradient back-propagation with weight sharing, reducing the architecture search process from over 3000 GPU days to only 1 GPU day.
  
  * Debugging and Interpretability for FL
  
* Communication and Compression
  * compression objects
    * Gradient compression: 最现实，因为1）client的upload速率比download慢；2）averaging提供了更多lossy的可能
      * 高斯噪声和量化手段有些冲突
      * We note that several recent works allow biased estimators and would work nicely
        with Laplacian noise [435], however those would not give differential privacy, as they break independence between rounds.
  
    * Model broadcast compression
    * Local computation reduction
  
  * wireless-FL co-design
    * to leverage the unique characteristics of wireless channels (e.g. broadcast and superposi- tion) as natural data aggregators, in which the simultaneously transmitted analog-waves by different workers are superposed at the server and weighed by the wireless channel coefficients [4]
    * in sharp contrast with the traditional orthogonal frequency division multiplexing (OFDM) paradigm, whereby workers upload their models over orthogonal frequencies whose performance degrades with increasing number of workers [174]
  
* Application To More Types of Machine Learning Problems and Models
  * Bayesian neural networks [419] 提高神经网络的预估准确度，针对小数据集的场景
  
##### 4. Preserving the Privacy of User Data

* FL 架构特点

  * decomposing the overall machine learning work-flow into the approachable modular units we desire.
  * model updates 的信息量可能少；不泄漏user信息；云端只需要短暂持有model updates
* Actors, Threat Models, and Privacy in Depth

  * Table 7: Various threat models for different adversarial actors.
    * clients: TEE (secure enclaves)
    * server
    * output models
    * deployed models
  * 安全假定
    * DP
      * a fraction γ of the clients
      * cryptographic mechanisms instantiated at a particular security level σ.
    * honest-but-curious(HBC) security
  * 手段
    * running portions of a Secure Multi-Party Computation (MPC) protocol inside a Trusted Execution Environment (TEE) to make it harder for an adversary to sufficiently compromise that component
    * using MPC to protect the aggregation of model updates, then using Private Disclosure
      techniques before sharing the aggregate updates beyond the server
  * graceful degradation as “Privacy in Depth,”
    * in analogy to the well-established network security principle of defense in depth[361]
* Tools and Technologies

  * Secure Computations
    * Secure Multi-Party Computation (MPC) is a subfield of cryptography
      concerned with the problem of having a set of parties compute an agreed-upon function of their private inputs in a way that only reveals the intended output to each of the parties.
    * 量化到finite fields，确保over(under)flows可控 [194, 10, 206, 84]
    * 确保函数securely computed的现实手段，Table 8
      * Differential Privacy (local, cen- tral, shuffled, aggregated, and hybrid models)
      * Homomorphic encryption (HE): 问题在于谁持有key
        * a trusted non-colluding party is not standard in the FL setting
        * most HE schemes require that the secret keys be renewed often (due to e.g. susceptibility to chosen ciphertext attacks [117])
        * Another way around this issue is relying on distributed (or threshold) encryption schemes [392] [398]
    * TEE (secure enclaves) [437]
      * Features
        * Confidentiality: The state of the code’s execution remains secret, unless the code explicitly publishes a message;
        * Integrity: The code’s execution cannot be affected, except by the code explicitly receiving an input;
        * Measurement/Attestation: The TEE can prove to a remote party what code (binary) is executing and what its starting state was, defining the initial conditions for confidentiality and integrity
      * TEEs have been instantiated in many forms, including Intel’s SGX-enabled CPUs [241, 134], Arm’s TrustZone [28, 22], and Sanctum on RISC-V [135], each varying in its ability to systematically offer the above facilities.
      * TEE on GPU [447]
      * necessary to structure the code running in the enclave as a data oblivious procedure, such that its runtime and memory access patterns do not reveal information about the data upon which it is computing (see for example [73]).
      * TEE 和 FL 的结合：只计算 aggregate 等关键函数
    * Secure computation problems of interest
      * Secure aggregation
      * Secure shuffling
      * Private information retrieval: PIR is a functionality for one client and
        one server. It enables the client to download an entry from a server-hosted database such that the server gains zero information about which entry the client requested
        * cPIR, itPIR
        * use of lattice-based cryptosystems
  * Privacy-Preserving Disclosures

<img src="https://www.zhihu.com/equation?tex=P%28A%28D%29%5Cin%20S%29%5Cle%20e%5E%7B%5Cepsilon%7DP%28A%28D%5E%7B%27%7D%29%5Cin%20S%29%2B%5Cdelta" alt="P(A(D)\in S)\le e^{\epsilon}P(A(D^{'})\in S)+\delta" class="ee_img tr_noresize" eeimg="1">
    * LDP(local DP): 常用于中心服务analysis
    * distributed DP
      * Distributed DP via secure aggregation: 给梯度加noise
      * Distributed DP via secure shuffling: Encode-Shuffle-Analyze (ESA) framework, LDP + secure shuffling
        * the Prochlo system [73]
    * Hybrid differential privacy
  * Verifiability
    * various terms: checking computations [42], certified computation [343], delegating computations [210], as well as verifiable computing [195].
    * Zero-knowledge proofs (ZKPs)
      * [369] Nowadays, ZKP protocols can achieve proof sizes of hundred of bytes and verifications of the order of milliseconds regardless of the size of the statement being proved.
  
    * Trusted execution environment and remote attestation
  
* Protections Against External Malicious Actors
  * Auditing the Iterates and Final Model
  * Training with Central Differential Privacy
    * With this technique, the server clips the L2 norm of individual updates, aggregates the clipped updates, and then adds Gaussian noise to the aggregate.
    * Sources of randomness (adapted from [336])
    * Auditing differential privacy implementations

  * Concealing the Iterates
  * Repeated Analyses over Evolving Data
  * Preventing Model Theft and Misuse
    * 使用MPC方法的问题在于inference时没有合适的第三方

* Protections Against an Adversarial Server
  * Challenges: Communication Channels, Sybil Attacks, and Selection
  * Limitations of Existing Solutions
  * Training with Distributed Differential Privacy
  * Preserving Privacy While Training Sub-Models
    * Is it possible to achieve communication-efficient sub-model federated learning while also keeping the client’s sub-model choice private? One promising approach is to use PIR for private sub-model download, while aggregating model updates using a variant of secure aggregation optimized for sparse vectors
    * 加噪声可能让sparse的梯度变成dense的

  * User Perception
    * Understanding Privacy Needs for Particular Analysis Tasks
    * Behavioral Research to Elicit Privacy Preferences


##### 5. Defending Against Attacks and Failures

* Adversarial Attacks on Model Performance
  * on model performance, not on data inference
  * These attacks can be broadly classified into training-time attacks (poi-soning attacks) and inference-time attacks (evasion attacks).
  * Goals and Capabilities of an Adversary (Table 11)
    * untargeted attacks (model downgrade) and targeted attacks (backdoor)
  * Model Update Poisoning
    * Untargeted and Byzantine attacks
    * Byzantine-resilient defenses
    * Targeted model update attacks
  * Data Poisoning Attacks
    * Data poisoning and Byzantine-robust aggregation
    * Data sanitization and network pruning
  * Inference-Time Evasion Attacks
  * Defensive Capabilities from Privacy Guarantees
    * The service provider can bound the contribution of any individual client to the overall model by (1) enforcing a norm constraint on the client model update (e.g. by clipping the client updates), (2) aggregating the clipped updates, (3) and adding Gaussian noise to the aggregate.
* Non-Malicious Failure Modes
  * Client reporting failures: unresponsive clients --> Secure Agg
    * select more devices than required within each round.
    * improve the efficiency of SecAgg
    * More speculatively, it may be possible to perform versions of SecAgg that aggregate over multiple computation rounds. This would allow straggler nodes to be included in subsequent rounds, rather than dropping out of the current round altogether.
  * Data pipeline failures
    * GAN 生成 debug 数据
  * Noisy model updates
    * noisy features[350] and noisy labels[356]

* Exploring the Tension between Privacy and Robustness
  * SLSGD: Secure and Efficient Distributed On-device Machine Learning

##### 6. Ensuring Fairness and Addressing Sources of Bias
individual fairness、demographic fairness、counterfactual fairness

* Bias in Training Data
  * e.g. minorities, day-shift vs night-shift work schedules
  * 《Fair resource allocation in federated learning》“a more fair distribution of the model performance across devices”, is employed in [302].
* Fairness Without Access to Sensitive Attributes
  * distributionally-robust optimization (DRO) which optimizes for the worst- case outcome across all individuals during training [225], and via multicalibration, which calibrates for fairness across subsets of the training data [232].
* Fairness, Privacy, and Robustness
  * the ideal of fairness seems to be in tension with the notions of privacy for which FL seeks to provide guarantees: differentially-private learning typically seeks to obscure individually-identifying characteristics, while fairness often requires knowing individuals’ membership in sensitive groups in order to measure or ensure fair predictions are being made.
* Leveraging Federation to Improve Model Diversity
* Federated Fairness: New Opportunities and Challenges
  * 设备特征进模型：For example, federated learning can introduce new sources of bias through the decision of which clients to sample based on considerations such as connection type/quality, device type, location, activity patterns, and local dataset size [81] 
  * centralized方法引入FL
    * Fairness without demograph- ics in repeated loss minimization, ICML 2018

##### 7. Addressing System Challenges

* Platform Development and Deployment Challenges
  * Code Deployment
  * Monitoring and Debugging
    * 端上debug麻烦，主要是log要求严格、不能访问原始input、难以追踪aggregate前的特定设备
    * 用federated analysis技术debug
* System Induced Bias
  * Device Availability Profiles
  * Examples of System Induced Bias
    * selection bias
      * In effect, devices active only at either fleet-wide availability peaks or troughs may be under-represented.
    * survival bias
      * biased towards devices with better network connections, faster processors, lower CPU load, and less data to process
      * For instance, language models may over-represent demographics that have high quality internet connections or high end devices; and ranking models may not incorporate enough contributions from high engagement users who produce a lot of training data and hence longer training times
  * Open Challenges in Quantifying and Mitigating System Induced Bias
    * A useful proxy metric for bias is to study the expected rate of contribution of a device to federated learning
    * device model, network connectivity, location 影响 contribution rate ---> post- stratification[312] or stratified sampling
    * base the weight of a contribution solely on a device’s past contribution profile
      * checkin不频繁的场景需要考虑clustering
* System Parameter Tuning
  * Goals
    * Model Performance
    * Convergence speed
    * Throughput (e.g. number of rounds, amount of data, or number of devices)
    * Model fairness, privacy and robustness (see section 6.3)
    * Resource use on server and clients
  * Various controls
    * Clients per round
    * Server-side scheduling
      * ideal resource assignment should be fair, avoid starvation, minimize wait times, and support relative priorities all at once.
    * Device-side scheduling
      * One extreme is to connect to the server and run computations as often as possible, leading to high load and resource use on both server and devices. Another choice are fixed intervals, but they need to be adjusted to reflect external factors such as number of devices overall and per round
      * pace steering [81]
* On-Device Runtime
  * tf作为device runtime的缺点
    * It offers no easy path to devices for alternative front ends such as PyTorch [370], JAX [86] or CNTK [410].
    * The runtime is not developed or optimized for resource constrained environments, incurring a large binary size, high memory use and comparatively low performance.
    * The intermediate representation GraphDef used by TensorFlow is not standardized or stable, and version skew between the frontend and older on-device backends causes frequent compatibility chal- lenges.
* The Cross-Silo Setting
  * allowing for authentication and verification, accounting, and contractually enforced penalties for misbehavior
  * 难点：
    * How data is generated, pre-processed and labeled.
    * Which software at which version powers training.
    * The approval process for how data may or may not be used.
      * establish data annotations: limiting the use of certain data to specific models, or encoding minimum aggregation requirements such as “require at least M clients per round
  * vertical partitioning
    * Learning with feature-partitioned data may require different communication patterns and additional processing steps e.g. for entity alignment and dealing with missing features



#### Async FL

* [Asynchronous Federated Learning](https://medium.com/@sukanya.me/asynchronous-federated-learning-3c52c5b1a409)
  * [Federated learning with Buffered Asynchronous aggregation](https://arxiv.org/pdf/2106.06639.pdf)
* 

#### Papers

##### 《Federated Evaluation and Tuning for On-Device Personalization: System Design & Applications》, Apple

https://analyticsindiamag.com/how-apple-tuned-up-federated-learning-for-its-iphones/

Differentiating between FL and FT, they write that Federated Learning(FL) requires model evaluation on held-out federated data. FL learns the parameters of, at times large global neural models. Whereas in Federated Tuning(FT), learning primarily occurs on the central server and is limited to a comparatively small set of personalization algorithm parameters that are evaluated across federated data.

* 场景：ASR system
* federated tuning + DP guarantees (private FL)

2.Related Work

与“Towards”论文的对比

* on-device ML systems and their evaluation and tuning/personalization in a federated setting was the initial goal，重心放在调度、端云交互方式上
* FT的参数大部分在云端共享，device侧只是tuning（可降低攻击风险）

元学习、个性化学习，论文 7、11

which shows that federated model averaging [1] is equivalent to an algorithm called Reptile [12]. Reptile is a form of model-agnostic meta learning (MAML) [13].

3.FE&T of ML Systems

3.1 Motivation

* 一个原则：only ‘sufficiently anonymized’ and ‘siloed’ ML data are available to us. 匿名的含义是不能从特征反推用户身份；siloed含义是无法从多个data silos聚合出单用户的完整数据
* The consideration of federated evaluation (FE) and the attached server side aggregation of individual evaluation results was driven by the fact that on-device evaluation data is non-IID and often rather limited per device. Hence, individual evaluation results suffer from large quantities of uncertainty and only global results, aggregated across many devices
  offer meaningful insight.

3.2 The Challenge of Ground Truth

* 一种解决方案是预估label（新模型输入特征输出word confidences）
* 用户调研满意度：highly dependent on and may be inconsistent across user

4. System Description

4.1 High Level System Design

client: task agnostic

4.2 Device Participation & Data Handling

* device participation: 训练条件包括用户opt-in以及设备条件，可能引入bias，通过AB来判断
* on-device data handling
* results data handling

4.3 Core System

* On-Device components
  * Task descriptor download occurs in random order on a per registered plug-in basis to more evenly distribute the globally available compute. Once all task descriptors for a plug-in are downloaded, the scheduler samples exactly one matching (more details below)
  * 和SysBon的对比：
    * SysBon：devices周期check in，服务端去准入，比如利用reservoir sampling。这一机制称作'pace steering'，是训练推进以及secure aggragation protocol的基础
    * 本文：Task specific sample likelihood approach, configured in server side

* server components
  * SysBon performs aggregation of results in memory.
  * 本文: persist per-device results

4.4 FL specific additions

DP guarantees: 

* individual model updates are encrypted on-device with a training round specific public key

5. On-Device Personalization Applications

* The problem of optimizing n parameters can be implemented as an n-dimensional clustering problem with clusters of size k, where k can be optimized

* word error rate (WER) = #mininum_word_edits/#reference_words
  * 引入 word confidence model (semi-supervised)



##### 《Towards Federated Learning at Scale: System Design》, 2019

1.Introduction

* FL 架构选型：同步/异步训练
  * Federated Averaging algorithm
* 隐私技术：DP、Secure Aggragation


Our work addresses numerous practical issues, these issues are addressed at the communication protocol, de-vice, and server levels:

* device availability that correlates with the local data distribution in complex ways (e.g., time zone dependency);
* unreliable device connectivity and interrupted execution;
* orchestration of lock-step execution across devices with varying availability;
* limited device storage and compute resources. 

2.Protocol

device + server

2.2 Phases: selection -> configuration -> reporting

2.3 Pace Steering

* For large FL populations, pace steering is used to randomize device check-in times, avoiding the “thundering herd” problem, and instructing devices to connect as frequently as needed to run all scheduled FL tasks, but not more.

3.Device

* example store：存label的db，考虑expiration、security

* conf: An application configures the FL runtime by providing an FL population name and registering its example stores. This schedules a periodic FL runtime job using Android’s JobScheduler.

* Communication between the application, the FL runtime, and the application’s example store is implemented via Android’s AIDL IPC mechanism

设备验证：

https://developer.android.com/training/safetynet

4.Server

* The FL server is designed around the Actor Programming Model (Hewitt et al., 1973). Actors are universal primitives of concurrent computation which use message passing as the sole communication mechanism.

* actor：串行执行任务，和其它actor通信，动态创建更多actor

* In-memory aggregation also removes the possibility of attacks within the data center that target persistent logs of per-device updates, because no such logs exist.

6.Secure Aggregation

2017年的隐私技术，不同于DP(differential privacy)

server性能消耗大，支持至多几百个用户，因此在aggregator级别做一次secure aggregation生成中间结果，master aggragator不需要做secure aggregation

7.Tools and Workflow

7.1 Modeling and Simulation

The configuration of tasks is also written in Python and includes runtime parameters such as the optimal number of devices in a round as well as model hyperparameters like learning rate.

7.2 Plan Generation

FL plan: 

* device: device portion, the TensorFlow graph itself, selection criteria for training data in the example store, instructions on how to batch data and how many epochs to run on the device, labels for the nodes in the graph which represent certain computations like loading and saving weights, and so on
* server: the aggregation logic, which is encoded in a similar way

7.3 Versioning, Testing, and Deployment

* For example, the old runtime may be missing a particular TensorFlow in an incompatible way. The FL infrastructure deals with this problem by generating versioned FL plans for each task.

一些数据：

* round completion rate曲线，峰值在凌晨
* 一轮一次训几百个设备，再多会影响效果
* drop out rate: 6~10%，因此select时选130%的设备

10.Related Work

11.Future Work

* Bias
* Convergence Time: 更大规模的device并行训练; online tuning protocol configuration
* Device Scheduling: device上训练哪个具体train session，考虑结合app相关的信息
* Bandwidth
* Federated Computation