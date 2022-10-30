[toc]

## Machine Learning

Materials

* http://neuralnetworksanddeeplearning.com/

### ML Basics

[为什么过多的特征（feature）导致过拟合（over-fitting)？ - Dr.Shiki的回答 - 知乎](https://www.zhihu.com/question/47375421/answer/306771331)

#### Algorithms

* crossentropy、KL散度、logistic regression、softmax
  * KL散度 ---> CE loss: [看得见的信息论-为什么用交叉熵作为逻辑回归的代价函数](https://zhuanlan.zhihu.com/p/31207556)
  * logistic regression ---> softmax
  * CE loss + softmax ---> 极其简洁的梯度形式
    * [求导推导](https://zhuanlan.zhihu.com/p/27223959)
    *  <img src="https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial%20l_%7BCE%7D%7D%7B%5Cpartial%20a_j%7D%3Dy_j%20-t_j" alt="\frac{\partial l_{CE}}{\partial a_j}=y_j -t_j" class="ee_img tr_noresize" eeimg="1"> 

* XGBoost: gradient boosted trees works by combining predictions from many simple models, each of which tries to address the weaknesses of the previous models. By doing this the collection of simple models can actually outperform large, complex models.



* Feature Bagging

  * offering a potentially useful way of managing the bias-variance tradeoff

  * We were also interested in this as a potentially useful way to further parallelize training

  * 《An experimental comparison of three methods for constructing ensembles of decision trees: Bagging, boosting, and randomization》


* Dropout
  * 保证training/serving一致性：training或serving时scale
  * In the dense setting, dropout serves to separate effects from strongly correlated features, resulting in a more robust classifier. But in our sparse, noisy setting adding in dropout appears to simply reduce the amount of data available for learning. 《Ad Click Prediction: a View from the Trenches》



[灾难遗忘现象](https://en.wikipedia.org/wiki/Catastrophic_interference)



#### Optimizer

##### Optimization Problem

* Total Error = Optimization Error + Representation Error
*  <img src="https://www.zhihu.com/equation?tex=F%28w_%7Balg%7D%29%20%3D%20F%28w_%7Balg%7D%29-F%28w_%2A%29%20%2B%20F%28w_%2A%29" alt="F(w_{alg}) = F(w_{alg})-F(w_*) + F(w_*)" class="ee_img tr_noresize" eeimg="1"> 
*  <img src="https://www.zhihu.com/equation?tex=F%28w_%2A%29%20%5Cequiv%20%5Cfrac%7B1%7D%7Bn%7D%20%5Csum_%7Bi%20%5Cin%20%5Bn%5D%7D%20l%28h_%7Bw%7D%28x_i%29%2C%20y_i%29%20" alt="F(w_*) \equiv \frac{1}{n} \sum_{i \in [n]} l(h_{w}(x_i), y_i) " class="ee_img tr_noresize" eeimg="1"> 
  * 模型预估误差均值，取决于模型结构

##### GD: 1st-order method

* 梯度下降： <img src="https://www.zhihu.com/equation?tex=w_t%20%5Cleftarrow%20w_%7Bt-1%7D%20-%20%5Ceta%20%5Cnabla%20F%28w_%7Bt-1%7D%29" alt="w_t \leftarrow w_{t-1} - \eta \nabla F(w_{t-1})" class="ee_img tr_noresize" eeimg="1"> 
* Explanation: 一阶泰勒展开

<img src="https://www.zhihu.com/equation?tex=%5Cmu%20%5Cleq%20%5Cfrac%7B%5C%7C%5Cnabla%20f%28a%29%20-%20%5Cnabla%20f%28b%29%5C%7C%7D%7B%5C%7Ca-b%5C%7C%7D%20%5Cleq%20L%2C%20%5Cforall%20a%2Cb%20%5Cin%20%5CR%5Ed" alt="\mu \leq \frac{\|\nabla f(a) - \nabla f(b)\|}{\|a-b\|} \leq L, \forall a,b \in \R^d" class="ee_img tr_noresize" eeimg="1">
  * 强凸：梯度变化率有下界
  * Lipchitz continuous gradient：梯度变化率有上界
* Note:
  * 令下标趋近，这个上下界本质是Hessian: f''(b) 的上下界

<img src="https://www.zhihu.com/equation?tex=F%28w_%7Bt%2B1%7D%29%20-%20F%28w_%2A%29%20%5Cleq%20%281-%5Cfrac%7B%5Cmu%7D%7BL%7D%29%20%5Cleft%28%20F%28w_t%29%20-%20F%28w_%2A%29%20%5Cright%29%5E1" alt="F(w_{t+1}) - F(w_*) \leq (1-\frac{\mu}{L}) \left( F(w_t) - F(w_*) \right)^1" class="ee_img tr_noresize" eeimg="1">

<img src="https://www.zhihu.com/equation?tex=F%28w%29%24%24%20%28there%20are%20other%20types%20of%20convergence%29%0A%0A%23%23%23%23%23%20Newton%27s%20method%3A%202nd-order%20method%0A%0A%2A%20Gradient%20descent%3A%20%24%20%3Cimg%20src%3D%22https%3A//www.zhihu.com/equation%3Ftex%3Dw_t%20%5Cleftarrow%20w_%7Bt-t%7D%20-%20%5Ceta%20%28%5Cnabla%5E2%20F%28w_%7Bt-1%7D%29%29%5E%7B-1%7D%20%5Cnabla%20F%28w_%7Bt-1%7D%29" alt="F(w)" alt="w_t \leftarrow w_{t-t} - \eta (\nabla^2 F(w_{t-1}))^{-1} \nabla F(w_{t-1})" alt="F(w)" class="ee_img tr_noresize" eeimg="1"> $ (there are other types of convergence)

##### Newton's method: 2nd-order method

* Gradient descent: $$w_t \leftarrow w_{t-t} - \eta (\nabla^2 F(w_{t-1}))^{-1} \nabla F(w_{t-1})" class="ee_img tr_noresize" eeimg="1">


<img src="https://www.zhihu.com/equation?tex=F%28w%29%20%3D%20%5Cfrac%7B1%7D%7B2%7D%20w%5E%5Ctop%20D%20w%24%24%20%2C%20D%20is%20a%20diagonal%20matrix%20with%20all%20positive%20diagonal%20elements%0A%0A%20%20%2A%20%E4%B8%BE%E4%B8%80%E4%B8%AA%E4%BE%8B%E5%AD%90%EF%BC%8C%E5%AF%B9%E6%AF%94GD%E5%92%8CNewton%E6%B3%95%E7%9A%84%E6%94%B6%E6%95%9B%E9%80%9F%E5%BA%A6%0A%20%20%2A%20GD%3A%20%E8%AE%A1%E7%AE%97%20%24%20%3Cimg%20src%3D%22https%3A//www.zhihu.com/equation%3Ftex%3DF%28w_%7Bt%2B1%7D%29%5Cle%20F%28w_t%29" alt="F(w_{t+1})\le F(w_t)" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=%20%E7%9A%84%E6%81%92%E6%88%90%E7%AB%8B%E6%9D%A1%E4%BB%B6%E6%98%AF%20" alt=" 的恒成立条件是 " class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=%5Ceta%20%5Clt%20%5Cfrac%7B2%7D%7Bmax_iD_%7Bii%7D%7D" alt="F(w) = \frac{1}{2} w^\top D w" alt="\eta \lt \frac{2}{max_iD_{ii}}" alt="F(w) = \frac{1}{2} w^\top D w" class="ee_img tr_noresize" eeimg="1"> $ , D is a diagonal matrix with all positive diagonal elements

  * 举一个例子，对比GD和Newton法的收敛速度
  * GD: 计算 $ <img src="https://www.zhihu.com/equation?tex=F%28w_%7Bt%2B1%7D%29%5Cle%20F%28w_t%29" alt="F(w_{t+1})\le F(w_t)" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=%20%E7%9A%84%E6%81%92%E6%88%90%E7%AB%8B%E6%9D%A1%E4%BB%B6%E6%98%AF%20" alt=" 的恒成立条件是 " class="ee_img tr_noresize" eeimg="1"> $\eta \lt \frac{2}{max_iD_{ii}}" class="ee_img tr_noresize" eeimg="1">
  * Newton法，在例子中一步收敛了
  * 类似于在不同维度使用 Adaptive Learning Rate（D^-1，反映gradient的变化率）的效果
  * Quadratic Convergence

* Convergence


<img src="https://www.zhihu.com/equation?tex=%5Cmu%24%20%3Cimg%20src%3D%22https%3A//www.zhihu.com/equation%3Ftex%3D%20and%20Lipschtiz%20hessian%20" alt=" and Lipschtiz hessian " class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=H" alt="\mu" alt="H" alt="\mu" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=%20and%20Lipschtiz%20hessian%20" alt=" and Lipschtiz hessian " class="ee_img tr_noresize" eeimg="1"> $H" class="ee_img tr_noresize" eeimg="1">

  * Quadratic convergence: 


<img src="https://www.zhihu.com/equation?tex=%5C%7Cw_%7Bt%2B1%7D%20-%20w_%2A%5C%7C%20%5Cleq%20%5Cfrac%7BH%7D%7B%5Cmu%7D%20%5C%7Cw_t%20-%20w_%2A%5C%7C%5E2" alt="\|w_{t+1} - w_*\| \leq \frac{H}{\mu} \|w_t - w_*\|^2" class="ee_img tr_noresize" eeimg="1">


<img src="https://www.zhihu.com/equation?tex=%5C%7Cw_0-w_%2A%5C%7C%20%5Cleq%20%5Cfrac%7B%5Cmu%7D%7B2H%7D" alt="\|w_0-w_*\| \leq \frac{\mu}{2H}" class="ee_img tr_noresize" eeimg="1">

##### Polyak Momentum


<img src="https://www.zhihu.com/equation?tex=w_t%20%5Cleftarrow%20w_%7Bt-1%7D%20-%20%5Ceta%20%5Cnabla%20F%28w_%7Bt-1%7D%29%20%2B%20%5Cbeta%28w_%7Bt-1%7D%20-%20w_%7Bt-2%7D%29" alt="w_t \leftarrow w_{t-1} - \eta \nabla F(w_{t-1}) + \beta(w_{t-1} - w_{t-2})" class="ee_img tr_noresize" eeimg="1">
* The formula above is equivalent to

<img src="https://www.zhihu.com/equation?tex=v_t%20%5Cleftarrow%20%5Ceta%20%5Cnabla%20F%28w_%7Bt-1%7D%29%20%2B%20%5Cbeta%20v_%7Bt-1%7D%24%20%3Cimg%20src%3D%22https%3A//www.zhihu.com/equation%3Ftex%3D%2C%20" alt=", " class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=w_t%20%5Cleftarrow%20w_%7Bt-1%7D%20-%20v_t" alt="v_t \leftarrow \eta \nabla F(w_{t-1}) + \beta v_{t-1}" alt="w_t \leftarrow w_{t-1} - v_t" alt="v_t \leftarrow \eta \nabla F(w_{t-1}) + \beta v_{t-1}" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=%2C%20" alt=", " class="ee_img tr_noresize" eeimg="1"> $w_t \leftarrow w_{t-1} - v_t" class="ee_img tr_noresize" eeimg="1">

<img src="https://www.zhihu.com/equation?tex=%5Ceta%24%20%3Cimg%20src%3D%22https%3A//www.zhihu.com/equation%3Ftex%3D%20inside%20momentum%20variable%20" alt=" inside momentum variable " class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=v" alt="\eta" alt="v" alt="\eta" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=%20inside%20momentum%20variable%20" alt=" inside momentum variable " class="ee_img tr_noresize" eeimg="1"> $v" class="ee_img tr_noresize" eeimg="1">

- But we can also put learning rate outside the momentum:

<img src="https://www.zhihu.com/equation?tex=v_t%20%5Cleftarrow%20%5Cnabla%20F%28w_%7Bt-1%7D%29%20%2B%20%5Cbeta%20v_%7Bt-1%7D%24%20%3Cimg%20src%3D%22https%3A//www.zhihu.com/equation%3Ftex%3D%2C%20" alt=", " class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=w_t%20%5Cleftarrow%20w_%7Bt-1%7D%20-%20%5Ceta%20v_t" alt="v_t \leftarrow \nabla F(w_{t-1}) + \beta v_{t-1}" alt="w_t \leftarrow w_{t-1} - \eta v_t" alt="v_t \leftarrow \nabla F(w_{t-1}) + \beta v_{t-1}" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=%2C%20" alt=", " class="ee_img tr_noresize" eeimg="1"> $w_t \leftarrow w_{t-1} - \eta v_t" class="ee_img tr_noresize" eeimg="1">
  - Caution: these 2 formulas will be different if the learning rate changes (warmup, decay)

##### Nesterov Momentum

- Concept: **lookahead** to get a better gradient estimation

- 理论上是两步，本方法基于最新model计算gradient，解决半步的staleness

* pytorch实际实现中，保留的是lookhead model

##### SGD: stochastic methods


<img src="https://www.zhihu.com/equation?tex=%5Cmin_%7Bt%7D%20E%5Cleft%5B%20%5C%7C%5Cnabla%20F%28w_%7Bt-1%7D%29%5C%7C%5E2%5Cright%5D%20%5Cleq%20%5Cfrac%7B1%7D%7BT%7D%20%5Csum_%7Bt%3D1%7D%5ET%20E%5Cleft%5B%20%5C%7C%5Cnabla%20F%28w_%7Bt-1%7D%29%5C%7C%5E2%20%5Cright%5D%20%5Cleq%20%5Cfrac%7B2E%5BF%28w_%7B0%7D%29%20-%20F%28w_%2A%29%5D%7D%7B%5Ceta%20T%7D%20%2B%20%5Cfrac%7BL%5Ceta%20V_1%7D%7Bb%7D" alt="\min_{t} E\left[ \|\nabla F(w_{t-1})\|^2\right] \leq \frac{1}{T} \sum_{t=1}^T E\left[ \|\nabla F(w_{t-1})\|^2 \right] \leq \frac{2E[F(w_{0}) - F(w_*)]}{\eta T} + \frac{L\eta V_1}{b}" class="ee_img tr_noresize" eeimg="1">
* 2 parts of error:
  - Escape from initial point to optimal
  - Variance (reduced by batch size)

<img src="https://www.zhihu.com/equation?tex=%5Ceta%5Cpropto%5Cfrac%7B1%7D%7B%5Csqrt%7BT%7D%7D" alt="\eta\propto\frac{1}{\sqrt{T}}" class="ee_img tr_noresize" eeimg="1">

<img src="https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%7BT%7D%20%5Csum_%7Bt%3D1%7D%5ET%20E%5Cleft%5B%20%5C%7C%5Cnabla%20F%28w_%7Bt-1%7D%29%5C%7C%5E2%20%5Cright%5D%20%5Cleq%20O%28%5Cfrac%7B1%7D%7B%5Csqrt%7BT%7D%7D%29" alt="\frac{1}{T} \sum_{t=1}^T E\left[ \|\nabla F(w_{t-1})\|^2 \right] \leq O(\frac{1}{\sqrt{T}})" class="ee_img tr_noresize" eeimg="1">


<img src="https://www.zhihu.com/equation?tex=%5Ceta_t%20%5Cpropto%20%5Cfrac%7B1%7D%7B%5Csqrt%7Bt%7D%7D" alt="\eta_t \propto \frac{1}{\sqrt{t}}" class="ee_img tr_noresize" eeimg="1">


<img src="https://www.zhihu.com/equation?tex=%5Cnabla%20F%28w%29%20%3D%200%24%24%2C%20could%20be%20a%20saddle%20point%20or%20local%20minimum%2C%20not%20necessarily%20a%20global%20minimum%0A%0A%23%23%23%23%23%20Federated%20Averaging%0A%0A%E3%80%8AAdvances%20and%20open%20problems%20in%20federated%20learning%E3%80%8Bp22%0A%0A%0A%23%23%23%23%23%20AdaGrad%3A%20a%20natural%20learning%20rate%20decay%0A%0A-%20Algorithm%3A%0A%0A%20%20-%20In%20step%20%24%20%3Cimg%20src%3D%22https%3A//www.zhihu.com/equation%3Ftex%3Dt" alt="\nabla F(w) = 0" alt="t" alt="\nabla F(w) = 0" class="ee_img tr_noresize" eeimg="1"> $, could be a saddle point or local minimum, not necessarily a global minimum

##### Federated Averaging

《Advances and open problems in federated learning》p22


##### AdaGrad: a natural learning rate decay

- Algorithm:

  - In step $$t" class="ee_img tr_noresize" eeimg="1">

<img src="https://www.zhihu.com/equation?tex=g_t%20%5Cequiv%20%5Cnabla%20f%28w_%7Bt-1%7D%29" alt="g_t \equiv \nabla f(w_{t-1})" class="ee_img tr_noresize" eeimg="1">

<img src="https://www.zhihu.com/equation?tex=v_t%20%5Cleftarrow%20v_%7Bt-1%7D%20%2B%20g_t%20%5Ccirc%20g_t" alt="v_t \leftarrow v_{t-1} + g_t \circ g_t" class="ee_img tr_noresize" eeimg="1">

<img src="https://www.zhihu.com/equation?tex=w_%7Bt%7D%20%5Cleftarrow%20w_%7Bt-1%7D%20-%20%5Ceta%20%5Cfrac%7Bg_t%7D%7B%5Csqrt%7Bv_t%20%2B%20%5Cepsilon%7D%7D%20" alt="w_{t} \leftarrow w_{t-1} - \eta \frac{g_t}{\sqrt{v_t + \epsilon}} " class="ee_img tr_noresize" eeimg="1">

- 本质：


<img src="https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%7Bt%7D%5Csum_t%20g_t%20%5Ccirc%20g_t%24%24%20to%20estimate%20hessian%20%0A%0A%20%20-%20a%20%2A%2Anaturally%20decay%20learning%20rate%2A%2A%20%24%20%3Cimg%20src%3D%22https%3A//www.zhihu.com/equation%3Ftex%3D%5Cfrac%7B%5Ceta%7D%7B%5Csqrt%7Bt%7D%7D" alt="\frac{1}{t}\sum_t g_t \circ g_t" alt="\frac{\eta}{\sqrt{t}}" alt="\frac{1}{t}\sum_t g_t \circ g_t" class="ee_img tr_noresize" eeimg="1"> $ to estimate hessian 

  - a **naturally decay learning rate** $$\frac{\eta}{\sqrt{t}}" class="ee_img tr_noresize" eeimg="1">

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

- Algorithm:

<img src="https://www.zhihu.com/equation?tex=t" alt="t" class="ee_img tr_noresize" eeimg="1">

<img src="https://www.zhihu.com/equation?tex=g_t%20%5Cequiv%20%5Cnabla%20f%28w_%7Bt-1%7D%29" alt="g_t \equiv \nabla f(w_{t-1})" class="ee_img tr_noresize" eeimg="1">

<img src="https://www.zhihu.com/equation?tex=m_t%20%5Cleftarrow%20%5Cbeta_1%20m_%7Bt-1%7D%20%2B%20%281-%5Cbeta_1%29%20g_t" alt="m_t \leftarrow \beta_1 m_{t-1} + (1-\beta_1) g_t" class="ee_img tr_noresize" eeimg="1">

<img src="https://www.zhihu.com/equation?tex=v_t%20%5Cleftarrow%20%5Cbeta_2%20v_%7Bt-1%7D%20%2B%20%281-%5Cbeta_2%29%20g_t%20%5Ccirc%20g_t" alt="v_t \leftarrow \beta_2 v_{t-1} + (1-\beta_2) g_t \circ g_t" class="ee_img tr_noresize" eeimg="1">

<img src="https://www.zhihu.com/equation?tex=%5Chat%7Bm%7D_t%20%5Cleftarrow%20%5Cfrac%7Bm_t%7D%7B1-%5Cbeta_1%5Et%7D" alt="\hat{m}_t \leftarrow \frac{m_t}{1-\beta_1^t}" class="ee_img tr_noresize" eeimg="1">
    - 动机是没有 learning rate decay
    - 可尝试去掉，等价于learning rate warmup，会有点接近AdaGrad

<img src="https://www.zhihu.com/equation?tex=%5Chat%7Bv%7D_t%20%5Cleftarrow%20%5Cfrac%7Bv_t%7D%7B1-%5Cbeta_2%5Et%7D" alt="\hat{v}_t \leftarrow \frac{v_t}{1-\beta_2^t}" class="ee_img tr_noresize" eeimg="1">

<img src="https://www.zhihu.com/equation?tex=w_%7Bt%7D%20%5Cleftarrow%20w_%7Bt-1%7D%20-%20%5Ceta%20%5Cfrac%7B%5Chat%7Bm%7D_t%7D%7B%5Csqrt%7B%5Chat%7Bv%7D_t%7D%20%2B%20%5Cepsilon%7D%20" alt="w_{t} \leftarrow w_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} " class="ee_img tr_noresize" eeimg="1">
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

<img src="https://www.zhihu.com/equation?tex=%5Cmax%5C%7B%5Chat%7Bv%7D_t%2C%20%5Chat%7Bv%7D_%7Bt-1%7D%2C%20%5Cldots%20%5Chat%7Bv%7D_1%5C%7D%24%20%3Cimg%20src%3D%22https%3A//www.zhihu.com/equation%3Ftex%3Dinstead%20of%20" alt="instead of " class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=%5Chat%7Bv%7D_t" alt="\hat{v}_t" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=to%20guarantee%20decreasing%20" alt="to guarantee decreasing " class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Ceta_t%7D%7B%5Csqrt%7B%5Chat%7Bv%7D_t%7D%20%2B%20%5Cepsilon%7D" alt="\frac{\eta_t}{\sqrt{\hat{v}_t} + \epsilon}" class="ee_img tr_noresize" eeimg="1"> $: AMSGrad
    - Take $ <img src="https://www.zhihu.com/equation?tex=%5Cbeta_2%20%5Cpropto%201-%5Cfrac%7B1%7D%7Bt%7D" alt="\beta_2 \propto 1-\frac{1}{t}" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=%2C%20approaches%201%20when%20" alt=", approaches 1 when " class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=t" alt="t" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=%20approaches%20infinity%2C%20%20" alt=" approaches infinity,  " class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=v_t" alt="v_t" class="ee_img tr_noresize" eeimg="1"> $barely changes at the end
- Note：
  - sparse 部分不适用 Adam：滑动平均用到了历史信息
  - 配合 slow start 技术，前期并发数缓慢增大

##### RMSProp

* RMSProp: Adam with $ <img src="https://www.zhihu.com/equation?tex=%5Cbeta_1%3D0" alt="\beta_1=0" class="ee_img tr_noresize" eeimg="1"> $, without any bias correction

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
  - In step $ <img src="https://www.zhihu.com/equation?tex=t" alt="t" class="ee_img tr_noresize" eeimg="1"> $:
  - Compute update based on any optimizer: $ <img src="https://www.zhihu.com/equation?tex=u_t" alt="\max\{\hat{v}_t, \hat{v}_{t-1}, \ldots \hat{v}_1\}" alt="u_t" alt="\max\{\hat{v}_t, \hat{v}_{t-1}, \ldots \hat{v}_1\}" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=instead%20of%20" alt="instead of " class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=%5Chat%7Bv%7D_t" alt="\hat{v}_t" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=to%20guarantee%20decreasing%20" alt="to guarantee decreasing " class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Ceta_t%7D%7B%5Csqrt%7B%5Chat%7Bv%7D_t%7D%20%2B%20%5Cepsilon%7D" alt="\frac{\eta_t}{\sqrt{\hat{v}_t} + \epsilon}" class="ee_img tr_noresize" eeimg="1"> $: AMSGrad
    - Take $ <img src="https://www.zhihu.com/equation?tex=%5Cbeta_2%20%5Cpropto%201-%5Cfrac%7B1%7D%7Bt%7D" alt="\beta_2 \propto 1-\frac{1}{t}" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=%2C%20approaches%201%20when%20" alt=", approaches 1 when " class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=t" alt="t" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=%20approaches%20infinity%2C%20%20" alt=" approaches infinity,  " class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=v_t" alt="v_t" class="ee_img tr_noresize" eeimg="1"> $barely changes at the end
- Note：
  - sparse 部分不适用 Adam：滑动平均用到了历史信息
  - 配合 slow start 技术，前期并发数缓慢增大

##### RMSProp

* RMSProp: Adam with $ <img src="https://www.zhihu.com/equation?tex=%5Cbeta_1%3D0" alt="\beta_1=0" class="ee_img tr_noresize" eeimg="1"> $, without any bias correction

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
  - In step $ <img src="https://www.zhihu.com/equation?tex=t" alt="t" class="ee_img tr_noresize" eeimg="1"> $:
  - Compute update based on any optimizer: $$u_t" class="ee_img tr_noresize" eeimg="1">

<img src="https://www.zhihu.com/equation?tex=u_t%20%3D%20g_t" alt="u_t = g_t" class="ee_img tr_noresize" eeimg="1">

<img src="https://www.zhihu.com/equation?tex=u_t%20%3D%20%5Cfrac%7B%5Chat%7Bm%7D_t%7D%7B%5Csqrt%7B%5Chat%7Bv%7D_t%7D%20%2B%20%5Cepsilon%7D%20" alt="u_t = \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} " class="ee_img tr_noresize" eeimg="1">

<img src="https://www.zhihu.com/equation?tex=u_t%3D%5Cfrac%7Bg_t%7D%7B%5Csqrt%7Bv_t%20%2B%20%5Cepsilon%7D%7D%20" alt="u_t=\frac{g_t}{\sqrt{v_t + \epsilon}} " class="ee_img tr_noresize" eeimg="1">
  - Layer-wise normalization:

<img src="https://www.zhihu.com/equation?tex=%5Chat%7Bu%7D_t%20%5Cleftarrow%20%5Cfrac%7B%5C%7Cw_%7Bt-1%7D%5C%7C%7D%7B%5C%7Cu_t%5C%7C%7D%20u_t" alt="\hat{u}_t \leftarrow \frac{\|w_{t-1}\|}{\|u_t\|} u_t" class="ee_img tr_noresize" eeimg="1">
  - Update model:

<img src="https://www.zhihu.com/equation?tex=w_t%20%5Cleftarrow%20w_%7Bt-1%7D%20-%20%5Ceta%20%5Chat%7Bu%7D_t" alt="w_t \leftarrow w_{t-1} - \eta \hat{u}_t" class="ee_img tr_noresize" eeimg="1">

- Intuition:
  - In large-batch training:

<img src="https://www.zhihu.com/equation?tex=%5C%7Cu_t%5C%7C%24%24%20is%20unstable%0A%20%20-%20Using%20large%20learning%20rate%3A%20diverge%20for%20large%20%24%20%3Cimg%20src%3D%22https%3A//www.zhihu.com/equation%3Ftex%3D%5C%7Cu_t%5C%7C" alt="\|u_t\|" alt="\|u_t\|" alt="\|u_t\|" class="ee_img tr_noresize" eeimg="1"> $ is unstable
  - Using large learning rate: diverge for large $$\|u_t\|" class="ee_img tr_noresize" eeimg="1">

<img src="https://www.zhihu.com/equation?tex=%5C%7Cu_t%5C%7C" alt="\|u_t\|" class="ee_img tr_noresize" eeimg="1">
  - LAMB: 

<img src="https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Ceta%20%5C%7Cw_%7Bt-1%7D%5C%7C%7D%7B%5C%7Cu_t%5C%7C%7D" alt="\frac{\eta \|w_{t-1}\|}{\|u_t\|}" class="ee_img tr_noresize" eeimg="1">

<img src="https://www.zhihu.com/equation?tex=%5C%7Cu_t%5C%7C%24%24is%20large%0A%20%20-%20Larger%20when%20%24%20%3Cimg%20src%3D%22https%3A//www.zhihu.com/equation%3Ftex%3D%5C%7Cu_t%5C%7C" alt="\|u_t\|" class="ee_img tr_noresize" eeimg="1"> $is small
  - Normalize $ <img src="https://www.zhihu.com/equation?tex=%5C%7Cu_t%5C%7C" alt="\|u_t\|" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=to%20the%20same%20scale%20of%20" alt="to the same scale of " class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=%5C%7Cw%5C%7C" alt="\|u_t\|" alt="\|w\|" alt="\|u_t\|" class="ee_img tr_noresize" eeimg="1"> $is large
  - Larger when $ <img src="https://www.zhihu.com/equation?tex=%5C%7Cu_t%5C%7C" alt="\|u_t\|" class="ee_img tr_noresize" eeimg="1"> $is small
  - Normalize $ <img src="https://www.zhihu.com/equation?tex=%5C%7Cu_t%5C%7C" alt="\|u_t\|" class="ee_img tr_noresize" eeimg="1">  <img src="https://www.zhihu.com/equation?tex=to%20the%20same%20scale%20of%20" alt="to the same scale of " class="ee_img tr_noresize" eeimg="1"> $\|w\|" class="ee_img tr_noresize" eeimg="1">

- Note:
  - We can apply LAMB normalization to any base optimizer
  - But the learning rate must be re-tuned

#### Validation

holdout validation, cross-validation, leave-one-out validation, etc

```python
train_data, validation_data, test_data = np.split(model_data.sample(frac=1, random_state=1729), [int(0.7 * len(model_data)), int(0.9 * len(model_data))])   # Randomly sort the data then split out first 70%, second 20%, and last 10%
```



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

### Bert

Transformer 具有 field reduce 能力，将 N 个 token reduce 成 M 个 token

model finetune

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



### Search & NLP

* 概念
  * [Bag-of-words(BoW) model](https://en.wikipedia.org/wiki/Bag-of-words_model) 可作为一种信息模型，表示句子或图片，用于衡量相似度或别的用途
  * stop words: 停用词
  * [Tokenization and text normalization](https://www.analyticsvidhya.com/blog/2021/03/tokenization-and-text-normalization/)

* ElasticSearch

  * denormalization
  * inverted index
    * An index is a collection of documents.
    * index: <word -> [documents]>
    * document: <field -> [values]>

  * document oriented tool


### CV

* group convolution
  * only the input channels in the same group are used for computing a given output channel. A group convolution with total Ci input, Co output channels and G groups is essentially G independent convolutions each with d=Ci/G input and Co/G output channels. 
  * depth-wise convolution: Ci=Co=G and consequently group size d=1
* [LBP (local binary patterns)](https://en.wikipedia.org/wiki/Local_binary_patterns)
  * resize到固定大小：大小越大则越准但有噪声，大小越小则误召回率高
  * hamming 距离度量



* [Constrastive Learning: MoCo and SimCLR](https://mp.weixin.qq.com/s/v5p9QA3vDl-WTF3-7shp4g)
* 



### Reinforce Learning

[深度强化学习（一）强化学习概述 - iker peng的文章 - 知乎](https://zhuanlan.zhihu.com/p/22542101)

[深度强化学习系列（二）强化学习基础 - iker peng的文章 - 知乎](https://zhuanlan.zhihu.com/p/23436744)



### 特征压缩

According to [JL-lemma](https://en.wikipedia.org/wiki/Johnson–Lindenstrauss_lemma), [random projection](https://en.wikipedia.org/wiki/Random_projection) reduces the dimensionality of data while approximately preserving the pairwise distances between data points.

* 压缩 feature num 而非压缩 embedding size (PCA, SVD)
* 输入 feature，输出 concated_compressed_embedding + dot_products
* 变种：instance-wise (Y adaptive to X), implicit-order, GroupCDot, CDotFlex
* 对比：比[AutoInt](https://arxiv.org/pdf/1810.11921.pdf)计算量小；比[DCN-V2](https://arxiv.org/pdf/2008.13535.pdf)线上效果好

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

![horovod-rank](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/Machine-Learning/horovod-rank.png)

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
  * 提高学习速度并减少使用 Dropout 的需求
  * 想法是针对每批数据对所有层的输入 进行规一化（这比简单地只对输入数据集进行规一化更为复杂）
  * Ghost BN
    * 计算更小批量的统计数据（“ghost 批量”）引入其他噪声
    * 按 GPU 逐个单独执行批量归一化
  * 将噪声添加至梯度
    * 确保权重更新的协方差随着批量大小的变动保持不变 
    * 不会改变权重更新的平均值 

<img src="https://www.zhihu.com/equation?tex=%5Chat%7Bg%7D%3D%5Cfrac%7B1%7D%7BM%7D%5Csum%5E%7BN%7D_%7Bn%5Cin%20B%7Dg_n%20z_n" alt="\hat{g}=\frac{1}{M}\sum^{N}_{n\in B}g_n z_n" class="ee_img tr_noresize" eeimg="1">
  * 更长的高学习率训练时间
  * 增加批量大小代替学习率衰减
  * LARS – 按层自适应学习率调整
    *  [LARS论文](https://arxiv.org/abs/1904.00962): 大LR -> LR warm-up -> LARS，只是能保证大batch训练能训，关于效果问题，作者认为“increasing the batch does not give much additional gradient information comparing to smaller batches.”
    *  [LARC](https://github.com/NVIDIA/apex/blob/master/apex/parallel/LARC.py): 带梯度裁剪的分层自适应学习率，以具有动力的SGD作为基础优化器
    *  [LAMB](https://arxiv.org/abs/1904.00962): 分层自适应学习率，以 Adam 作为基础优化器，在BERT等语言模型上比LARC更成功
    *  [NovoGrad](https://arxiv.org/abs/1905.11286): 按层计算的移动平均值，在几个不同的领域也有不错的表现

![training_result](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/Machine-Learning/training_result.png)

### 术语

NLU: Natural Language Understanding