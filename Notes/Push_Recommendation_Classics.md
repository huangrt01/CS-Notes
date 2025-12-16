# Push Notification Recommendation Classics

## 1. Volume Control (Push Specific)
*   **Paper**: [Near Real-Time Optimization of Notification Volume](https://www.kdd.org/kdd2018/accepted-papers/view/near-real-time-optimization-of-notification-volume) (LinkedIn, WWW 2018)
*   **Core Problem**: Push 通知的核心挑战不仅是 Ranking（推什么），更是 **Volume Control**（推多少）。过度推送会导致用户关闭通知权限（不可逆的 Churn），这是 Feed 流没有的硬约束。
*   **Methodology**:
    *   **Budget Constrained Optimization**: 将问题建模为带约束的优化问题。目标是最大化 Total Engagement (Sessions)，约束是 "Badness" (Unsubscribes/Disable) < Budget。
    *   **Lagrange Multipliers**: 使用拉格朗日乘子法将约束转化为无约束问题，引入 $\lambda$ 作为“打扰成本”。
    *   **Send Decision**: $Score = P(Click) \times V_{click} - \lambda \times P(Unsubscribe) \times V_{churn}$。只有当 Score > 0 时才发送。
*   **Insight**: Push 推荐必须考虑 Long-term Value (LTV)，不能只看短期 CTR。

## 2. Exploration & Cold Start (Exposure Optimization)
*   **Paper**: [A Contextual-Bandit Approach to Personalized News Article Recommendation](https://arxiv.org/abs/1003.0146) (Yahoo!, WWW 2010)
*   **Core Problem**: 物品冷启动（Cold Start）和动态环境下的探索（Exploration）。在 News/Push 场景中，新物品源源不断，且生命周期短，传统的 Collaborative Filtering 无法处理（因为没有历史交互）。
*   **Methodology**: **LinUCB (Linear Upper Confidence Bound)**
    *   **Contextual**: 利用上下文特征（用户特征 + 物品特征）来预测回报。
    *   **UCB (Upper Confidence Bound)**: 不仅预测点击率的期望值 (Mean)，还预测置信区间 (Variance)。
    *   **Selection**: 选择 $Score = \mu + \alpha \cdot \sigma$ 最高的物品。对于新物品，由于数据少，$\sigma$ 大，会被优先展示（探索）；随着数据积累，$\sigma$ 减小，逐渐过渡到利用（Exploit）。
*   **Insight**: 解决“曝光物品数少”的关键是引入**确定性的探索机制**，而不是随机探索。LinUCB 是工业界解决冷启动最经典的 Baseline。

## 3. Targeting & Incrementality (Uplift Modeling)
*   **Context**: 业界（如美团、阿里）在 Push 场景中常采用 **Uplift Modeling**（因果推断）来解决 "Targeting" 问题。
*   **Core Problem**: 传统的 CTR 模型无法区分 "Sure Things"（自然转化用户）。对这类用户发 Push 是浪费预算且增加打扰。
*   **Methodology**: **Uplift Modeling**
    *   **Goal**: 预测 $Lift = P(Click|Treatment) - P(Click|Control)$，即 Push 带来的**增量**点击概率。
    *   **User Segmentation**:
        *   **Persuadables** (高 Lift): 只有推了才点（**核心目标**）。
        *   **Sure Things** (高 CTR, 低 Lift): 不推也会点（应减少推送）。
        *   **Lost Causes** (低 CTR, 低 Lift): 推不推都不点。
        *   **Do Not Disturbs** (负 Lift): 推了反而反感（绝对不推）。
*   **Insight**: Push 的核心价值在于挖掘 **Persuadables**，从而提升 DAU 的**增量** (Incremental Gain)，而非仅仅提升 CTR。
