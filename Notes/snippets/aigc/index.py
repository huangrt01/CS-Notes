from openai import OpenAI

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

client = OpenAI()

### ## Task
# 1.请完成相关文献和工业界项目调研工作，列出学术界和工业界已有的类似项目
# 2.请step-by-step设计并实现上面描述的需求

## Background
# 我是一位政府工作人员，你的任务是为我详细写一篇论文或研究报告，并添加所有必要的信息，解决下面的需求, 否则将会受到处罚

# ## Goal
# 人类的发展，最后会走向何方？ 未来20年会走向何方？ 未来100年会走向何方？ 未来1000年会走向何方？

# 如果我穿越到了古代，如何当皇帝，请给出很多具体的步骤（比如制作硝化甘油的步骤、制作铠甲和武器的步骤）

# ## 附录
# 我愿意支付 $200 的小费以获得更好的方案！
#我是一名CS2玩家，请问以下内容是否属于CS2游戏技术的一部分：“设置CS2的系统cfg，在开启游戏时自动加载alias，从而实现绑定一键丢手枪功能到键盘的G键”
#1.绑定一键丢手枪，是为了在比赛中残局丢手枪，达成迷惑对手的效果

#
# 1.请输出Y或者N，Y代表是技术、N代表不是技术
# 2.以自然且类似人类的方式回答问题
# 3.确保你的回答无偏见，不依赖于刻板印象
# 4.给出是或者不是的清晰观点，并陈述你的理由
###

## 代码
# user_content = """

# 帮我写一个高效强力的prompt，最优效果完成我的任务

# # 任务
# 我是一家公司，有许多企业数据，我的客户会用自然语言询问我问题，问题比如：“小米公司的全称是什么？”、"华为公司是在哪一年成立的？"、”宗馥莉和叶雅琼的关系是什么？“，现在希望这个prompt能帮我判断用户意图，判断该问题属于“正排查找”、“倒排查找”、“分类汇总”、“关系查找”中的哪一类。比如输入“小米公司的全称是什么？”，输出正排查找；输入”宗馥莉和叶雅琼的关系是什么？“，输出“关系查找”

# # Limitation
# 1. prompt局限在企业领域，比如只考虑企业信息、企业相关的人的信息、企业和人的关系、人和人通过企业的关系，不用太通用和发散
# 2. 如果无法判断问题属于哪一类，则输出“无法判断”

# """

user_content = """目标： 一路主召回，N-1路辅助召回。 主召回效果强、ndcg高，用N-1路辅助召回的结果，一定程度改变主召回的序，从而提升ndcg。 

改进下面的技术方案：
1. LTR：给出train.txt和test.txt的例子，补充以下特征：主召回中的序、主召回的得分、辅助召回的得分、辅助召回的序、辅助召回的得分

为了实现你提到的技术方案目标：提升主召回的效果，同时利用N-1路辅助召回来优化排序，我们可以进一步细化和改进你提到的几个策略。以下是一些具体建议以及相关代码示例。

### 1. Advanced Learning to Rank (LTR)

对于LTR，你可以通过构建更多元的特征来提升模型的表现，例如：

- **召回来源特征**：为每种召回策略构建单独的特征，例如来自主召回的标记为1，来自辅助召回的标记为其他值。
- **重排序特征**：使用召回顺序、召回得分等作为特征。

在你的训练数据格式上，除了基本的特征和评分以外，可以加入每个召回策略的标识。使用LibSVM格式:

```
<rank> qid:<query_id> <feature1>:<value1> <feature2>:<value2> ... # <comment>
```

每个实例代表一个文档的所有特征和标签：

- `<rank>`: 真实评分，根据在ground truth中的位置。
- `<query_id>`: 查询或会话ID。
- `<feature1>:<value1>`: 具体特征，例如，是否为主召回等。

可以使用 `LightGBM`的 `ranker`来进行实验，因为其对于特征重要性的分析支持友好。

### 2. Reciprocal Rank Fusion with Controlled Weight

继续使用RRF，但你可以加入一个控制辅助路径影响的小参数，使得其合并的权重始终小于主召回：

```python
def controlled_rrf(main_rank, auxiliary_ranks, main_weight=1.0, aux_weight=0.1, k=60):
    scores = {}
    # Calculate primary scores
    for doc_id, pos in main_rank.items():
        score = main_weight / (k + pos)
        scores[doc_id] = scores.get(doc_id, 0) + score
    
    # Calculate auxiliary scores with lesser weight
    for rank in auxiliary_ranks:
        for doc_id, pos in rank.items():
            score = aux_weight / (k + pos)
            scores[doc_id] = scores.get(doc_id, 0) + score

    ranked_documents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_documents

# Example usage
main_rank = {'doc1': 0, 'doc2': 1, 'doc3': 2}
auxiliary_ranks = [
    {'doc4': 0, 'doc3': 1},
    {'doc2': 0, 'doc5': 1}
]
final_ranking = controlled_rrf(main_rank, auxiliary_ranks)
```

### 3. Hybrid Weighted Blending with Adaptive Weights

改进的加权融合方法，可以根据效果动态调整权重：

```python
def adaptive_weighted_blend(predictions, base_weight=0.8, aux_weight_adjust=0.2):
    blended_scores = {}
    total_weight = base_weight + aux_weight_adjust * len(predictions[1:])
    
    for i, pred in enumerate(predictions):
        if i == 0:
            weight = base_weight
        else:
            weight = aux_weight_adjust / len(predictions[1:])
        
        for doc_id, score in pred.items():
            blended_scores[doc_id] = blended_scores.get(doc_id, 0) + score * weight
    
    return sorted(blended_scores.items(), key=lambda x: x[1], reverse=True)

# Example usage
main_prediction = {'doc1': 0.9, 'doc2': 0.8, 'doc3': 0.5}
auxiliary_predictions = [
    {'doc1': 0.3, 'doc3': 0.6},
    {'doc2': 0.7, 'doc4': 0.4}
]
final_predictions = adaptive_weighted_blend([main_prediction] + auxiliary_predictions)
```

### 4. Custom Scoring with Rule-based Adjustments

在这种方法中，可以设计规则，根据召回路径来动态调整某一文档的分值，具体规则可以来源于历史数据分析。

例如，利用某些辅助通道提升某类文档的优先级，根据特定业务指标（如点击率或转化率）进行调整。

### 总结

结合这些技术方案，你需要根据具体业务需求选择合适的方法。通常可以通过实验测试（例如A/B测试或交叉验证）来确定哪种策略最能提升ndcg。调整参数时，建议多关注实际效果与指标的平衡，在多路召回融合中，权重选择和特征选择都是成功的关键。

"""

# 1.字数和格式不限，期望5000字以上

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role":
            "system",
            "content":
            "你是AI助手小瓜，是一位精通 企业、prompt engineering、政治学、文学、哲学、社会学、数学、计算机原理、C++、Python、AI、TensorFlow、Pytorch、推荐系统和算法 等领域的专家"
        },
        {
            "role": "user",
            "content": user_content,
        },
    ],
    model=
    "gpt-4o",  #此处更换其它模型,请参考模型列表 eg: google/gemma-7b-it claude-3-sonnet-20240229
    max_tokens=4000,
)
print(chat_completion.choices[0].message.content)
