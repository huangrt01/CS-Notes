# 知识库

本文档介绍如何在 VeADK 中使用知识库。

## 导入

```python
from veadk.knowledgebase import KnowledgeBase
```

## 定义

通过 `KnowledgeBase` 类可以定义一个知识库，并挂载到智能体上。

```python
from veadk.knowledgebase import KnowledgeBase

# 定义知识库
knowledgebase = KnowledgeBase(
    name="my_knowledgebase",
    description="A knowledge base about ...",
    backend="viking",
    index=app_name,
)

agent = Agent(knowledgebase=knowledgebase)
```

其中，`backend` 为知识库后端，当前支持 `viking` 后端。`name` 为知识库名称，`description` 为知识库描述，你需要根据业务场景和知识库内容，来定义一个有意义的名称和描述。
