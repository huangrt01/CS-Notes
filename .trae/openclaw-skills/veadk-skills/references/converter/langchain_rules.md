# Langchain 与 VeADK 对应规则

## Agent

Langchain 写法：

```python
agent = create_agent(model=llm)
```

VeADK 写法：

```python
agent = Agent(
    name="...",
    description="...",
    instruction="...", # 系统提示词
    model_name="..." # 模型名称
)
```

## 工具

Langchain 写法：

```python
agent = create_agent(
    model=llm,
    context_schema=Context,
    tools=[load_knowledgebase], # 工具挂载
)
```

VeADK 写法：

```python
agent = Agent(
    name="...",
    description="...",
    instruction="...", # 系统提示词
    model_name="...", # 模型名称
    tools=[load_knowledgebase], # 工具挂载
)
```
