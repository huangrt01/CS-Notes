# Tools 定义方法

## 导入方法

- 网络搜索：`from veadk.tools.builtin_tools.web_search import web_search`
- 链接读取：`from veadk.tools.builtin_tools.link_reader import link_reader`
- 图像生成：`from veadk.tools.builtin_tools.image_generate import image_generate`
- 视频生成：`from veadk.tools.builtin_tools.video_generate import video_generate`
- 代码沙箱执行（用来执行 Python 代码）：`from veadk.tools.builtin_tools.run_code import run_code`

## 自定义 Tool

你可以通过撰写一个 Python 函数来定义一个自定义 Tool（你必须清晰地定义好 Docstring）：

```python
def add(a: int, b: int) -> int:
    """Add two integers together.
    
    Args:
        a (int): The first integer.
        b (int): The second integer.
    
    Returns:
        int: The sum of a and b.
    """
    return a + b


agent = Agent(tools=[add])
```

如果你使用自定义 Tool，请遵循以下规范：

推荐：（1）返回 dict，字段名稳定且语义清晰；（2）对外部错误用“可解释的错误字符串”或 {"error": "...", "details": ...}，避免直接抛异常导致整轮失败

不推荐：（1）返回复杂对象实例（模型侧不可读）；（2）返回超大文本（建议先做裁剪/分页/只返回必要字段）

为了让模型更愿意正确用工具，挂载与触发建议：（1）instruction 里明确“什么时候必须用工具”；（2）工具函数参数名要贴近业务语义（例如 order_id、city）；（3）返回里提供模型可直接引用的字段（例如 result、items、summary）

## 代码规范

你可以通过如下方式将某个工具挂载到智能体上，例如 `web_search` 网络搜索工具：

```python
from veadk.tools.builtin_tools.web_search import web_search

root_agent = Agent(
    name="...",
    description="...",
    instruction="...", # 智能体系统提示词
    tools=[web_search] # 挂载工具列表
)
```
