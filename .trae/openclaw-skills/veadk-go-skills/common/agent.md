# Agent 定义方法

## 导入方法

- LLM Agent: `import veagent "github.com/volcengine/veadk-go/agent/llmagent"`
- Sequential Agent: `import "github.com/volcengine/veadk-go/agent/workflowagents/sequentialagent"`
- Loop Agent: `import "github.com/volcengine/veadk-go/agent/workflowagents/loopagent"`
- Parallel Agent: `import "github.com/volcengine/veadk-go/agent/workflowagents/parallelagent"`

其中，LLM Agent 是最基础的智能体（由 LLM 启动进行自主决策），Sequential Agent 是按顺序执行的智能体，Loop Agent 是循环执行的智能体，Parallel Agent 是并行执行的智能体。

## 代码规范

### 1、你可以通过如下方式定义智能体：

```go
import (
	"context"
	"fmt"

	veagent "github.com/volcengine/veadk-go/agent/llmagent"
	"github.com/volcengine/veadk-go/apps"
	"github.com/volcengine/veadk-go/apps/agentkit_server_app"
	vetool "github.com/volcengine/veadk-go/tool"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/tool"
)

func main() {
	ctx := context.Background()

	subAgent, err := veagent.New(&veagent.Config{
		Config: llmagent.Config{
			Name:        "...",
			Description: "...",
			Instruction: `...`,
		},
		ModelName: "...",
	})
	if err != nil {
		fmt.Printf("NewLLMAgent subAgent failed: %v", err)
		return
	}

	rootAgent, err := veagent.New(&veagent.Config{
		Config: llmagent.Config{
			Name:        "...",
			Description: "...",
			Instruction: `...`,
			SubAgents: []agent.Agent{subAgent},
		},
		ModelName: "...",
	})
	if err != nil {
		fmt.Printf("NewLLMAgent rootAgent failed: %v", err)
		return
	}

	app := agentkit_server_app.NewAgentkitServerApp(apps.DefaultApiConfig())

	err = app.Run(ctx, &apps.RunConfig{
		AgentLoader: agent.NewSingleLoader(rootAgent),
	})
	if err != nil {
		fmt.Printf("Run failed: %v", err)
	}
}

```

### 2、可以生成一个强制按顺序执行的智能体：

```go
import (
	"context"
	"fmt"

	veagent "github.com/volcengine/veadk-go/agent/llmagent"
	"github.com/volcengine/veadk-go/agent/workflowagents/sequentialagent"
	"github.com/volcengine/veadk-go/apps"
	"github.com/volcengine/veadk-go/apps/agentkit_server_app"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
)

func main() {
	ctx := context.Background()

	agent1, err := veagent.New(&veagent.Config{
		Config: llmagent.Config{
			Name:        "...",
			Description: "...",
			Instruction: "...",
		},
	})
	if err != nil {
		fmt.Printf("NewLLMAgent agent1 failed: %v", err)
		return
	}

	agent2, err := veagent.New(&veagent.Config{
		Config: llmagent.Config{
			Name:        "...",
			Description: "...",
			Instruction: "...",
		},
	})
	if err != nil {
		fmt.Printf("NewLLMAgent agent failed: %v", err)
		return
	}

	rootAgent, err := sequentialagent.New(sequentialagent.Config{
		AgentConfig: agent.Config{
			Name:        "...",
			SubAgents:   []agent.Agent{agent1, agent2},
			Description: "...",
		},
	})

	if err != nil {
		fmt.Printf("NewSequentialAgent failed: %v", err)
		return
	}

	app := agentkit_server_app.NewAgentkitServerApp(apps.DefaultApiConfig())

	err = app.Run(ctx, &apps.RunConfig{
		AgentLoader: agent.NewSingleLoader(rootAgent),
	})
	if err != nil {
		fmt.Printf("Run failed: %v", err)
	}
}
```

`agent1` 与 `agent2` 将会严格按顺序执行

注意，根智能体的命名必须为 `rootAgent`。

## 让 Agent 结构化输出

为保证更高的准确率和 Agent 执行时的可控性，使用结构化输出是一种有效的手段。

在定义 Agent 时，通过 `model_extra_config={"response_format": ...}` 可以让 Agent 结构化输出。其中，`...` 是你定义的 Pydantic 模型，用于描述 Agent 的输出格式。

```python
from pydantic import BaseModel
from veadk import Agent, Runner


# 定义分步解析模型（对应业务场景的结构化响应）
class Step(BaseModel):
    explanation: str  # 步骤说明
    output: str  # 步骤计算结果


# 定义最终响应模型（包含分步过程和最终答案）
class MathResponse(BaseModel):
    steps: list[Step]  # 解题步骤列表
    final_answer: str  # 最终答案


agent = Agent(
    instruction="你是一位数学辅导老师，需详细展示解题步骤",
    model_extra_config={"response_format": MathResponse},
)
```
