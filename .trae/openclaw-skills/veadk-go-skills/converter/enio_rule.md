# Enio 与 VeADK-Go 对应规则

你可以通过下面的介绍，来了解 Enio 与 VeADK-Go 对应规则。具体的 VeADK-Go 定义方法可以参照 `references/samples/` 目录中的内容。

## Enio 常用类型

- ReAct Agent 和 ChatModel 节点：对应 VeADK-Go的 LLM Agent，请参照 `references/common/agent.md`
- RetrieverNode: 对应VeADK-Go的KnowledgeBase，请参照 `references/common/knowledgebase.md` 中的知识库定义和使用方法
- 工具节点/ToolsNode：对应VeADK-Go的工具，请参照 `references/common/tools.md`
- Chain和Graph的固定流程编排：直接用 Go 代码实现。
    - 其中不包含大模型的逻辑节点，按照该节点与模型调用以及工具调用的相对位置，封装于 callBack
      函数中，VeADK-Go的callBack函数，请参照 `references/common/callback.md`

## Enio 与 VeADK-Go 代码映射示例

### 1、Agent

Enio 代码实现

React Agent 代码实现
```go
func main() {
    // 先初始化所需的 chatModel
    toolableChatModel, err := openai.NewChatModel(...)
    
    // 初始化所需的 tools
    tools := compose.ToolsNodeConfig{
        InvokableTools:  []tool.InvokableTool{mytool},
        StreamableTools: []tool.StreamableTool{myStreamTool},
    }
    
    // 创建 agent
    agent, err := react.NewAgent(ctx, &react.AgentConfig{
        ToolCallingModel: toolableChatModel,
        ToolsConfig: tools,
        ...
    }
}
```

基于chain编排的Agent实现

```go
func main() {
    // 初始化 tools
    todoTools := []tool.BaseTool{
        getAddTodoTool(),                               // NewTool 构建
    }

    // 创建并配置 ChatModel
    chatModel, err := openai.NewChatModel(context.Background(), &openai.ChatModelConfig{
        Model:       "...",
        APIKey:      os.Getenv("OPENAI_API_KEY"),
    })
    if err != nil {
        log.Fatal(err)
    }
    // 获取工具信息并绑定到 ChatModel
    toolInfos := make([]*schema.ToolInfo, 0, len(todoTools))
    for _, tool := range todoTools {
        info, err := tool.Info(ctx)
        if err != nil {
            log.Fatal(err)
        }
        toolInfos = append(toolInfos, info)
    }
    err = chatModel.BindTools(toolInfos)
    if err != nil {
        log.Fatal(err)
    }


    // 创建 tools 节点
    todoToolsNode, err := compose.NewToolNode(context.Background(), &compose.ToolsNodeConfig{
        Tools: todoTools,
    })
    if err != nil {
        log.Fatal(err)
    }

    // 构建完整的处理链
    chain := compose.NewChain[[]*schema.Message, []*schema.Message]()
    chain.
        AppendChatModel(chatModel, compose.WithNodeName("chat_model")).
        AppendToolsNode(todoToolsNode, compose.WithNodeName("tools"))

    // 编译并生成 agent
    agent, err := chain.Compile(ctx)
    if err != nil {
        log.Fatal(err)
    }

}

```

VeADK-Go 代码实现

```go
func main() {
	ctx := context.Background()
	rootAgent, err := veagent.New(&veagent.Config{
		Config: llmagent.Config{
			Tools: []tool.Tool{utils.Must(AddTodoTool())},
		},
		ModelName:   "...",
		ModelAPIKey: os.Getenv("OPENAI_API_KEY"),
	})

	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}
}

```

### 2、Tool

Enio 代码实现
- 请注意：VeADK-Go的函数工具参数中，jsonschema标签下的说明，禁止包含'describr=' 或者任何 '***=' 的说明样式。

```go
// 处理函数
func AddTodoFunc(_ context.Context, params *TodoAddParams) (string, error) {
    // Mock处理逻辑
    return `{"msg": "add todo success"}`, nil
}

func getAddTodoTool() tool.InvokableTool {
    // 工具信息
    info := &schema.ToolInfo{
        Name: "add_todo",
        Desc: "Add a todo item",
        ParamsOneOf: schema.NewParamsOneOfByParams(map[string]*schema.ParameterInfo{
            "content": {
                Desc:     "The content of the todo item",
                Type:     schema.String,
                Required: true,
            },
            "started_at": {
                Desc: "The started time of the todo item, in unix timestamp",
                Type: schema.Integer,
            },
            "deadline": {
                Desc: "The deadline of the todo item, in unix timestamp",
                Type: schema.Integer,
            },
        }),
    }

    // 使用NewTool创建工具
    return utils.NewTool(info, AddTodoFunc)
}
```

VeADK-Go 代码实现

```go
// AddTodoParams 定义加法工具的入参。使用静态类型，便于 LLM 以 JSON 方式调用。
type AddTodoParams struct {
	Content   string `json:"content" jsonschema:"The content of the todo item"`
	StartedAt int64  `json:"started_at" jsonschema:"The started time of the todo item, in unix timestamp"`
	Deadline  int64  `json:"deadline" jsonschema:"The deadline of the todo item, in unix timestamp"`
}

// AddTodoTool 返回一个符合 ADK functiontool 规范的工具。
func AddTodoTool() (tool.Tool, error) {
	handler := func(ctx tool.Context, args AddTodoParams) (map[string]any, error) {
		return map[string]any{
			"msg": "add todo success",
		}, nil
	}

	return functiontool.New(
		functiontool.Config{
			Name:        "add_todo",
			Description: "Add a todo item",
		},
		handler,
	)
}
```
