# Tools 定义方法

## 自定义 Tool

你可以通过撰写一个 Go 函数来定义一个自定义 Tool（你必须清晰的定义好 Docstring）：

```go
import (
	"context"
	"fmt"
	"log"

	veagent "github.com/volcengine/veadk-go/agent/llmagent"
	"github.com/volcengine/veadk-go/apps"
	"github.com/volcengine/veadk-go/apps/agentkit_server_app"
	"github.com/volcengine/veadk-go/utils"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

// CalculatorAddArgs 定义加法工具的入参。使用静态类型，便于 LLM 以 JSON 方式调用。
type CalculatorAddArgs struct {
	A float64 `json:"a" jsonschema:"第一个加数，支持整数或小数"`
	B float64 `json:"b" jsonschema:"第二个加数，支持整数或小数"`
}

// CalculatorAddTool 返回一个符合 ADK functiontool 规范的工具。
// 该工具用于执行两数相加，并返回 result 字段。
func CalculatorAddTool() (tool.Tool, error) {
	handler := func(ctx tool.Context, args CalculatorAddArgs) (map[string]any, error) {
		result := args.A + args.B
		return map[string]any{
			"result":  result,
			"explain": fmt.Sprintf("%g + %g = %g", args.A, args.B, result),
		}, nil
	}

	return functiontool.New(
		functiontool.Config{
			Name:        "calculator_add",
			Description: "一个简单的计算器工具，执行两数相加。参数: a, b; 返回: result(浮点数)",
		},
		handler,
	)
}
func main() {
	ctx := context.Background()
	rootAgent, err := veagent.New(&veagent.Config{
		Config: llmagent.Config{
			Tools: []tool.Tool{utils.Must(CalculatorAddTool())},
		},
	})

	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
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

## 使用内置工具

你可以通过如下方式将某个工具挂载到智能体上，例如 `web_search` 网络搜索工具：

```go
import (
	"context"
	"fmt"
	"log"
	"os"

	veagent "github.com/volcengine/veadk-go/agent/llmagent"
	"github.com/volcengine/veadk-go/common"
	"github.com/volcengine/veadk-go/tool/builtin_tools/web_search"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
)

func main() {
	ctx := context.Background()
	cfg := veagent.Config{
		ModelName:    "...",
		ModelAPIBase: "...",
		ModelAPIKey:  "...",
	}

	webSearch, err := web_search.NewWebSearchTool(&web_search.Config{})
	if err != nil {
		fmt.Printf("NewWebSearchTool failed: %v", err)
		return
	}

	cfg.Tools = []tool.Tool{webSearch}

	a, err := veagent.New(&cfg)
	if err != nil {
		fmt.Printf("NewLLMAgent failed: %v", err)
		return
	}

	config := &launcher.Config{
		AgentLoader:    agent.NewSingleLoader(a),
		SessionService: session.InMemoryService(),
	}

	l := full.NewLauncher()
	if err = l.Execute(ctx, config, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}

```
