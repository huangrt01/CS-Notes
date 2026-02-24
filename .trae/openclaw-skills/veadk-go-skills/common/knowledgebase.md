# 知识库

本文档介绍如何在 VeADK-Go 中使用知识库。

## 导入

```go
import (
	"context"
	"fmt"
	"log"

	veagent "github.com/volcengine/veadk-go/agent/llmagent"
	"github.com/volcengine/veadk-go/apps"
	"github.com/volcengine/veadk-go/apps/agentkit_server_app"
	"github.com/volcengine/veadk-go/integrations/ve_tos"
	"github.com/volcengine/veadk-go/knowledgebase"
	"github.com/volcengine/veadk-go/knowledgebase/backend/viking_knowledge_backend"
	"github.com/volcengine/veadk-go/knowledgebase/ktypes"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/session"
)
```

## 定义

通过 `KnowledgeBase` 类可以定义一个知识库，并挂载到智能体上。

```go
func main() {
	ctx := context.Background()
	knowledgeBase, err := knowledgebase.NewKnowledgeBase(
		ktypes.VikingBackend,
		knowledgebase.WithBackendConfig(
			&viking_knowledge_backend.Config{
				Index:            "...",
				CreateIfNotExist: true, // 当 Index 不存在时会自动创建
				TosConfig: &ve_tos.Config{
					Bucket: "...",
				},
			}),
	)
	if err != nil {
		log.Fatal("NewVikingKnowledgeBackend error: ", err)
	}

	veAgent, err := veagent.New(&veagent.Config{
		Config: llmagent.Config{
			Name:        "...",
			Description: "...",
			Instruction: `...`,
		},
		ModelName: "...",
		KnowledgeBase: knowledgeBase,
	})
	if err != nil {
		fmt.Printf("NewLLMAgent failed: %v", err)
		return
	}
}
```
