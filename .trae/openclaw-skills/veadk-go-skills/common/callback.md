# CallBack 定义方法

## 方法说明

### 1、BeforeModelCallBack
type BeforeModelCallback func(ctx agent.CallbackContext, llmRequest *model.LLMRequest) (*model.LLMResponse, error)

BeforeModelCallback that is called before sending a request to the model.
If it returns non-nil LLMResponse or error, the actual model call is skipped
and the returned response/error is used.

### 2、AfterModelCallback
type AfterModelCallback func(ctx agent.CallbackContext, llmResponse *model.LLMResponse, llmResponseError error) (*model.LLMResponse, error)

AfterModelCallback that is called after receiving a response from the model.
If it returns non-nil LLMResponse or error, the actual model response/error
is replaced with the returned response/error.

### 3、BeforeToolCallback
type BeforeToolCallback func(ctx tool.Context, tool tool.Tool, args map[string]any) (map[string]any, error)

BeforeToolCallback is executed before a tool's Run method.
Callbacks are executed in the order they are provided.
If a callback returns a non-nil result or an error:
- execution of remaining callbacks stops
- the actual tool call is skipped
- the returned result is used as the tool result

To modify tool arguments and still run the tool,
update args in place and return (nil, nil).

### 4、AfterToolCallback
type AfterToolCallback func(ctx tool.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error)
AfterToolCallback is a function type executed after a tool's Run method has completed, 
regardless of whether the tool returned a result or an error.

Callbacks are executed in the order they are provided.
If a callback returns a non-nil result or an error:
- execution of remaining callbacks stops
- the returned result and/or error is used as the final tool output


## callback方法示例

### 1、BeforeModelCallBack 代码示例
何时触发： 在LlmAgent流程中向 LLM 发送请求之前调用。
用途： 允许检查和修改发送给 LLM 的请求。用例包括添加动态指令、基于状态注入少量示例、修改模型配置、实现防护机制 (如亵渎过滤器) 或实现请求级缓存。
返回值效果： 如果回调返回 nil，LLM 继续其正常工作流程。如果回调返回 LlmResponse 对象，则跳过对 LLM 的调用。返回的 LlmResponse 直接使用，就像它来自模型一样。这对于实现防护栏或缓存非常强大。

```go
func onBeforeModel(ctx agent.CallbackContext, req *model.LLMRequest) (*model.LLMResponse, error) {
    log.Printf("[Callback] BeforeModel triggered for agent %q.", ctx.AgentName())

    // Modification Example: Add a prefix to the system instruction.
    if req.Config.SystemInstruction != nil {
        prefix := "[Modified by Callback] "
        // This is a simplified example; production code might need deeper checks.
        if len(req.Config.SystemInstruction.Parts) > 0 {
            req.Config.SystemInstruction.Parts[0].Text = prefix + req.Config.SystemInstruction.Parts[0].Text
        } else {
            req.Config.SystemInstruction.Parts = append(req.Config.SystemInstruction.Parts, &genai.Part{Text: prefix})
        }
        log.Printf("[Callback] Modified system instruction.")
    }

    // Skip Example: Check for "BLOCK" in the user's prompt.
    for _, content := range req.Contents {
        for _, part := range content.Parts {
            if strings.Contains(strings.ToUpper(part.Text), "BLOCK") {
                log.Println("[Callback] 'BLOCK' keyword found. Skipping LLM call.")
                return &model.LLMResponse{
                    Content: &genai.Content{
                        Parts: []*genai.Part{{Text: "LLM call was blocked by before_model_callback."}},
                        Role:  "model",
                    },
                }, nil
            }
        }
    }

    log.Println("[Callback] Proceeding with LLM call.")
    return nil, nil
}

rootAgent, err := veagent.New(&veagent.Config{
		Config: llmagent.Config{
			Name:        "...",
			Description: "...",
			Instruction: "...",
			BeforeModelCallbacks:[]llmagent.BeforeModelCallback{
				onBeforeModel,
			},
		},
	})
```

### 2、AfterModelCallBack 代码示例
何时触发： 在从 LLM 接收到响应 (LlmResponse) 之后，在调用智能体进一步处理之前调用。
用途： 允许检查或修改原始 LLM 响应。用例包括：
记录模型输出，
重新格式化响应，
审查模型生成的敏感信息，
从 LLM 响应中解析结构化数据并将其存储在callback_context.state中
或处理特定错误代码。

```go
func onAfterModel(ctx agent.CallbackContext, resp *model.LLMResponse, respErr error) (*model.LLMResponse, error) {
    log.Printf("[Callback] AfterModel triggered for agent %q.", ctx.AgentName())
    if respErr != nil {
        log.Printf("[Callback] Model returned an error: %v. Passing it through.", respErr)
        return nil, respErr
    }
    if resp == nil || resp.Content == nil || len(resp.Content.Parts) == 0 {
        log.Println("[Callback] Response is nil or has no parts, nothing to process.")
        return nil, nil
    }
    // Check for function calls and pass them through without modification.
    if resp.Content.Parts[0].FunctionCall != nil {
        log.Println("[Callback] Response is a function call. No modification.")
        return nil, nil
    }

    originalText := resp.Content.Parts[0].Text

    // Use a case-insensitive regex with word boundaries to find "joke".
    re := regexp.MustCompile(`(?i)\bjoke\b`)
    if !re.MatchString(originalText) {
        log.Println("[Callback] 'joke' not found. Passing original response through.")
        return nil, nil
    }

    log.Println("[Callback] 'joke' found. Modifying response.")
    // Use a replacer function to handle capitalization.
    modifiedText := re.ReplaceAllStringFunc(originalText, func(s string) string {
        if strings.ToUpper(s) == "JOKE" {
            if s == "Joke" {
                return "Funny story"
            }
            return "funny story"
        }
        return s // Should not be reached with this regex, but it's safe.
    })

    resp.Content.Parts[0].Text = modifiedText
    return resp, nil
}
```

### 3、BeforeToolCallback 代码示例

何时触发： 在调用特定工具的run_async方法之前，在 LLM 为其生成函数调用之后调用。

用途： 允许检查和修改工具参数，在执行前执行授权检查，记录工具使用尝试，或实现工具级缓存。

返回值效果：

如果回调返回 nil，工具方法将使用（可能修改的）args 执行。
如果返回map，工具方法将被跳过。返回的字典直接用作工具调用的结果。这对于缓存或覆盖工具行为很有用。

```go
func onBeforeTool(ctx tool.Context, t tool.Tool, args map[string]any) (map[string]any, error) {
    log.Printf("[Callback] BeforeTool triggered for tool %q in agent %q.", t.Name(), ctx.AgentName())
    log.Printf("[Callback] Original args: %v", args)

    if t.Name() == "getCapitalCity" {
        if country, ok := args["country"].(string); ok {
            if strings.ToLower(country) == "canada" {
                log.Println("[Callback] Detected 'Canada'. Modifying args to 'France'.")
                args["country"] = "France"
                return args, nil // Proceed with modified args
            } else if strings.ToUpper(country) == "BLOCK" {
                log.Println("[Callback] Detected 'BLOCK'. Skipping tool execution.")
                // Skip tool and return a custom result.
                return map[string]any{"result": "Tool execution was blocked by before_tool_callback."}, nil
            }
        }
    }
    log.Println("[Callback] Proceeding with original or previously modified args.")
    return nil, nil // Proceed with original args
}
```

### 4、AfterToolCallback 代码示例
何时触发： 在工具的执行方法成功完成后立即调用。
用途： 允许在将工具结果发送回 LLM(可能在摘要后) 之前对其进行检查和修改。适用于记录工具结果、后处理或格式化结果，或将结果的特定部分保存到会话状态。

返回值效果：
如果回调返回 nil，使用原始的 tool_response。
如果返回新map，它替换原始的 tool_response。这允许修改或过滤 LLM 看到的结果。

```go
func onAfterTool(ctx tool.Context, t tool.Tool, args map[string]any, result map[string]any, err error) (map[string]any, error) {
    log.Printf("[Callback] AfterTool triggered for tool %q in agent %q.", t.Name(), ctx.AgentName())
    log.Printf("[Callback] Original result: %v", result)

    if err != nil {
        log.Printf("[Callback] Tool run produced an error: %v. Passing through.", err)
        return nil, err
    }

    if t.Name() == "getCapitalCity" {
        if originalResult, ok := result["result"].(string); ok && originalResult == "Washington, D.C." {
            log.Println("[Callback] Detected 'Washington, D.C.'. Modifying tool response.")
            modifiedResult := make(map[string]any)
            for k, v := range result {
                modifiedResult[k] = v
            }
            modifiedResult["result"] = fmt.Sprintf("%s (Note: This is the capital of the USA).", originalResult)
            modifiedResult["note_added_by_callback"] = true
            return modifiedResult, nil
        }
    }

    log.Println("[Callback] Passing original tool response through.")
    return nil, nil
}
```



