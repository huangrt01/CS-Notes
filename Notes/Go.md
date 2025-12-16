# Go

[toc]

## 常用命令

### 模块管理 (Go Modules)

*   **`go mod init <module_path>`**
    *   **用途**：初始化一个新的模块（Module）。
    *   **说明**：会在当前目录下生成 `go.mod` 文件，定义模块路径。

*   **`go mod tidy`**
    *   **用途**：自动整理依赖关系（最常用）。
    *   **说明**：
        1.  **添加缺失依赖**：扫描代码中 import 的包，如果 `go.mod` 中缺少，会自动添加。
        2.  **移除未使用依赖**：如果 `go.mod` 中记录了某个包但代码中未引用，会自动移除。
        3.  **更新校验和**：同步更新 `go.sum` 文件。

*   **`go get`**
    *   **用途**：下载并修改 `go.mod` 中的依赖。
    *   **用法**：
        *   `go get example.com/pkg`：下载并添加/更新该包的最新版本。
        *   `go get example.com/pkg@v1.2.3`：下载指定版本。
    *   **注意**：在 Go 1.18+ 版本中，`go get` 主要专注于管理依赖配置，不再推荐用于安装可执行工具（安装工具请使用 `go install`）。

### 构建与运行

*   **`go run`**
    *   **用途**：编译并直接运行 Go 程序。
    *   **用法**：`go run main.go` 或 `go run .`
    *   **说明**：
        *   它会在临时目录中编译代码并立即执行，**不会**在当前目录下生成可执行文件。
        *   非常适合在开发调试阶段快速验证代码。

*   **`go build`**
    *   **用途**：编译包或源文件。
    *   **用法**：
        *   `go build`：在当前目录编译。
        *   `go build -o myapp`：编译并指定输出文件名为 `myapp`。
    *   **说明**：
        *   如果是 `main` 包，会生成可执行文件（Windows 下为 .exe，Linux/Mac 下为二进制文件）。
        *   如果是库包（非 main 包），只会进行编译检查，不会生成输出文件。

*   **`go install`**
    *   **用途**：编译并安装包（通常是可执行工具）。
    *   **用法**：
        *   `go install .`：安装当前模块（如果是 main 包）。
        *   `go install example.com/cmd/tool@latest`：安装远程工具的最新版本。
    *   **说明**：
        *   编译后的二进制文件会被放置在 `$GOPATH/bin` 或 `$GOBIN` 环境变量指定的目录下（从而可以直接在命令行运行）。

### 其他常用

*   **`go fmt ./...`**
    *   **用途**：格式化代码。
*   **`go vet ./...`**
    *   **用途**：静态代码检查，发现潜在错误。
*   **`go test ./...`**
    *   **用途**：运行测试。

## 基础语法 (对比 C++/Python)

### 结构体与初始化 (Struct & Literal)

Go 经常使用**匿名结构体**和**字面量初始化**来组织测试数据（Table-Driven Tests），这在 C++/Python 中较少见。

*   **匿名结构体切片**：
    ```go
    // 定义一个匿名结构体的切片 (slice of anonymous struct)
    tests := []struct {
        name        string
        input       int
        expected    int
    }{
        // 初始化切片元素
        {
            name:     "case 1",
            input:    10,
            expected: 20, // 注意：多行初始化时，最后一个元素必须有逗号
        },
        {
            name:     "case 2",
            input:    5,
            expected: 10,
        },
    }
    ```
    *   **对比 C++**：类似 `std::vector<struct { string name; int input; ... }>`，但 Go 的语法更直接，无需先定义 struct 类型。
    *   **对比 Python**：类似 `[{'name': 'case1', ...}, ...]` 字典列表，但 Go 是强类型的。

### 方法定义与多返回值 (Method Definition)

Go 的方法是作用在接收者（Receiver）上的函数，且支持返回多个值。

```go
// 示例：定义一个获取数据的方法
func (c Client) FetchData(
    ctx context.Context, 
    ids []string,
) (map[string]*Item, error)
```

*   **接收者 (Receiver)**：
    *   `func (c Client)`：位于 `func` 关键字和方法名之间。
    *   表示该函数是 `Client` 结构体的一个**方法**（类似 Class 的成员函数），可以通过 `instance.FetchData()` 调用。
    *   **值接收者 vs 指针接收者**：
        *   `(c Client)`：**值拷贝**。方法内修改 `c` 不影响原对象，且涉及拷贝开销。
        *   `(c *Client)`：**指针引用**。可修改原对象，无拷贝开销（推荐用于大对象或需修改状态的场景）。
*   **多返回值**：
    *   `(map[string]*Item, error)`：Go 函数支持返回多个值。
    *   惯例是最后一个返回值返回 `error` 类型（接口），如果操作成功则返回 `nil`。

### Map 的定义与初始化

*   **语法**：`map[KeyType]ValueType`
*   **初始化**：
    ```go
    // 嵌套初始化
    itemMap := map[string]*model.EmbeddingResult{
        "item1": {DenseEmbedding: []float64{1.0, 2.0}}, // 自动推导类型
        "item2": &model.EmbeddingResult{...},           // 显式写法
    }
    ```
    *   **注意**：Go 的字面量初始化支持**类型推导**，内部元素可以省略类型名（如直接写 `{...}` 而不是 `&model.EmbeddingResult{...}`）。

### 延迟执行 (defer)

*   **用途**：推迟函数的执行直到当前函数返回前（Return 之前）。
*   **特性**：
    1.  **LIFO (后进先出)**：如果有多个 `defer`，最后声明的先执行（栈顺序）。
    2.  **参数预计算**：`defer` 声明时，函数参数的值就已经被计算并固定了（除非参数是闭包引用）。
    3.  **支持 Panic**：即使发生 panic，`defer` 也会被执行（常用于 `recover`）。
*   **常见场景**：
    *   资源释放：`defer file.Close()`、`defer mutex.Unlock()`。
    *   并发控制：`defer wg.Done()`。
    *   错误处理：`defer func() { if err := recover(); err != nil { ... } }()`。
*   **对比 C++**：类似析构函数（RAII），但更灵活（针对函数作用域而非对象生命周期）。
*   **对比 Python**：类似 `try...finally` 或 `with` 语句（Context Manager）。

### 错误处理：Panic vs C++ Exception

Go 的 `panic`/`recover` 机制常被拿来与 C++ 的异常机制对比，但二者在设计哲学和使用场景上有本质区别。

#### 1. 设计哲学与使用场景
*   **C++ (Exception)**：
    *   **常规错误处理机制**：C++ 中异常被视为处理运行时错误的“标准”方式（尽管 Google Style Guide 等规范可能禁用）。
    *   **隐式传播**：异常会沿着调用栈自动向上传播，直到遇到 `catch`。
*   **Go (Panic)**：
    *   **Crash 机制**：Go 将错误分为 `error`（预期内的、可处理的错误）和 `panic`（不可恢复的程序错误，如空指针引用、数组越界）。
    *   **显式处理**：绝大多数业务逻辑应通过 `error` 返回值处理。`panic` 仅用于严重 Bug 或库的初始化失败。
    *   **局限性**：`panic` 只能在当前 Goroutine 中被 `recover`，**无法跨 Goroutine 捕获**（子协程 panic 会导致整个进程崩溃）。

#### 2. 性能开销对比
*   **Happy Path (无错误发生)**：
    *   **C++ (Zero-cost)**：在 Itanium ABI 下，进入 `try` 块几乎**零开销**（编译器生成静态跳转表，不执行指令）。
    *   **Go (defer)**：**低开销**。Go 1.14+ 引入了 Open-coded defer 优化，将 defer 调用内联到函数尾部，开销已大幅降低（~6ns），但在循环中使用 defer 仍有较高开销（涉及堆分配）。
*   **Sad Path (抛出异常/Panic)**：
    *   **C++**：**极高开销**。涉及查表（LSDA）、栈展开（Stack Unwinding）、RTTI 类型匹配，可能比正常返回慢 100 倍以上。
    *   **Go**：**高开销**。涉及 Runtime 停止正常流程、执行 defer 链、打印堆栈信息等。

#### 3. 开发实践建议
*   **C++**：
    *   如果在允许使用异常的项目中，利用 RAII（智能指针、`std::lock_guard`）保证资源释放（Stack Unwinding 时会自动调用析构函数）。
    *   如果在禁用异常的项目（如 Google），必须使用错误码（`absl::Status`）。
*   **Go**：
    *   **不要滥用 Panic 做控制流**。
    *   **必须捕获的情况**：
        1.  **防止服务崩溃**：在 HTTP Server 的中间件（Middleware）或 RPC 框架入口处使用 `recover`，防止单个请求的 panic 导致整个服务进程退出。
        2.  **跨语言调用**：在调用 CGO 代码或不可控的第三方库时防御性捕获。
    *   **资源管理**：不依赖析构函数，而是显式使用 `defer`（如 `defer f.Close()`）。

### 指针与取地址 (& 符号)

*   **语法**：`&StructName{...}`
*   **含义**：
    1.  `StructName{...}`：创建一个结构体**实例**（值类型）。
    2.  `&`：**取地址**操作符。
    3.  **合起来**：在堆上创建一个结构体实例，并返回指向它的**指针**。
    *   **对比 C++**：等价于 `new StructName(...)`。
    *   **对比 Python**：Python 所有对象默认都是引用（指针），所以没有显式的 `&`。Go 需要区分“值传递”和“指针传递”，如果希望传递对象引用（避免拷贝），必须用 `&`。

### 指针切片 (`[]*Type`)

*   **语法**：`[]*StructName`
*   **含义**：一个切片（Slice），其中的每个元素都是指向 `StructName` 的**指针**。
*   **示例**：
    ```go
    // 定义一个指针切片
    var items []*model.RecallItemInfo
    
    // 初始化
    items = []*model.RecallItemInfo{
        &model.RecallItemInfo{ItemID: "1", Score: 0.9}, // 必须取地址
        &model.RecallItemInfo{ItemID: "2", Score: 0.8},
    }
    ```
*   **为什么要用指针切片？**
    1.  **避免拷贝**：如果 Struct 很大，在切片扩容或传递时，复制指针比复制整个结构体开销小得多。
    2.  **修改原值**：可以通过切片中的指针直接修改原始对象。
    3.  **支持 nil**：指针可以是 `nil`，而结构体值类型不行。
*   **对比 C++**：等价于 `std::vector<StructName*>`。
*   **对比 Python**：Python 的 List `[obj1, obj2]` 本质上就是指针列表（引用列表）。

### 结构体标签与可选字段 (Struct Tags & Optional Fields)

在定义配置结构体时，使用指针类型来处理可选参数是一种常见模式。

```go
type RecallStrategyConfig struct {
    Strategy *string  `json:"strategy"` // 策略: "mean" or "rrf"
    Alpha    *float64 `json:"alpha"`    // 衰减因子 (Decay factor)
    RRFK     *int     `json:"rrf_k"`    // RRF 常数 K
}
```

*   **字段定义三要素**：
    1.  **字段名 (`Strategy`)**：必须**首字母大写**（Exported），否则 JSON 库无法通过反射访问该字段，解析时会忽略。
    2.  **类型 (`*string`)**：字段的数据类型。
    3.  **标签 (`json:"strategy"`)**：Struct Tag，指示 `encoding/json` 库将 JSON 对象中的键 `"strategy"` 映射到此字段。如果不写标签，默认匹配字段名（不区分大小写）。

*   **指针类型的意义**：
    *   **区分零值**：如果是值类型（如 `int`），当 JSON 中缺少该字段时会默认为 `0`，无法区分是“用户指定了 0”还是“用户未指定”。
    *   **三态逻辑**：使用指针（`*int`）可以表达：存在且有值、存在且为零值、不存在（`nil`）。

### JSON 反序列化与未知字段处理

Go 的 `encoding/json` 标准库在处理 JSON 反序列化时有一些特定的行为，特别是关于未知字段的处理。

*   **默认行为：自动忽略**
    *   **机制**：`json.Unmarshal` 默认会**自动忽略** JSON 中存在但结构体中未定义的字段。
    *   **优点**：天然支持向前兼容（Forward Compatibility），服务端增加新字段不会导致旧版客户端报错。
    *   **缺点**：可能掩盖字段拼写错误（例如将 `json:"userID"` 错写为 `json:"userid"`，导致解析不到值且无报错）。

*   **严格模式：`DisallowUnknownFields`**
    *   **用途**：强制要求 JSON 数据中的字段必须在结构体中有定义，否则报错。常用于对输入数据要求严格的 API 接口。
    *   **实现**：必须使用 `json.Decoder`（而非 `json.Unmarshal`）并调用 `DisallowUnknownFields()`。
    *   **示例**：
        ```go
        type Config struct {
            Host string `json:"host"`
            Port int    `json:"port"`
        }

        func parseConfigStrict(data []byte) (*Config, error) {
            var cfg Config
            // 1. 创建 Decoder
            decoder := json.NewDecoder(bytes.NewReader(data))
            
            // 2. 开启严格模式：发现未知字段立即报错
            decoder.DisallowUnknownFields()
            
            // 3. 解码
            if err := decoder.Decode(&cfg); err != nil {
                return nil, err // 如果包含未定义字段，会返回 error
            }
            return &cfg, nil
        }
        ```

*   **知识拓展：Tag 常用控制符**
    *   **`json:"-"`**：**完全忽略**。无论序列化还是反序列化，都忽略该字段。
    *   **`json:"name,omitempty"`**：**序列化时忽略零值**。
        *   如果字段值为零值（`0`, `""`, `false`, `nil` 等），生成的 JSON 中将不包含该 key。
        *   **注意**：`omitempty` 仅影响序列化（Marshal），不影响反序列化（Unmarshal）。
    *   **`json:",string"`**：**类型转换**。
        *   常用于解决精度丢失问题（如将 64 位整数作为字符串传输）：`Int64Val int64 json:",string"`。
        *   前端传 `"12345"` (string)，Go 解析为 `12345` (int64)。

## 并发编程 (Concurrency)

Go 的并发模型基于 Goroutine 和 Channel，其中 `sync.WaitGroup` 是等待一组 Goroutine 完成的标准方式。

### sync.WaitGroup 模式

用于等待一组并发任务全部完成。

```go
var wg sync.WaitGroup // 定义 WaitGroup

for _, item := range items {
    wg.Add(1) // 1. 启动前计数器 +1
    
    // 启动 Goroutine
    go func(val string) {
        defer wg.Done() // 3. 退出时计数器 -1 (使用 defer 保证一定执行)
        
        // 执行业务逻辑...
        // 这里的 val 是通过参数传递进来的，避免闭包捕获循环变量的问题
    }(item) 
}

wg.Wait() // 2. 阻塞直到计数器归零
close(resultsChan) // 所有任务完成后，安全关闭 Channel
```

*   **核心三部曲**：`Add(1)` -> `go func` -> `Wait()`。
*   **闭包陷阱 (Common Pitfall)**：
    *   **错误写法**：直接在 `go func` 中使用循环变量 `i` 或 `item`。因为 Goroutine 启动可能有延迟，运行时循环可能已经结束，导致所有 Goroutine 都读取到同一个（最后的）变量值。
    *   **正确写法**：将变量作为**参数传递**给匿名函数（如上例中的 `val`），或者在循环内部重新赋值（`v := item`）。
*   **defer wg.Done()**：务必使用 `defer`，确保即使函数发生 panic 或提前 return，计数器也能正确减少，防止死锁。

### sync.Map vs Map (Thread Safety)

*   **普通 Map (`map`)**
    *   **非线程安全**：在多个 Goroutine 中并发读写（或至少有一个写）同一个 map 会导致 **fatal error: concurrent map writes** (Panic 且无法 recover)，导致程序 crash。
    *   **解决方案**：使用 `sync.RWMutex` 或 `sync.Mutex` 进行加锁保护。

*   **sync.Map**
    *   **用途**：Go 1.9 引入的**线程安全**的 Map 实现。
    *   **特点**：
        1.  **无需加锁**：内部实现了复杂的锁机制和原子操作，开箱即用。
        2.  **类型擦除**：Key 和 Value 都是 `interface{}` 类型，使用时需要进行**类型断言**。
    *   **适用场景**：
        1.  **读多写少**：Key 集合稳定，写入很少，读取很多。
        2.  **Key 分离**：多个 Goroutine 读写 disjoint (不相交) 的 Key 集合。
    *   **性能权衡**：在普通场景下（如频繁读写不同 Key），性能可能不如 `Mutex + Map`。
    *   **常用方法**：
        *   `Store(key, value)`: 存储
        *   `Load(key)`: 读取
        *   `LoadOrStore(key, value)`: 读取或存储（原子操作）
        *   `Delete(key)`: 删除
        *   `Range(func(key, value interface{}) bool)`: 遍历

    *   **遍历示例 (Range)**：
        ```go
        // 假设 sourceMap 是一个 sync.Map (key: string, value: []*Item)
        if sourceMap != nil {
            // 1. 创建一个普通的 map 用于接收数据
            targetMap := make(map[string][]*Item)
        
            // 2. 使用 Range 遍历 sync.Map
            // Range 接收一个回调函数 func(key, value any) bool
            sourceMap.Range(func(key, value any) bool {
                // 3. 类型断言 (Type Assertion)
                // sync.Map 存储的是 interface{} (any)，取出后必须断言为具体类型
                k, ok1 := key.(string)
                v, ok2 := value.([]*Item)
        
                if ok1 && ok2 {
                    targetMap[k] = v // 4. 存入普通 map
                }
        
                return true // 返回 true 继续遍历，返回 false 停止遍历 (break)
            })
        
            // 5. 使用结果
            result.Data = targetMap
        }
        ```

### 接口与 Mock (Interface & Mock)

Go 的接口是隐式实现的（Duck Typing），测试时常用 Mock 对象。

*   **Mock 实现**：
    ```go
    type MockRepo struct {
        mock.Mock // 嵌入 mock.Mock (类似继承)
    }

    func (m *MockRepo) Fetch(id string) error {
        args := m.Called(id) // 记录调用参数
        return args.Error(0) // 返回预设的返回值
    }
    ```
*   **类型断言 (Type Assertion)**：
    ```go
    // args.Get(0) 返回的是 interface{} (类似 void* 或 PyObject)
    // 需要断言转回具体类型
    return args.Get(0).([]*model.Item) 
    ```
    *   **对比 C++**：类似 `dynamic_cast` 或 `static_cast`。
    *   **对比 Python**：Python 是动态类型，不需要断言，但 Go 必须显式转换。

### 常用测试模式 (Table-Driven Tests)

Go 社区非常推崇“表格驱动测试”：

```go
for _, tt := range tests {
    t.Run(tt.name, func(t *testing.T) {
        // 1. Setup Mock
        // 2. Execute
        // 3. Assert
    })
}
```
*   `t.Run`：启动子测试（Subtest），方便在报告中区分不同用例。
*   `:=` (短变量声明)：定义并初始化变量，自动推导类型。

## 环境配置

### GOPROXY

在国内环境下，通常需要配置 GOPROXY 以加速依赖下载。

```bash
go env -w GOPROXY=https://goproxy.cn,direct
```

*   `direct` 表示如果代理找不到，则直接回源下载。


