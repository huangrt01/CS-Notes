[toc]

### docker

```shell
docker ps -a

docker run -it -u `id -u`:`id -g` -v /home/ <img src="https://www.zhihu.com/equation?tex=%28whoami%29%3A/home/" alt="(whoami):/home/" class="ee_img tr_noresize" eeimg="1"> (whoami)
# -i 以交互模式运行容器，通常与 -t 同时使用
# -t 为容器重新分配一个伪输入终端，通常与 -i 同时使用
# -u 表示映射账户
# -v /宿主机目录:/容器目录   表示映射磁盘目录，映射的目录才会共享（将宿主机目录挂载到容器里），这里选择把user账户所有内容都映射

docker container ls -a
# exec进入容器
sudo docker exec -it [-w $(pwd)] 34d2b0644938 /bin/bash
# 如果容器已经停止，需要先启动再进入
sudo docker start 34d2b0644938
# Docker 里没有 sudo组，如果需要在 docker 里安装程序，可以先使用 root 账户进入容器
sudo docker exec -it [-w $(pwd)] 34d2b0644938 /bin/bash -u root 
```

添加 docker 权限给当前用户 ，使 docker 命令免 sudo

```shell
# 添加docker group
sudo groupadd docker
# 将当前用户添加到docker组
sudo gpasswd -a ${USER} docker
# 重启docker服务
sudo service docker restart
```

### log4j

```properties
# 业务日志
log4j.rootLogger=INFO,R

log4j.appender.stdout=org.apache.log4j.ConsoleAppender
log4j.appender.stdout.layout=org.apache.log4j.PatternLayout
log4j.appender.stdout.layout.ConversionPattern=%d %5p %m (%F:%L)%n

log4j.appender.R=org.apache.log4j.DailyRollingFileAppender
log4j.appender.R.File=/folder/service.log
log4j.appender.R.DatePattern=.yyyy-MM-dd_HH
log4j.appender.R.BufferedIO=false
log4j.appender.R.MaxBackupIndex=36
log4j.appender.R.Threshold=INFO
log4j.appender.R.layout=org.apache.log4j.PatternLayout
log4j.appender.R.layout.ConversionPattern=%d %5p %m%n

# 服务化平台日志
log4j.logger.RpcCall=TRACE,Call

log4j.appender.Call=org.apache.log4j.DailyRollingFileAppender
log4j.appender.Call.File=/folder/service.call.log
log4j.appender.Call.DatePattern=.yyyy-MM-dd_HH
log4j.appender.Call.BufferedIO=false
log4j.appender.Call.MaxBackupIndex=36
log4j.appender.Call.layout=org.apache.log4j.PatternLayout
log4j.appender.Call.layout.ConversionPattern=%p %d{yyyy-MM-dd HH:mm:ss} %F:%L %m%n

log4j.logger.RpcAccess=TRACE,Access

log4j.appender.Access=org.apache.log4j.DailyRollingFileAppender
log4j.appender.Access.File=/folder/service.access.log
log4j.appender.Access.DatePattern=.yyyy-MM-dd_HH
log4j.appender.Access.BufferedIO=false
log4j.appender.Access.MaxBackupIndex=36
log4j.appender.Access.layout=org.apache.log4j.PatternLayout
log4j.appender.Access.layout.ConversionPattern=%p %d{yyyy-MM-dd HH:mm:ss} %F:%L %m%n
```

### google::protobuf

[Language Guide](https://developers.google.com/protocol-buffers/docs/proto3)

[API Reference](https://developers.google.com/protocol-buffers/docs/reference/overview)

```shell
protoc -I= <img src="https://www.zhihu.com/equation?tex=SRC_DIR%20--python_out%3D" alt="SRC_DIR --python_out=" class="ee_img tr_noresize" eeimg="1"> DST_DIR $SRC_DIR/addressbook.proto
```

```protobuf
syntax = "proto3";

message SearchRequest {
  string query = 1;
  int32 page_number = 2;
  int32 result_per_page = 3;
  enum Corpus {
    UNIVERSAL = 0;
    WEB = 1;
    IMAGES = 2;
    LOCAL = 3;
    NEWS = 4;
    PRODUCTS = 5;
    VIDEO = 6;
  }
  Corpus corpus = 4;
}

message MyMessage1 {
  enum EnumAllowingAlias {
    option allow_alias = true;
    UNKNOWN = 0;
    STARTED = 1;
    RUNNING = 1;
  }
}

enum Foo {
  reserved 2, 15, 9 to 11, 40 to max;
  reserved "FOO", "BAR";
}
```

* scalar type有默认值
* 对于新增的optional type，需要注意加`[default = value]`或者用has方法判断是否存在
* allow_alias: 允许alias，属性的数字相同
* reserved values

Importing Definitions

* 允许import proto2，但不能直接在proto3用proto2 syntax
* import public允许pb的替换

```protobuf
// old.proto
// This is the proto that all clients are importing.
import public "new.proto";
import "other.proto";
```

Nested Types

```protobuf
message SearchResponse {
  message Result {
    string url = 1;
    string title = 2;
    repeated string snippets = 3;
  }
  repeated Result results = 1;
}

message SomeOtherMessage {
  SearchResponse.Result result = 1;
}
```

Any Type
```c++
import "google/protobuf/any.proto";

message ErrorStatus {
  string message = 1;
  repeated google.protobuf.Any details = 2;
}

// Storing an arbitrary message type in Any.
NetworkErrorDetails details = ...;
ErrorStatus status;
status.add_details()->PackFrom(details);
// Reading an arbitrary message from Any.
ErrorStatus status = ...;
for (const Any& detail : status.details()) {
  if (detail.Is<NetworkErrorDetails>()) {
    NetworkErrorDetails network_error;
    detail.UnpackTo(&network_error);
    ... processing network_error ...
  }
}
```

Oneof Type
* Changing a single value into a member of a new oneof is safe and binary compatible. Moving multiple fields into a new oneof may be safe if you are sure that no code sets more than one at a time. Moving any fields into an existing oneof is not safe.
* 小心oneof出core，设了另一个field会把原先的删掉，不能再设原先的内部field
* [Backwards-compatibility issues](https://developers.google.com/protocol-buffers/docs/proto3#backwards-compatibility_issues)



Map Type

* When parsing from the wire or when merging, if there are duplicate map keys the last key seen is used. When parsing a map from text format, parsing may fail if there are duplicate keys.
* Backwards compatibility
```protobuf
message MapFieldEntry {
  key_type key = 1;
  value_type value = 2;
}

repeated MapFieldEntry map_field = N;
```



Packages

* 相当于C++的namespace
* `foo.bar.Open`，从后往前搜索，先搜索bar再搜索foo，如果是`.foo.bar.Open`，则从前往后搜索
```protobuf
package foo.bar;
message Open { ... }

message Foo {
  ...
  foo.bar.Open open = 1;
  ...
}
```



**RPC Service**

* 定义

```protobuf
service SearchService {
  rpc Search(SearchRequest) returns (SearchResponse);
}
```

* client code
```c++
using google::protobuf;

protobuf::RpcChannel* channel;
protobuf::RpcController* controller;
SearchService* service;
SearchRequest request;
SearchResponse response;

void DoSearch() {
  // You provide classes MyRpcChannel and MyRpcController, which implement
  // the abstract interfaces protobuf::RpcChannel and protobuf::RpcController.
  channel = new MyRpcChannel("somehost.example.com:1234");
  controller = new MyRpcController;

  // The protocol compiler generates the SearchService class based on the
  // definition given above.
  service = new SearchService::Stub(channel);

  // Set up the request.
  request.set_query("protocol buffers");

  // Execute the RPC.
  service->Search(controller, request, response, protobuf::NewCallback(&Done));
}

void Done() {
  delete service;
  delete channel;
  delete controller;
}
```
* service code
```c++
using google::protobuf;

class ExampleSearchService : public SearchService {
 public:
  void Search(protobuf::RpcController* controller,
              const SearchRequest* request,
              SearchResponse* response,
              protobuf::Closure* done) {
    if (request->query() == "google") {
      response->add_result()->set_url("http://www.google.com");
    } else if (request->query() == "protocol buffers") {
      response->add_result()->set_url("http://protobuf.googlecode.com");
    }
    done->Run();
  }
};

int main() {
  // You provide class MyRpcServer.  It does not have to implement any
  // particular interface; this is just an example.
  MyRpcServer server;

  protobuf::Service* service = new ExampleSearchService;
  server.ExportOnPort(1234, service);
  server.Run();

  delete service;
  return 0;
}
```



Options

`google/protobuf/descriptor.proto`

```proto
option optimize_for = CODE_SIZE; //SPEED(DEFAULT), LITE_RUNTIME
option cc_enable_arenas = true;
int32 old_field = 6 [deprecated = true];
```



#### Python API

[Python API](https://googleapis.dev/python/protobuf/latest/), [Python Tutorial](https://developers.google.com/protocol-buffers/docs/pythontutorial), [Python Pb Guide](https://www.datascienceblog.net/post/programming/essential-protobuf-guide-python/)

```python
import addressbook_pb2
person = addressbook_pb2.Person()
person.id = 1234
person.name = "John Doe"
person.email = "jdoe@example.com"
phone = person.phones.add()
phone.number = "555-4321"
phone.type = addressbook_pb2.Person.HOME
```

* Wrapping protocol buffers is also a good idea if you don't have control over the design of the `.proto` file
* You should never add behaviour to the generated classes by inheriting from them

```python
# serialize proto object
import os
out_dir = "proto_dump"
with open(os.path.join(out_dir, "person.pb"), "wb") as f:
    # binary output
    f.write(person.SerializeToString())
with open(os.path.join(out_dir, "person.protobuf"), "w") as f:
    # human-readable output for debugging
    # by default, entries with a value of 0 are never printed
    f.write(str(person))
```



python动态解析oneof字段

```python
data = getattr(config, config.WhichOneof('config')).value
```

#### C++ 代码阅读

* google/protobuf/repeated_field.h

```c++
AddAllocatedInternal()
// arena 一样则 zero copy
// if current_size < allocated_size，Make space at [current] by moving first allocated element to end of allocated list.
    
// DeleteSubgrange，注意性能
template <typename Element>
inline void RepeatedPtrField<Element>::DeleteSubrange(int start, int num) {
  GOOGLE_DCHECK_GE(start, 0);
  GOOGLE_DCHECK_GE(num, 0);
  GOOGLE_DCHECK_LE(start + num, size());
  for (int i = 0; i < num; ++i) {
    RepeatedPtrFieldBase::Delete<TypeHandler>(start + i);
  }
  ExtractSubrange(start, num, NULL);
}

mutable_obj()->Add()->CopyFrom(*old_objptr);
mutable_obj()->AddAllocated(old_objptr);
```



#### tools

```python
import pathlib
import os
from subprocess import check_call

def generate_proto_code():
    proto_interface_dir = "./src/interfaces"
    generated_src_dir = "./src/generated/"
    out_folder = "src"
    if not os.path.exists(generated_src_dir):
        os.mkdir(generated_src_dir)
    proto_it = pathlib.Path().glob(proto_interface_dir + "/**/*")
    proto_path = "generated=" + proto_interface_dir
    protos = [str(proto) for proto in proto_it if proto.is_file()]
    check_call(["protoc"] + protos + ["--python_out", out_folder, "--proto_path", proto_path])
    
from setuptools.command.develop import develop
from setuptools import setup, find_packages

class CustomDevelopCommand(develop):
    """Wrapper for custom commands to run before package installation."""
    uninstall = False

    def run(self):
        develop.run(self)

    def install_for_development(self):
        develop.install_for_development(self)
        generate_proto_code()

setup(
    name='testpkg',
    version='1.0.0',
    package_dir={'': 'src'},
    cmdclass={
        'develop': CustomDevelopCommand, # used for pip install -e ./
    },
    packages=find_packages(where='src')
)
```

### grafana

* 如何在同一个panel中使用不同的纵轴
  * 设置 Left Y 和 Right Y
    *  `Percent (0.0-1.0)`
    * `time: YYYY-MM-DD HH`
  * 设置 Series overrides
    * Alias or regex: `/MyMetric.*/`
      * `Y-axis: 2`

### gtest

https://google.github.io/googletest/primer.html

* 基础概念：
  * An assertion’s result can be *success*, *nonfatal failure* (EXPECT), or *fatal failure* (ASSERT).
  * test suite, test fixture class
    *  test命名不能带下划线，因为Gtest源码使用下划线将他们拼接为独立的类名

```c++
ASSERT_EQ(x.size(), y.size()) << "Vectors x and y are of unequal length";

for (int i = 0; i < x.size(); ++i) {
  EXPECT_EQ(x[i], y[i]) << "Vectors x and y differ at index " << i;
}
```

```c++
// Tests factorial of 0.
TEST(FactorialTest, HandlesZeroInput) {
  EXPECT_EQ(Factorial(0), 1);
}

// Tests factorial of positive numbers.
TEST(FactorialTest, HandlesPositiveInput) {
  EXPECT_EQ(Factorial(1), 1);
  EXPECT_EQ(Factorial(2), 2);
  EXPECT_EQ(Factorial(3), 6);
  EXPECT_EQ(Factorial(8), 40320);
}

-- InGtest(googletest/include/gtest/internal/gtest-internal.h) --
// Expands to the name of the class that implements the given test.
#define GTEST_TEST_CLASS_NAME_(test_suite_name, test_name) \                                                                                                      
  test_suite_name##_##test_name##_Test
```

```c++
class TestFixtureName : public ::testing::Test {
 protected:
	...
  void SetUp();
  void TearDown();
};

TEST_F(TestFixtureName, TestName) {
  ... test body ...
}
```

* TEST_F
  * different tests in the same test suite have different test fixture objects
  * 是否使用 SetUp/TearDown，参考[FAQ](https://google.github.io/googletest/faq.html#CtorVsSetUp)
  * TestFixtureName 常命名为 XxxTest

```c++
#include "gtest/gtest.h"

template <typename E>  // E is the element type.
class Queue {
 public:
  Queue();
  void Enqueue(const E& element);
  E* Dequeue();  // Returns NULL if the queue is empty.
  size_t size() const;
  ...
};

class QueueTest : public ::testing::Test {
 protected:
  void SetUp() override {
     q1_.Enqueue(1);
     q2_.Enqueue(2);
     q2_.Enqueue(3);
  }

  // void TearDown() override {}

  Queue<int> q0_;
  Queue<int> q1_;
  Queue<int> q2_;
};

TEST_F(QueueTest, IsEmptyInitially) {
  EXPECT_EQ(q0_.size(), 0);
}

TEST_F(QueueTest, DequeueWorks) {
  int* n = q0_.Dequeue();
  EXPECT_EQ(n, nullptr);

  n = q1_.Dequeue();
  ASSERT_NE(n, nullptr);
  EXPECT_EQ(*n, 1);
  EXPECT_EQ(q1_.size(), 0);
  delete n;

  n = q2_.Dequeue();
  ASSERT_NE(n, nullptr);
  EXPECT_EQ(*n, 2);
  EXPECT_EQ(q2_.size(), 1);
  delete n;
}

int main(int argc, char **argv) {
  signal(SIGPIPE, SIG_IGN);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
```

* When invoked, the `RUN_ALL_TESTS()` macro:
  * Saves the state of all googletest flags.
  * Creates a test fixture object for the first test.
  * Initializes it via `SetUp()`.
  * Runs the test on the fixture object.
  * Cleans up the fixture via `TearDown()`.
  * Deletes the fixture.
  * Restores the state of all googletest flags.
  * Repeats the above steps for the next test, until all tests have run.



https://google.github.io/googletest/advanced.html

* More Assertions
  * Explicit Success and Failure
    * `SUCCEED()`、`FAIL()`、`ADD_FAILURE`、`ADD_FAILURE_AT`
  * Exception Assertions
    * EXPECT_NO_THROW，{NO}{}{ANY}
  * Predicate Assertions for Better Error Messages
  * [`EXPECT_PRED_FORMAT*`](https://google.github.io/googletest/reference/assertions.html#EXPECT_PRED_FORMAT)
  * [Floating-Point Comparison](https://google.github.io/googletest/reference/assertions.html#floating-point)
  * Asserting Using Gmock Matchers
    * https://google.github.io/googletest/reference/assertions.html#EXPECT_THAT
  * More String Assertions
    * https://google.github.io/googletest/reference/matchers.html#string-matchers
  * Type Assertions
  * Assertion Placement
    * The one constraint is that assertions that generate a fatal failure (`FAIL*` and `ASSERT_*`) can only be used in void-returning functions.
    * 同样也不能用在构造/析构函数里

```c++
EXPECT_NO_THROW({
  int n = 5;
  DoSomething(&n);
});
```

```c++
namespace testing {

// Returns an AssertionResult object to indicate that an assertion has
// succeeded.
AssertionResult AssertionSuccess();

// Returns an AssertionResult object to indicate that an assertion has
// failed.
AssertionResult AssertionFailure();

}

testing::AssertionResult IsEven(int n) {
  if ((n % 2) == 0)
    return testing::AssertionSuccess();
  else
    return testing::AssertionFailure() << n << " is odd";
}

EXPECT_FALSE(IsEven(Fib(6))
```

```c++
// Type Assertions
::testing::StaticAssertTypeEq<T1, T2>();

template <typename T> class Foo {
 public:
  void Bar() { testing::StaticAssertTypeEq<int, T>(); }
};
void Test2() { Foo<bool> foo; foo.Bar(); }
```

* Skipping test execution

```c++
TEST(SkipTest, DoesSkip) {
  GTEST_SKIP() << "Skipping single test";
  EXPECT_EQ(0, 1);  // Won't fail; it won't be executed
}

class SkipFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    GTEST_SKIP() << "Skipping all tests for this fixture";
  }
};

// Tests for SkipFixture won't be executed.
TEST_F(SkipFixture, SkipsOneTest) {
  EXPECT_EQ(5, 7);  // Won't fail
}
```

* Running Test Programs: Advanced Options

  * --gtests_list_tests
  * `./foo_test --gtest_filter="FooTest.*:BarTest.*-FooTest.Bar:BarTest.Foo"`
    * 注意引号
  * `TEST_FAIL_FAST` environment variable or `--gtest_fail_fast` flag
  * To include disabled tests in test execution, just invoke the test program with the `--gtest_also_run_disabled_tests` flag or set the `GTEST_ALSO_RUN_DISABLED_TESTS` environment variable to a value other than `0`. You can combine this with the `--gtest_filter` flag to further select which disabled tests to run.
  * Controlling How Failures Are Reported
    * TEST_PREMATURE_EXIT_FILE
    * **GTEST_BREAK_ON_FAILURE** 好用，适合debugger
    * Disabling Catching Test-Thrown Exceptions
    * Sanitizer support

```c++
// Tests that Foo does Abc.
TEST(FooTest, DISABLED_DoesAbc) { ... }

class DISABLED_BarTest : public testing::Test { ... };

// Tests that Bar does Xyz.
TEST_F(DISABLED_BarTest, DoesXyz) { ... }
```

```shell
$ foo_test --gtest_repeat=1000
Repeat foo_test 1000 times and don't stop at failures.

$ foo_test --gtest_repeat=-1
A negative count means repeating forever.

$ foo_test --gtest_repeat=1000 --gtest_break_on_failure
Repeat foo_test 1000 times, stopping at the first failure.  This
is especially useful when running under a debugger: when the test
fails, it will drop into the debugger and you can then inspect
variables and stacks.

$ foo_test --gtest_repeat=1000 --gtest_filter=FooBar.*
Repeat the tests whose name matches the filter 1000 times.

--gtest_shuffle

GTEST_TOTAL_SHARDS、GTEST_SHARD_INDEX
```

```c++
extern "C" {
void __ubsan_on_report() {
  FAIL() << "Encountered an undefined behavior sanitizer error";
}
void __asan_on_error() {
  FAIL() << "Encountered an address sanitizer error";
}
void __tsan_on_report() {
  FAIL() << "Encountered a thread sanitizer error";
}
}  // extern "C"
```



* DCHECK
  * [DCHECK只在debug模式生效，用于先验知道生效的CHECK](https://groups.google.com/a/chromium.org/g/chromium-dev/c/LU6NWiaSSRc)

### Hadoop

Hadoop Shell 命令：https://hadoop.apache.org/docs/r1.0.4/cn/hdfs_shell.html

```shell
hadoop fs -test -e filename
```

### thrift

```shell
brew install thirft@0.9
thrift -version

If you need to have thrift@0.9 first in your PATH, run:
  echo 'export PATH="/usr/local/opt/thrift@0.9/bin:$PATH"' >> ~/.zshrc

For compilers to find thrift@0.9 you may need to set:
  export LDFLAGS="-L/usr/local/opt/thrift@0.9/lib"
  export CPPFLAGS="-I/usr/local/opt/thrift@0.9/include"

For pkg-config to find thrift@0.9 you may need to set:
  export PKG_CONFIG_PATH="/usr/local/opt/thrift@0.9/lib/pkgconfig"
```



```python
# easy thrift client
def _retry(self, func, req, times=5):
  for _ in range(times):
    try:
      resp = getattr(self._client, func)(req)
      if resp.BaseResp.StatusCode == 0:
        return resp
      err = resp.BaseResp.StatusMessage
    except Exception as e:
      err = repr(e)
    logging.warning('%s errored with %s', func, err)
    time.sleep(random.random() * 3)
    self.reset_client()
  raise RuntimeError(err)
```

```c++
if (!info.fromJsonString(str)) {
  if (!info.fromBinaryString(str)) {
    ...
  }
}
```



### YAML

* 基础语法

```yaml
# - 表示数组
series:
	- target:
		actions:
  - target:
  	actions:
```



* multiple documents in the same stream
  * https://stackoverflow.com/questions/50788277/why-3-dashes-hyphen-in-yaml-file

```yaml
doc 1
...
%TAG !bar! !bar-types/
---
doc 2
```

```python
# conf读取
import yaml
confs = list(yaml.safe_load_all(f))
custom_conf = confs[0]

# merge conf
def _merge_conf(custom_conf: dict, default_conf: dict):
    new_conf = copy.deepcopy(default_conf)
    for k, custom_v in custom_conf.items():
        if isinstance(custom_v, dict):
            new_conf[k] = _merge_conf(custom_v, default_conf.get(k, {}))
        elif k not in new_conf or new_conf[k] is None or isinstance(custom_v, list):
            new_conf[k] = custom_v
        elif isinstance(new_conf[k], bool):
            new_conf[k] = custom_v in (True, 'True', 'true', 'TRUE', 't', '1')
        else:
            new_conf[k] = type(new_conf[k])(custom_v)
    return new_conf
```



