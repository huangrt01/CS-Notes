[toc]

### ansible

```
ansible $host -u tiger -f 100 -T 60 -mshell -a 'bash -x /path/test.sh' -e "ansible_shell_executable=/bin/bash""
```

* Ansible 是一种自动化工具，它基于 SSH（Secure Shell）协议来实现系统配置和管理。下面是关于 Ansible 基于 SSH 协议的解释和理解：
  * 远程执行：Ansible 使用 SSH 协议与远程主机进行通信，通过在远程主机上执行命令和任务来实现自动化配置和管理。SSH 是一种安全的远程登录协议，它提供了加密的通信通道，确保数据的机密性和完整性。
  * 无需代理：与其他自动化工具不同，Ansible 不需要在远程主机上安装和配置任何额外的代理程序或客户端。它利用 SSH 协议本身的功能，直接与目标主机进行通信，并执行所需的操作。这简化了部署和管理过程，减少了对目标主机的依赖性和配置要求。
  * 无需持久连接：Ansible 采用基于 SSH 的无需持久连接的模型。这意味着 Ansible 在执行任务时，会建立一个临时的 SSH 连接到远程主机上，执行任务后立即断开连接。这种模型避免了长时间保持连接的开销和管理复杂性，并提供了更好的灵活性和可伸缩性。
  * 身份验证和授权：SSH 协议提供了多种身份验证方法，包括密码、公钥和证书等。Ansible 可以利用这些身份验证方法来连接和认证远程主机。此外，Ansible 还支持使用 SSH Agent 来管理和传递身份验证凭据，以提供更便捷和安全的身份验证方式。
  * 并行执行：Ansible 基于 SSH 协议实现了高度并行的任务执行。它可以同时与多个远程主机建立 SSH 连接，并在这些主机上并行执行任务，从而提高执行效率和速度。

* 总结起来，Ansible 基于 SSH 协议使得远程主机之间的自动化配置和管理变得简单、安全和高效。它利用 SSH 的加密通信和身份验证机制，无需额外的代理程序，实现了远程主机的批量操作和任务执行。

* jinja2(j2)语法
  * https://ansible.leops.cn/basic/Jinja2/
* Issue
  * “In **non** docker environment, these "left behind" processes are picked up by init process, which is capable of reaping zombies. In docker setup the solution is to add `init: true` into docker-compose, to enable init process inside docker.”
    * https://github.com/ansible/ansible/issues/49270
  


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

### protobuf

见【code-reading笔记】

### grafana

* 如何在同一个panel中使用不同的纵轴
  * 设置 Left Y 和 Right Y
    *  `Percent (0.0-1.0)`
    * `time: YYYY-MM-DD HH`
  * 设置 Series overrides
    * Alias or regex: `/MyMetric.*/`
      * `Y-axis: 2`

### glog

https://github.com/google/glog



### grpc

* debug

```python
os.environ['GRPC_VERBOSITY'] = 'DEBUG'
os.environ['GRPC_TRACE'] = 'all'
```

* 坑
  * failed to pick subchannel
    * https://github.com/grpc/grpc/issues/23340
    * https://stackoverflow.com/questions/69444526/python-grpc-failed-to-pick-subchannel

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

#### 使用技巧

* 用std::cerr输出信息

* DCHECK
  * [DCHECK只在debug模式生效，用于先验知道生效的CHECK](https://groups.google.com/a/chromium.org/g/chromium-dev/c/LU6NWiaSSRc)
  * 配合bazel的 `-c dbg`

* capture stdout

```cpp
testing::internal::CaptureStdout();
std::cout << "My test";
std::string output = testing::internal::GetCapturedStdout();
```

### Hadoop

Hadoop Shell 命令：https://hadoop.apache.org/docs/r1.0.4/cn/hdfs_shell.html

```shell
hadoop fs -test -e filename

hadoop fs -ls -h
hadoop fs -ls hdfs://abc/def/*/*/models/*/model.pb # 可匹配

hadoop fs -du -h
hadoop fs -get ... .
-mkdir
-cp

hadoop fs -copyToLocal <HDFS文件路径> <本地文件路径>
```

* 设置 JVM 的最大堆内存限制

```
LIBHDFS_OPTS = [
        os.environ.get('HADOOP_ROOT_LOGGER',
                       '-Dhadoop.root.logger=WARN,console'), "-Xmx4096m"
    ]
os.environ["LIBHDFS_OPTS"] = ' '.join(LIBHDFS_OPTS)
```

### Redis

* [redis.clients.jedis.exceptions.JedisClusterException: CLUSTERDOWN The cluster is down](https://github.com/redis/jedis/issues/2144#top)

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



