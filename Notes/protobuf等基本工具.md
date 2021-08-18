[toc]

### docker

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
protoc -I=$SRC_DIR --python_out=$DST_DIR $SRC_DIR/addressbook.proto
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

### Hadoop

Hadoop Shell 命令：https://hadoop.apache.org/docs/r1.0.4/cn/hdfs_shell.html

