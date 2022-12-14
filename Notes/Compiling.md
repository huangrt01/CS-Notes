[toc]

https://godbolt.org

### Theory

* [Shared Library Symbol Conflicts (on Linux)](https://holtstrom.com/michael/blog/post/437/Shared-Library-Symbol-Conflicts-(on-Linux).html)
  * 从左往右查找：Note that the linker only looks further down the line when looking for symbols used by but not defined in the current lib.
  
* [Linux 下 C++so 热更新](https://zhuanlan.zhihu.com/p/162366167)

* ABI (Application Binary Interface)
  * 应用程序的二进制接口，对于一个二进制的动态库或者静态库而言，可以详细描述在其中的函数的调用方式，定义在其中的数据类型的大小，数据结构的内存布局方式等信息
  * ABI 信息 对不同操作系统、不同编译链版本、不同二进制库对应源码版本 有或大或小的差异，从而造成预编译二进制库的兼容性问题，导致 compile error 或 执行时coredump
  
* 编译器有能力让不同 target 的 cpp 文件的不同编译选项，有区分地生效。但无法控制其它cpp文件对头文件的使用，因此头文件为主体的开源项目，经常不得不很小心地去处理各种使用情况。

#### Linking

linking with libraries: -lXXX

* statically-linked library:  libXXX.a(lib)
* dynamically-linked library : libXXX.so(dll)   
* -I /foo/bar : 头文件路径 compile line 
* -L 库文件路径: link line

Separate Compilation: -c, 只产生object file, 不link, 后面联合link-editor

#### LTO (Link Time Optimization)

  * 本质想解决的问题：编译 a.cpp 的时候看不到 b.cpp，编译器做不了优化
  * 解决方法：翻译 a.cpp 代码成中间语言 (LLVM IR Bitcode)，放到 a.o 里；链接阶段把它们都放在一起，一个大文件来做优化
  * 运行方式：linker调用编译器提供的plugin
  * 开启方式：`-flto`

##### GTC2022 - Automated Performance Improvement Using CUDA Link Time Optimization [S41595]

* CUDA 5.0：separate compilation

![nvcc-lto](Compiling/nvcc-lto.jpg)

![use-lto](Compiling/use-cuda-lto.png)

* LTO
  * how to use 如上图
  * Partial LTO，需要 execuable 支持 LTO
* JIT LTO (just in time LTO)
  * linking is performed at runtime
  * Generation of LTO IR is either offline with nvcc, or at runtime with nvrtc

![jit-lto](Compiling/jit-lto.png)

* Use JIT LTO
  * 用法见下图
  * The CUDA math libraries (cuFFT, cuSPARSE, etc) are starting to use JIT LTO; see [GTC Fall 2021 talk “JIT LTO Adoption in cuSPARSE/cuFFT: Use Case Overview](https://www.nvidia.com/en-us/on-demand/session/gtcfall21-a31155?playlistId=playList-ead11304-9931-4e91-9d5a-fb0e1ef27014)”
    * indirect user callback 转化为 JIT LTO callback
    * another use case: configure the used kernels ---> minimal library size

```c++
// Use nvrtc to generate the LTOIR (“input” is CUDA C++ string):
nvrtcProgram prog;
nvrtcCreateProgram(&prog, input, name, 0, nullptr, nullptr);
const char *options[2] = {"-dlto", "-dc"};
const nvrtcResult result = nvrtcCompileProgram(prog, 2, options);
size_t irSize;
nvrtcGetNVVMSize(prog, &irSize);
char *ltoIR = (char*)malloc(irSize);
nvrtcGetNVVM(prog, ltoIR); // returns LTO IR

// LTO inputs are then passed to cuLink* driver APIs, so linking is performed at runtime
CUlinkState state;
CUjit_option jitOptions[] = {CUjit_option::CU_JIT_LTO};
void *jitOptionValues[] = {(void*) 1};
cuLinkCreate(1, jitOptions, jitOptionValues, &state);
cuLinkAddData(state, CUjitInputType::CU_JIT_INPUT_NVVM,
ltoIR, irSize, name, 0, NULL, NULL);
cuLinkAddData( /* another input */);
size_t size;
void *linkedCubin;
cuLinkComplete(state, linkedCubin, &size);
cuModuleLoadData(&mod, linkedCubin);

// Math libraries hide the cuLink details in their CreatePlan APIs.
```

* LTO WITH REFERENCE INFORMATION
  * Starting in CUDA 11.7, nvcc will track host references to device code, which LTO can use to remove unused code. 
  * JIT LTO needs user to tell it this information, so new cuLinkCreate options:
    * CU_JIT_REFERENCED_KERNEL_NAMES
    * CU_JIT_REFERENCED_VARIABLE_NAMES
    * CU_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES
    * The *NAMES strings use implicit wildcards, so “foo” will match a mangled name like “Z3fooi”.

```c++
__device__ int array1[1024];
__device__ int array2[256];
__global__ void kernel1 (void) {
… array1[i]…
}
__global__ void kernel2 (void) {
… array2[i]…
}
….
kernel2<<<1,1>>>(); // host code launches kernel2
```

* 收益来源
  * Much of the speedup comes from cross-file inlining, which then helps keep the data in registers. 
  * Seeing the whole callgraph also helps to remove any dead code.
* References:
  * https://developer.nvidia.com/blog/improving-gpu-app-performance-with-cuda-11-2-device-lto/ -- offline LTO
  * https://developer.nvidia.com/blog/discovering-new-features-in-cuda-11-4/ -- JIT LTO
  * https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#optimization-of-separate-compilation -- nvcc
  * https://docs.nvidia.com/cuda/nvrtc/index.html -- nvrtc
  * https://docs.nvidia.com/cuda/nvrtc/index.html -- cuLink APIs
  * https://docs.nvidia.com/cuda/nvrtc/index.html -- compatibility guarantees
  * [Application paper](https://www.osti.gov/biblio/1798430-enhancements-supporting-ic-usage-pem-libraries-next-gen-platforms)

#### PGO (Profile Guided Optimization)

* 

### C++

* 常用编译宏
  * inline
    * inline 的坏处：代码变多了，变量变多了，可能寄存器不够分配了，只能偷内存，性能变差，尤其是发生在 loop 中
    * 编译器基本无视普通的 inline 关键字，根据自己的决策来做，内部有 cost model 评判 inline 是否有收益
    * [如果一个inline会在多个源文件中被用到，那么必须把它定义在头文件中](https://gist.github.com/hongyangqin/a7638016a78610f318d00d9a421ad6c9)，否则会找不到符号

```c
#pragma once

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#define DISABLE_COPY(T) \
T(const T&) = delete; \
T& operator=(const T&) = delete

#define FORCEDINLINE __attribute__((always_inline))
```

* 解决头文件互相引用的问题
  * 至少需要一方是使用指针，或者皆为指针，或者作为函数参数。不能同时都在类内定义实体对象。
  * 在二者之一的类中包含另一个的头文件，另一个头文件采用class xxx;的方式声明，并在cpp文件中包含头文件。

* 不同操作系统的编译:

```c++
#ifdef __APPLE__
	#include "TargetConditionals.h"
	#ifdef TARGET_OS_MAC
		#include <GLUT/glut.h>
		#include <OpenGL/OpenGL.h>
	#endif
#elif defined _WIN32 || defined _WIN64
	#include <GL\glut.h>
#elif defined __LINUX__
	XXX
#endif
```

* Multiple definition报错的讨论：如果在.h文件里定义函数，需要[加inline防止重复定义](https://softwareengineering.stackexchange.com/questions/339486/when-a-function-should-be-declared-inline-in-c)
  * [inline specifier in C++](https://en.cppreference.com/w/cpp/language/inline)支持重复引用.h内函数
  * 如果是gcc而非g++编译，会报错
* 保证只编译一次
  * 方法一：`#pragma once`
  * 方法二：

```c++
#ifdef HEADER_H
#define HEADER_H
...
#endif
```

* [LD_PRELOAD Trick](https://www.baeldung.com/linux/ld_preload-trick-what-is)
  * [How to Show All Shared Libraries Used by Executables in Linux?](https://www.baeldung.com/linux/show-shared-libraries-executables)
  * Alternative: `/etc/ld.so.preload`，系统层面的替换

```shell
ldd /usr/bin/vim
objdump -p /usr/bin/vim | grep 'NEEDED'
awk '$NF!~/\.so/{next} {$0=$NF} !a[$0]++' /proc/1585728/maps
```

  * LD_PRELOAD 的应用
    * when two libraries export the same symbol and our program links with the wrong one
    * when an optimized or custom implementation of a library function should be preferred
    * various profiling and monitoring tools widely use LD_PRELOAD for instrumenting code

```shell
LD_PRELOAD="/data/preload/lib/malloc_interpose.so:/data/preload/lib/free_interpose.so" ls -lh
```

* C++ 参数 overload 匹配优先级
  * 原生的类型优于自定义类型
  * 标准中的匹配优先级：overload resolution 章节
    * https://timsong-cpp.github.io/cppwp/n4618/over.match#over.ics.user
  * [call of overloaded brace-enclosed initializer list is ambiguous, how to deal with that?](https://stackoverflow.com/questions/14587436/call-of-overloaded-brace-enclosed-initializer-list-is-ambiguous-how-to-deal-w)

```c++
#include <iostream>

struct A{
        A(int) {}
};

void foo(A a) {std::cout << "foo(A a)" << std::endl; }
void foo(int b) { std::cout << "foo(int)" << std::endl; }

int main() {
        foo({2});
}
```

* template dependent 的类/函数不会做编译检查
  * 因为如果传入的T确实有xxx的方法，它其实是合法的

```c++
template <typename T> class A : public B<T> {
  T foo();
  void bar() {
    foo()->xxx(2, 3);
  }
};
```

* Partial Solution: [显式模版实例化](https://docs.oracle.com/cd/E19205-01/821-0389/bkafk/index.html)

```c++
template class B<BaseT>;
```

* forward declare a template class

```c++
template<typename Type, typename IDType=typename Type::IDType>
class Mappings;
template<typename Type, typename IDType>
class Mappings
{
public:
    ...
    Type valueFor(const IDType& id) { // return value }
    ...
};
```

* `(void)n;` 的用处：消除编译器对无用变量的报警



* 查看编译时 local object 的构造

```shell
readelf -sW my_bin |grep LOCAL|grep OBJECT | grep -v __PRETTY_FUNCTION__|grep -v vlocal__|grep -v '\.'|awk -F ' ' '{print $8}' | c++filt|sort|uniq -c |sort -n -k1 |grep -v vtable|grep -v '1 '|grep -v typeinfo 
```



#### pragma

* `pack` pragma
  * https://docs.microsoft.com/en-us/cpp/preprocessor/pack?view=msvc-170

```c++
#pragma pack( show )
#pragma pack( push [ , identifier ] [ , n ] )
#pragma pack( pop [ , { identifier | n } ] )
#pragma pack( [ n ] )
```

#### 坑

* ```C++
  Foo f(1);
  int main() {
      std::cout << f.x << std::endl;
      Foo f = (f);
      std::cout << f.x << std::endl;
      return 0;
  }
  // 1 0
  ```

  * 太坑了。。。clang有warning，gcc8的会导致f的值随机 https://godbolt.org/z/Wr4oTbWbz
  * warning: variable 'f' is uninitialized when used within its own initialization [-Wuninitialized]

### 基本工具

#### gcc

* libc: Linux 下的 ANSI C 函数库

* gcc

  * cpp文件预处理相关的# $\longrightarrow$ cc1: 由C到汇编 $\longrightarrow$ ac：assembler $\longrightarrow$ ld: linker

* flags

  * -o：output
  * -Wall：better warnings，显示所有错误警告
    *  https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html
  * -g：保留调试符号信息，开-g的时候不要开-O
  * -O     optimization
  * `-D ABC` 定义宏
  * -E  寻找所有依赖
  * -pedantic
  * -std=c++17
  * One issue with mem.c is that address space randomization is usually on by default. To turn it off: Just compile/link as follows: gcc -o mem mem.c -Wall -Wl,-no_pie
  * -march=native
    * https://gcc.gnu.org/onlinedocs/gcc-4.5.3/gcc/i386-and-x86_002d64-Options.html
    * will enable all instruction subsets supported by the local machine (hence the result might not run on different machines)
  * -m32
  
  * `FLAGS = -Wall -pthread, INCLUDES = ../include, gcc -I $(INCLUDES) -o t0 t0.c $(FLAGS)`

#### clang

* Clang 线程安全注解 https://clang.llvm.org/docs/ThreadSafetyAnalysis.html
  * 以 muduo 库为例：[介绍文章](https://zhuanlan.zhihu.com/p/47837673)
    * [Enable Clang Thread Safety Analysis. · chenshuo/muduo@4cc25e6](https://link.zhihu.com/?target=https%3A//github.com/chenshuo/muduo/commit/4cc25e67a0384fa4332ed49d2a5c73eceb002716) 
    * 需要自己包一个Mutex类，加上注解
      * `GUARDED_BY(mutex_);`
      * 对于 MutexLock:`CAPABILITY("mutex")`
      * 对于 MutexLockGuard: `SCOPED_CAPABILITY`
  * `clang -Wthread-safety`

##### 《C/C++ thread safety analysis》

* The annotations can be written using either GNU-style attributes (e.g., attribute ((...)) ) or C++11-style attributes (e.g., [[...]] ). For portability, the attributes are typically hidden behind macros that are disabled when not compiling with Clang.
  * 也可用于 Thread Role 的 annotation
* Basic Concepts
  * Capabilities can be either unique or shared.
  * Background: Uniqueness and Linear Logic
    * 编译器的概念，许多对象是linear的，例子如 string stream
    * Functions that use the object without consuming it must be written using a hand-off protocol
  * Clang thread safety analysis tracks capabilities as unnamed objects that are passed implicitly.
    * `Cap<mu>`
* THREAD SAFETY ANNOTATIONS
  * GUARDED BY(...) and PT_GUARDED_BY(...) GUARDED
  * REQUIRES(...) and REQUIRES_SHARED(...) REQUIRES
    * a function takes the given capability as an implicit argument and hands it back to the caller when it returns, as an implicit result.
  * ACQUIRE(...) and RELEASE(...)
  * CAPABILITY(...)
  * TRY_ACQUIRE(b, ...) and TRY_ACQUIRE_SHARED(b, ...)
  * NO_THREAD_SAFETY_ANALYSIS
  * Negative Requirements
    * REQUIRE(!mu)

```c++
class CAPABILITY(”mutex”) Mutex {
  public :
  	void lock() ACQUIRE( this );
		void readerLock() ACQUIRE_SHARED( this );
  	void unlock() RELEASE( this );
		void readerUnlock() RELEASE_SHARED( this );
}
```

* Implementation
  * Clang
    * Clang initially parses a C++ input file to an abstract syntax tree (AST), which is an accurate representation of the original source code, down to the location of parentheses. In contrast, many compilers, including GCC, lower to an intermediate language during parsing. The accuracy of the AST makes it easier to emit quality diagnostics, but complicates the analysis in other respects.
    * The Clang semantic analyzer (Sema) decorates the AST with semantic information. Name lookup, function overloading, operator overloading, template instantiation, and type checking are all performed by Sema when constructing the AST. Clang inserts special AST nodes for implicit C++ operations, such as automatic casts, LValue-to-RValue conversions, implicit destructor calls, and so on, so the AST provides an accurate model of C++ program semantics.
    * Finally, the Clang analysis infrastructure constructs a control flow graph (CFG) for each function in the AST. This is not a lowering step; each statement in the CFG points back to the AST node that created it. The CFG is shared infrastructure; the thread safety analysis is only one of its many clients.
  * Analysis Algorithm
    * performing a topological sort of the CFG, and identifying back edges
    * Requiring that capability sets be the same:
      *  a joint point
      * back edges (like loops)
  * Intermediate Representation
    * A dependent type system must be able to compare expres-sions for semantic (not syntactic) equality. The analyzer im-plements a simple compiler intermediate representation (IR), and lowers Clang expressions to the IR for comparison. It
      also converts the Clang CFG into single static assignment (SSA) form so that the analyzer will not be confused by local variables that take on different values in different places.
  * Limitations
    * No attributes on types.
      * it was deemed infeasible for C++ because it would require invasive changes to the C++ type system that could potentially affect core C++ semantics in subtle ways, such as template instantiation and function overloading.
    * No dependent type parameters.
    * No alias analysis.
      * Pointer aliasing -> false negatives
* Google’s philosophy is that incorrect annotations are “bugs in the documentation.”



#### bazel

* https://bazel.build/?hl=en
  * 构建快。支持增量编译。对依赖关系进行了优化，从而支持并发执行。
  * 可构建多种语言。bazel可用来构建Java C++ Android ios等很多语言和框架，并支持mac windows linux等不同平台
  * 可伸缩。可处理任意大小的代码库，可处理多个库，也可以处理单个库
  * 可扩展。使用bazel扩展语言可支持新语言和新平台。

```
wget https://github.com/bazelbuild/bazel/releases/download/3.1.0/bazel-3.1.0-installer-linux-x86_64.sh && \
  mkdir -p  /opt/abc/bazel && \
  chmod +x bazel-3.1.0-installer-linux-x86_64.sh && \
  ./bazel-3.1.0-installer-linux-x86_64.sh --prefix=/opt/abc/bazel && \
  rm bazel-3.1.0-installer-linux-x86_64.sh
export PATH=/opt/abc/bazel/bin:$PATH
```

* Basilisk:
  * https://github.com/bazelbuild/bazelisk
  * https://stackoverflow.com/questions/65656165/how-do-you-install-bazel-using-bazelisk

```shell
> wget https://github.com/bazelbuild/bazelisk/releases/download/v1.15.0/bazelisk-linux-amd64
> chmod +x bazelisk-linux-amd64
> sudo mv bazelisk-linux-amd64 /usr/local/bin/bazel
     
# make sure you get the binary available in $PATH
> which bazel
bazel is /usr/local/bin/bazel
```

* Ide integration
  * vscode bazel插件
  * intellisense: https://github.com/hedronvision/bazel-compile-commands-extractor



* 官方tutorial（没有写动态静态链接库so怎么生成）
  * https://bazel.build/start/cpp?hl=en
  * https://bazel.build/docs/bazel-and-cpp
  * To keep focusing on C++, read about common [C++ build use cases](https://bazel.build/tutorials/cpp-use-cases).
  * To get started with building other applications with Bazel, see the tutorials for [Java](https://bazel.build/tutorials/java), [Android application](https://bazel.build/tutorials/android-app), or [iOS application](https://bazel.build/tutorials/ios-app).
  * To learn more about working with local and remote repositories, read about [external dependencies](https://bazel.build/docs/external).
  * To learn more about Bazel’s other rules, see this [reference guide](https://bazel.build/rules).

```
git clone https://github.com/bazelbuild/examples
cd cpp-tutorial/stage1
bazel build //main:hello-world
```

```bazel
# src/BUILD
cc_library(
    name = "mylib",
    srcs = ["mylib.cc"],
    hdrs = ["mylib.h"],
    deps = [":lower-level-lib"]
)

cc_test(
    name = "mylib_test",
    srcs = ["mylib_test.cc"],
    deps = [":mylib"]
)

cc_library(
    name = "hello-time",
    srcs = ["hello-time.cc"],
    hdrs = ["hello-time.h"],
    visibility = ["//main:__pkg__"],
)
```

* 动态库使用

```
# src/BUILD
cc_library(
    name = "hello-time.so",
    srcs = ["hello-time.cc"],
    hdrs = ["hello-time.h"],
    linkshared=True,//设置为生成动态库
    visibility = ["//main:__pkg__"],
)

# main/BUILD

cc_import(                             
    name = "hello-time",
    hdrs = ["hello-time.h"],
    shared_library = "hello-time.so",//链接src中生成的so文件
)

cc_binary(
    name = "hello-demo",//目标二进制文件名
    srcs = [
        "hello-demo.c",
        "hello-time.h",
    ],//源文件
    deps = [
        "//src:hello-time",
    ],//依赖，必须先完成cc_import，才能完成cc_binary
    copts = ["-lm"],//附加选项，需要math库
)
```





* Concepts: https://bazel.build/concepts/build-ref

  * 项目结构：Workspace, packages, targets
    * BUILD file 对应 target
    * 给target引入依赖的两种方式：rule、src file
    * Package groups are sets of packages whose purpose is to limit accessibility of certain rules. Package groups are defined by the `package_group` function. They have three properties: the list of packages they contain, their name, and other package groups they include. The only allowed ways to refer to them are from the `visibility` attribute of rules or from the `default_visibility` attribute of the `package` function; they do not generate or consume files. For more information, refer to the [`package_group` documentation](https://bazel.build/reference/be/functions#package_group).
  * Label
    * 同一个包下，标签可以省略包名部分，如:app_binary表示同一个包下的目标。不同包之间，则千万不能省略包名。

* Follow these guidelines for include paths:

  - Make all include paths relative to the workspace directory.

  - Use quoted includes (`#include "foo/bar/baz.h"`) for non-system headers, not angle-brackets (`#include <foo/bar/baz.h>`).

  - Avoid using UNIX directory shortcuts, such as `.` (current directory) or `..` (parent directory).

  - For legacy or `third_party` code that requires includes pointing outside the project repository, such as external repository includes requiring a prefix, use the [`include_prefix`](https://bazel.build/reference/be/c-cpp#cc_library.include_prefix) and [`strip_include_prefix`](https://bazel.build/reference/be/c-cpp#cc_library.strip_include_prefix) arguments on the `cc_library` rule target.

* C/C++ rules：https://bazel.build/reference/be/c-cpp#rules
  * cc_binary 生成二进制文件
    * [select语法](https://bazel.build/docs/configurable-attributes)
  * cc_import 导入链接库
  * cc_library 生成链接库
* Command Line Ref: https://bazel.build/reference/command-line-reference
  * [清除build结果](https://bazel.build/docs/user-manual#cleaning-build-outputs)(from User Manual)
    * `bazel clean --expunge` 清除编译链
  * `--override_repository=xxx=$(pwd)/third_party/xxx`
  * `--output_filter=IGNORE_LOGS`
* Workspace 规则: https://bazel.build/docs/external
  * 在需要**远程依赖**的支持或导入**本地非当前目录下的依赖**需要使用
  * Depending on other Bazel projects
    * `@coworkers_project//foo:bar`
  * Depending on non-Bazel projects
    * new_http_archive: 下载文件，然后解压它，然后和其中包含的build_file一起创建bazel目录
    * https://bazel.build/reference/be/workspace#new_local_repository
  * bazel fetch, bazel sync
  * Prefer [`http_archive`](https://bazel.build/rules/lib/repo/http#http_archive) to `git_repository` and `new_git_repository`.
  * Tf serving的例子：
    * https://github.com/tensorflow/serving/blob/master/WORKSPACE
    * https://github.com/tensorflow/serving/blob/master/tensorflow_serving/repo.bzl

```bazel
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
```

```
""" TensorFlow Http Archive
Modified http_archive that allows us to override the TensorFlow commit that is
downloaded by setting an environment variable. This override is to be used for
testing purposes.
Add the following to your Bazel build command in order to override the
TensorFlow revision.
build: --action_env TF_REVISION="<git commit hash>"
  * `TF_REVISION`: tensorflow revision override (git commit hash)
"""

_TF_REVISION = "TF_REVISION"

def _tensorflow_http_archive(ctx):
    git_commit = ctx.attr.git_commit
    sha256 = ctx.attr.sha256
    patch = getattr(ctx.attr, "patch", None)

    override_git_commit = ctx.os.environ.get(_TF_REVISION)
    if override_git_commit:
        sha256 = ""
        git_commit = override_git_commit

    strip_prefix = "tensorflow-%s" % git_commit
    urls = [
        "https://mirror.bazel.build/github.com/tensorflow/tensorflow/archive/%s.tar.gz" % git_commit,
        "https://github.com/tensorflow/tensorflow/archive/%s.tar.gz" % git_commit,
    ]
    ctx.download_and_extract(
        urls,
        "",
        sha256,
        "",
        strip_prefix,
    )
    if patch:
        ctx.patch(patch, strip = 1)

tensorflow_http_archive = repository_rule(
    implementation = _tensorflow_http_archive,
    attrs = {
        "git_commit": attr.string(mandatory = True),
        "sha256": attr.string(mandatory = True),
        "patch": attr.label(),
    },
)
```

* bazel远程构建 https://zhuanlan.zhihu.com/p/266510840

```python
write_to_bazelrc("\n".join([
      "# Bazel building cluster",
      "build --spawn_strategy=remote,local",
      "build --jobs %s --remote_executor=grpc://%s:%d" %
      (best_cluster.jobs, best_cluster.host, best_cluster.port),
  ]))
```

* 以 tensorflow/cc/BUILD 为例
  * 为多个编译目标target指定一个名字，glob是一个帮助函数，指定了目录中哪些文件会include，哪些会exclude。visibility指定了target的可见性，也就是可以被哪些package调用

```
native.filegroup(
		name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
```



* 坑
  * [bazel: protobuf依赖](https://zhuanlan.zhihu.com/p/488199658)
    * pb冲突: `build --sandbox_block_path=/usr/local`
    * bazel4: `build --incompatible_blacklisted_protos_requires_proto_info=false`



#### blade

https://github.com/chen3feng/blade-build/blob/master/doc/en/command_line.md

```shell
blade build (folder):target --toolchain=x86_64-gcc830 --bundle=debug --cxxflags="-D ABC"

--generate-dynamic
## cc_library可以通过--generate-dynamic来生成动态库，不要用这种方式来生成动态库作为终端产物，推荐使用cc_plugin，cc_library一般作为cc_binary的前置依赖存在
-p debug/release
# debug：关闭配置在BUILD文件，或者BLADE_ROOT文件中的所有optimize参数，并且追加诸                              如-g类型的编译参数
# release：默认打开配置在BUILD文件等配置文件中的optimize参数


blade query :target --deps --output-tre
```

BUILD文件怎么写

```python
# BUILD
load("//workspace/BUILD.share", "*")

# BUILD.share
def_keys = ['a', 'b', 'c']
for key in def_keys:
    if os.getenv(key):
        defs.append(key)
```

过滤链接系统库的so

```python
--filterflags="-lssl -lcrypto -lcrypt -levent -lz -lbz2 -lmsgpack"
```

#### make

* make install
  * https://superuser.com/questions/360178/what-does-make-install-do
