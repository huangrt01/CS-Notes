### Theory

* [Shared Library Symbol Conflicts (on Linux)](https://holtstrom.com/michael/blog/post/437/Shared-Library-Symbol-Conflicts-(on-Linux).html)
  * 从左往右查找：Note that the linker only looks further down the line when looking for symbols used by but not defined in the current lib.
* [Linux 下 C++so 热更新](https://zhuanlan.zhihu.com/p/162366167)
* ABI (Application Binary Interface)
  * 应用程序的二进制接口，对于一个二进制的动态库或者静态库而言，可以详细描述在其中的函数的调用方式，定义在其中的数据类型的大小，数据结构的内存布局方式等信息
  * ABI 信息 对不同操作系统、不同编译链版本、不同二进制库对应源码版本 有或大或小的差异，从而造成预编译二进制库的兼容性问题，导致 compile error 或 执行时coredump
* 编译器有能力让不同 target 的 cpp 文件的不同编译选项，有区分地生效。但无法控制其它cpp文件对头文件的使用，因此头文件为主体的开源项目，经常不得不很小心地去处理各种使用情况。
* LTO (Link Time Optimization)
  * 本质想解决的问题：编译 a.cpp 的时候看不到 b.cpp，编译器做不了优化
  * 解决方法：翻译 a.cpp 代码成中间语言 (LLVM IR Bitcode)，放到 a.o 里；链接阶段把它们都放在一起，一个大文件来做优化
  * 运行方式：linker调用编译器提供的plugin

### C++

* 常用编译宏
  * 编译器基本无视普通的 inline 关键字，根据自己的决策来做
    * inline 的坏处：代码变多了，变量变多了，可能寄存器不够分配了，只能偷内存，性能变差，尤其是发生在 loop 中
    * 编译器内部有 cost model 评判 inline 是否有收益

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
awk ' <img src="https://www.zhihu.com/equation?tex=NF%21~/%5C.so/%7Bnext%7D%20%7B" alt="NF!~/\.so/{next} {" class="ee_img tr_noresize" eeimg="1"> 0= <img src="https://www.zhihu.com/equation?tex=NF%7D%20%21a%5B" alt="NF} !a[" class="ee_img tr_noresize" eeimg="1"> 0]++' /proc/1585728/maps
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





* 查看编译时 local object 的构造

```shell
readelf -sW my_bin |grep LOCAL|grep OBJECT | grep -v __PRETTY_FUNCTION__|grep -v vlocal__|grep -v '\.'|awk -F ' ' '{print $8}' | c++filt|sort|uniq -c |sort -n -k1 |grep -v vtable|grep -v '1 '|grep -v typeinfo 
```







### 基本工具

```shell
gcc -D ABC     # 定义宏
```

```shell
blade build :target --toolchain=x86_64-gcc830 --bundle=debug --cxxflags="-D ABC"
blade query :target --deps --output-tre

--generate-dynamic
## cc_library可以通过--generate-dynamic来生成动态库，不要用这种方式来生成动态库作为终端产物，推荐使用cc_plugin，cc_library一般作为cc_binary的前置依赖存在
```
