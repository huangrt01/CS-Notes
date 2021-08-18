### Theory

* [Shared Library Symbol Conflicts (on Linux)](https://holtstrom.com/michael/blog/post/437/Shared-Library-Symbol-Conflicts-(on-Linux).html)
  * 从左往右查找：Note that the linker only looks further down the line when looking for symbols used by but not defined in the current lib.

* [Linux 下 C++so 热更新](https://zhuanlan.zhihu.com/p/162366167)

* ABI (Application Binary Interface)
  * 应用程序的二进制接口，对于一个二进制的动态库或者静态库而言，可以详细描述在其中的函数的调用方式，定义在其中的数据类型的大小，数据结构的内存布局方式等信息
  * ABI 信息 对不同操作系统、不同编译链版本、不同二进制库对应源码版本 有或大或小的差异，从而造成预编译二进制库的兼容性问题，导致 compile error 或 执行时coredump

### C++

* 常用编译宏

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

  * LD_PRELOAD的应用
    * when two libraries export the same symbol and our program links with the wrong one
    * when an optimized or custom implementation of a library function should be preferred
    * various profiling and monitoring tools widely use LD_PRELOAD for instrumenting code

```shell
LD_PRELOAD="/data/preload/lib/malloc_interpose.so:/data/preload/lib/free_interpose.so" ls -lh
```

* C++参数overload匹配优先级
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
