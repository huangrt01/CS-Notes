[toc]

各种资源

* [大神 Andrei Alexandrescu 的 ppt，讲C语言优化](https://www.slideshare.net/andreialexandrescu1/three-optimization-tips-for-c-15708507)
* https://github.com/miloyip/itoa-benchmark/


### 《Effective Modern C++》, Scott Meyers 

**Intro**

* lvalue: 和地址密切联系，all parameters are lvalues
* function object可以是定义了operator ()的object，只要有function-calling syntax就行
* deprecated feature: `std::auto_ptr -> std::unique_ptr`
* Undefined behavior: 数组越界、dereferencing an uninitialized iterator、engaging in a data race

#### chpt 1 Deducing Types

deducing types的机制：

* c++98: function template 

  =>

* c++11: auto, decltype

  =>

* c++14: decltype(auto)

##### Item 1: Understand template type deduction.

auto的deduction机制和template type deduction紧密相联

模版类型推断是综合性的，不止和T相关，也和ParamType相关(adornments)，有三种情形

```c++
template<typename T> 
void f(ParamType param);

f(expr); // deduce T and ParamType from expr
```

* Case 1: **ParamType** is a Reference or Pointer, but not a Universal

  Reference

* Case 2: **ParamType** is a Universal Reference
  * 右值->右值引用；左值->左值引用

  * 唯一T会deduce为引用的场景

* Case 3: **ParamType** is Neither a Pointer nor a Reference
  * 去除const、&等修饰
  * ` const char* const ptr =  // ptr is const pointer to const object
     "Fun with pointers";` 保留左边的const
  
* array arguments
  
  * the type of an array that’s passed to a template function by value is deduced to be a pointer type，需要注意数组传引用的时候不一样！存在数组引用`constchar(&)[13]` !

```c++
// return size of an array as a compile-time constant. (The
// array parameter has no name, because we care only about
// the number of elements it contains.)
template<typename T, std::size_t N> // see info 
constexpr std::size_t arraySize(T (&)[N]) noexcept // below on 
{																									// constexpr
	return N; // and
}// noexcept

int keyVals[] = { 1, 3, 7, 9, 11, 22, 35 };      // keyVals has
                                                    // 7 elements
int mappedVals[arraySize(keyVals)]; // so does // mappedVals

std::array<int, arraySize(keyVals)> mappedVals; // mappedVals' 
																								// size is 7
```

##### Item 2: Understand **auto** type deduction.

```c++
template<typename T> 
void f(ParamType param);

f(expr); // deduce T and ParamType from expr
```

auto对应T，type specifier对应ParamType，因此同Item1，也有三个cases，但有一个exception

唯一的exception：`auto x = { 20 }; `  the deduced type is a std::initializer_list，如果里面元素类型不一致，不会编译
* the only real difference between auto and template type deduction is that auto assumes that a braced initializer represents a std::initializer_list, but template type deduction doesn’t

Things to Remember
* auto type deduction is usually the same as template type deduction, but auto type deduction assumes that a braced initializer represents a std::initial izer_list, and template type deduction doesn’t.
* auto in a function return type or a lambda parameter implies template type deduction, not auto type deduction.

##### Item 3: Understand decltype.

In C++11, perhaps the primary use for decltype is declaring function templates where the function’s return type depends on its parameter types.

* the type returned by a container’s operator[] depends on the container，比如`vector<bool>`返回的不是引用
  * c++14甚至可以不需要->，用模版判断，但存在无法引用的问题

```c++
template<typename Container, typename Index>  // works, but requires 			
auto authAndAccess(Container& c, Index i)     // refinement
	-> decltype(c[i])
{
  authenticateUser();
  return c[i];
}

template<typename Container, typename Index> // C++14; works,
decltype(auto)															 // but still requires 
authAndAccess(Container& c, Index i)				 // refinement
{
  authenticateUser();
  return c[i];
}


```

* decltype(auto)利用decltype rule，帮助单独的auto返回引用
  * 本质上是因为auto难以识别type specifier
  * decltype(**(x)**) is therefore int&. Putting parentheses around a name can change the type that decltype reports for it!

##### Item 4: Know how to view deduced types.

```c++
template<typename T>       // declaration only for TD;
class TD;                  // TD == "Type Displayer"

TD<decltype(x)> xType;     // elicit errors containing
TD<decltype(y)> yType;     // x's and y's types

///
std::cout << typeid(x).name() << '\n'; // display types for std::cout << typeid(y).name() << '\n'; // x and y
```

`std::type_info::name` 对模版中变量类型的判断不准

```c++
#include <boost/type_index.hpp>
template<typename T>
void f(const T& param)
{
  using std::cout;
  using boost::typeindex::type_id_with_cvr;
  
  // show T
  cout << "T = "<< type_id_with_cvr<T>().pretty_name() << '\n';
  // show param's type
  cout << "param = "<< type_id_with_cvr<decltype(param)>().pretty_name() 
    	 << '\n'; 
}
```

#### chpt 2 auto

##### Item 5: Prefer auto to explicit type declarations.
```c++
template<typename It>
void dwim(It b, It e)
{
  while (b != e) {
// algorithm to dwim ("do what I mean")
// for all elements in range from
// b to e
		typename std::iterator_traits<It>::value_type 
			currValue = *b;
	} 
}
```

auto必须初始化，潜在地避免未初始化的问题

```c++
auto derefUPLess =
[](const std::unique_ptr<Widget>& p1,
const std::unique_ptr<Widget>& p2) 
{ return *p1 < *p2; };

// C++14内部也可用auto

std::function<bool(const std::unique_ptr<Widget>&, const std::unique_ptr<Widget>&)>
derefUPLess = [](const std::unique_ptr<Widget>& p1, const std::unique_ptr<Widget>& p2)
               { return *p1 < *p2; };

// std::function相比auto占用内存多，效率低（对象统一进行了转化）
```

##### Item 6: Use the explicitly typed initializer idiom when **auto** deduces undesired types.

`有关vector<bool>：because operator[] for std::vector<T> is supposed to return a T&, but C++ forbids references to bits.`

`std::vector<bool>::reference` is an example of a proxy class: a class that exists for the purpose of emulating and augmenting the behavior of some other types

proxy class
* invisible: `vector<bool>`
* apparent: `std::shared_ptr, std::unique_ptr`
* expression templates: `Matrix sum = m1 + m2 + m3 + m4;`



#### chpt3: Moving to Modern C++

##### Item 7: Distinguish between () and {} when creating objects.

`std::vector<int> v{ 1, 3, 5 }; // v's initial content is 1, 3, 5`

**特性**：

* uncopyable objects (e.g., std::atomics—see Item 40) 

`std::atomic<int> ai1{ 0 }; // fine`

*  it prohibits implicit *narrowing conversions* among built-in types
*  immunity to C++’s *most vexing parse*
  * A side effect of C++’s rule that anything that can be parsed as a declaration must be interpreted as one

**缺点**：

* the unusually tangled relationship among braced initializers, std::initializer_lists, and constructor overload resolution

  * `Widget(std::initializer_list<long double> il);`对{}的优先级极高，只有在实在转换不了的情况下才会用non-std::initializer_list constructors
  * Empty braces mean no arguments, not an empty std::initializer_list
  * vector对{}和()初始化的区分是不合理的
  * 因此：重载initializer_list参数的函数尤其要谨慎

  

##### Item 8: Prefer nullptr to 0 and NULL.

1. 0和NULL混用带来函数重载问题 -> counterintuitive behavior 
2. nullptr’s advantage is that it doesn’t have an integral type, but you can think of it as a pointer of *all* types
3. NULL和0在template type deduction中问题更严重

##### Item 9: Prefer alias declarations to typedefs.

```c++
template<typename T> // MyAllocList<T> 
using MyAllocList = std::list<T, MyAlloc<T>>; // is synonym for
                                                  // std::list<T,
                                                  //   MyAlloc<T>>
MyAllocList<Widget> lw; // client code


template<typename T> // MyAllocList<T>::type 
struct MyAllocList { // is synonym for
typedef std::list<T, MyAlloc<T>> type; // std::list<T, 
}; 																			// MyAlloc<T>>
MyAllocList<Widget>::type lw;           // client code

template<typename T>
class Widget {                         // Widget<T> contains
 private:                               // a MyAllocList<T>
	typename MyAllocList<T>::type list; // as a data member
...
};


```

using的优势是能让编译器知道这是一个type，尤其在模版的使用中，避免混淆`MyAllocList<T>::type`

 template metaprogramming (TMP)

* C++11中的type traits是用typedef实现的，因此使用的时候要在前面声明`typename`

```c++
std::remove_const<T>::type					// yields T from const T
std::remove_reference<T>::type			// yields T from T& and T&&
std::add_lvalue_reference<T>::type	// yields T& from T
  
std::remove_const_t<T>
std::remove_reference_t<T>
std::add_lvalue_reference_t<T>
  
  
template <class T>
using remove_const_t = typename remove_const<T>::type;

template <class T>
using remove_reference_t = typename remove_reference<T>::type;

template <class T>
using add_lvalue_reference_t =
	typename add_lvalue_reference<T>::type;

```


##### Item 10: Prefer scoped **enum**s to unscoped **enum**s.

```c++
enum class Color { black, white, red };
auto c = Color::white;

enum Color;               // error!
enum class Color;         // fine
```

=> The reduction in namespace pollution 

=> There are no implicit conversions from enumerators in a scoped enum to any other type

=> scoped enums may always be forward-declared, unscoped enums只有指定了underlying type才可以

* 普通的enum默认是char，编译器需要选类最小的类型=> unscoped enums不需要大部分重新编译
* Scoped enums默认类型int，可指定

```c++
enum class Status: std::uint32_t; // underlying type for
																	// Status is std::uint32_t
                                  // (from <cstdint>)

```

unscoped enum更好的少数场景，uiEmail有implicit转换为size_t
```c++
enum UserInfoFields { uiName, uiEmail, uiReputation };
   UserInfo uInfo;                        // as before
   ...
auto val = std::get<uiEmail>(uInfo); // ah, get value of // email field


enum class UserInfoFields { uiName, uiEmail, uiReputation };
   UserInfo uInfo;                        // as before
   ...
auto val = std::get<static_cast<std::size_t>(UserInfoFields::uiEmail)>(uInfo);
```
如果用函数转，需要constexpr
```c++
template<typename E>
constexpr typename std::underlying_type<E>::type
	toUType(E enumerator) noexcept
{
	return static_cast<typename std::underlying_type<E>::type>(enumerator);
}
```

c++14可以用` static_cast<std::underlying_type_t<E>>(enumerator)`

```c++
template<typename E> // C++14 
constexpr auto
	toUType(E enumerator) noexcept
{
	return static_cast<std::underlying_type_t<E>>(enumerator); 
}
auto val = std::get<toUType(UserInfoFields::uiEmail)>(uInfo);
```



##### Item 11: Prefer deleted functions to private undefined ones.

e.g.: std::basic_ios，禁止copy streams

相比undefined private function

* 错误信息更清晰
* 可以delete非成员函数

* 针对模版

```c++
template<typename T>
void processPointer(T* ptr);

template<>
void processPointer<void>(void*) = delete;
template<>
void processPointer<char>(char*) = delete;
template<>
void processPointer<const void>(const void*) = delete;
template<>
void processPointer<const char>(const char*) = delete;

// const volatile char*, const volatile void*, std::wchar_t, std::char16_t, std::char32_t.

//如果在类内定义
template<>
void Widget::processPointer<void>(void*) = delete;
```



##### Item 12: Declare overriding functions **override**.

```c++
class Base {
public:
  virtual void doWork();
...
};
class Derived: public Base {
public:
  virtual void doWork();
...
};
std::unique_ptr<Base> upb = std::make_unique<Derived>();
upb->doWork();
// derived class function is invoked
```

* override的各种要求，还包括reference qualifier
* 改变signature方便看有多少依赖
* Applying **final** to a virtual function prevents the function from being overridden in derived classes. final may also be applied to a class, in which case the class is prohibited from being used as a base class.

*  Member function reference qualifiers make it possible to treat lvalue and rvalue objects (*this) differently.

```c++
class Widget {
public:
  using DataType = std::vector<double>;
  DataType& data() & { return values; }
  DataType data() && { return std::move(values); } ...
  private:
    DataType values;
};
```

##### Item 13: Prefer **const_iterator**s to **iterator**s.

**pointer-to-const == const_iterators**

there’s no portable conversion from a const_iterator to an iterator, not even with a static_cast. Even the semantic sledgehammer known as reinterpret_cast can’t do the job. 

```c++
std::vector<int> values; // as before ...
auto it = // use cbegin
	std::find(values.cbegin(),values.cend(), 1983); // and cend
values.insert(it, 1998);
```

```c++
template<typename C, typename V>
void findAndInsert(C& container,
{
  const V& targetVal,
  const V& insertVal)
  using std::cbegin;
  using std::cend;
  auto it = std::find(cbegin(container), cend(container),
  targetVal);
  container.insert(it, insertVal); 
}
                   
template <class C>
auto cbegin(const C& container)->decltype(std::begin(container)) {
  return std::begin(container);         // see explanation below
}
```

Non-member cbegin只有C++14才有；C++11中，即使是vector非const，begin返回的带const引用的vector也会是const_iterator类型



##### Item 14: Declare functions **noexcept** if they won’t emit exceptions.

noexcept意味着函数行为const

e.g. C++11，vector.push_back(std::move(XX))影响异常安全性，`std::vector::push_back` takes advantage of this “move if you can, but copy if you must” strategy，依据是否declared noexcept来判断

```c++
template <class T, size_t N>
void swap(T (&a)[N], // see
					T (&b)[N]) noexcept(noexcept(swap(*a, *b))); // below

template <class T1, class T2>
struct pair {
	...
	void swap(pair& p) noexcept(noexcept(swap(first, p.first)) && 						   								 					noexcept(swap(second, p.second)));
	...
};
```

* Wide contract functions更适合noexcept，因为没有后续调试抛出异常的需求



>Things to Remember
>
>• noexcept is part of a function’s interface, and that means that callers may depend on it.
>
>• noexcept functions are more optimizable than non-noexcept functions.
>
>• noexcept is particularly valuable for the move operations, swap, memory deallocation functions, and destructors.
>
>• Most functions are exception-neutral rather than noexcept.



##### Item 15: Use **constexpr** whenever possible.

* translation时知晓，包括compilation和linking
* constexpr变量声明时需要初始化
* constexpr function，用法参考Item 1
  * c++11只能写一句return

```c++
constexpr
int pow(int base, int exp) noexcept
{
...
}
constexpr auto numConds = 5;
std::array<int, pow(3, numConds)> results;

constexpr int pow(int base, int exp) noexcept     //c++11
{
	return (exp == 0 ? 1 : base * pow(base, exp - 1)); 
}

constexpr int pow(int base, int exp) noexcept
{
  auto result = 1;
  for (int i = 0; i < exp; ++i) result *= base;
  return result;
}

```

P100: 自定义的类型也可以constexpr

* C++14中即使是Point.set也可以constexpr



##### Item 16: Make **const** member functions thread safe.

```c++
class Polynomial {
  public:
    using RootsType = std::vector<double>;
    RootsType roots() const
    {
      std::lock_guard<std::mutex> g(m);
      if (!rootsAreValid) {
        ...
        rootsAreValid = true;
      }
      return rootVals;
    }
  private:
  	mutable std::mutex m;
    mutable bool rootsAreValid{ false };
    mutable RootsType rootVals{};
};
```

私有成员`mutable std::atomic<unsigned> callCount{ 0 };`适合用于计数

* std::mutex和std::atomic都是move-only types，不能copy
  * 如果想定义一个数组，数组中的对象包含atomic类型，不能用vector直接构造（因为会调用复制构造函数），可以构造`vector<shared_ptr<Object>>`
  * atomic适合递增操作，但如果是先运算后赋值，可能出现竞争

```c++
class Widget {
public:
...
  int magicValue() const
  {
    std::lock_guard<std::mutex> guard(m);
    if (cacheValid) return cachedValue;
    else {
      auto val1 = expensiveComputation1();
      auto val2 = expensiveComputation2();
      cachedValue = val1 + val2;
      cacheValid = true;
      return cachedValue;
    }
	}
private:
  mutable std::mutex m;
  mutable int cachedValue;
  mutable bool cacheValid{ false };
};
```



##### Item 17: Understand special member function generation

问题的来源：memberwise move的思路，move能move的，剩下的copy

**special member functions**: 

* c++98: the default constructor, the destructor, the copy constructor, and the copy assignment operator
* c++11: the move constructor and the move assignment operator
  * 和copy constructor的区别在于，c++11的这两个fucntion是dependent的
  * 如果有explicit copy constructor，move constructor不会自动生成

*the rule of three*: copy constructor, copy assignment operator, or destructor

动机：the rule of three和move/copy的dependent特性有冲突 => C++11 does *not* generate move operations for a class with a user-declared destructor

`=default`和`=delete`，前者在自己有写构造函数的情况下生成默认构造函数，减小代码量，后者禁止函数

```c++
//! moving is allowed; copying is disallowed; default construction not possible
//!@{
~TCPConnection();  //!< destructor sends a RST if the connection is still open
TCPConnection() = delete;
TCPConnection(TCPConnection &&other) = default;
TCPConnection &operator=(TCPConnection &&other) = default;
TCPConnection(const TCPConnection &other) = delete;
TCPConnection &operator=(const TCPConnection &other) = delete;
//!@}
```

Note: Member function templates never suppress generation of special member functions.





#### chpt 4: Smart Pointers

##### Item 18: Use std::unique_ptr for exclusive-ownership resource management.

* unique_ptr is a move-only type
* also applied to the Pimpl Idiom (Item 22)
* 还有`std::unique_ptr<T[]>`
  * About the only situation I can conceive of when a `std::unique_ptr<T[]>` would make sense would be when you’re using a C-like API that returns a raw pointer to a heap array that you assume ownership of.
* `std::shared_ptr<Investment> sp = makeInvestment( arguments );` 

```c++
class Investment {
public:
  virtual ~Investment();
};
class Stock:
public Investment { ... };
class Bond:
public Investment { ... };
class RealEstate:
public Investment { ... };

template<typename... Ts> std::unique_ptr<Investment> makeInvestment(Ts&&... params);

{
	auto pInvestment = // pInvestment is of type 
		makeInvestment( arguments ); // std::unique_ptr<Investment>
}// destroy *pInvestment
```

```c++
auto delInvmt = [](Investment* pInvestment)
                {
                  makeLogEntry(pInvestment);
                  delete pInvestment;
                };

template<typename... Ts>
std::unique_ptr<Investment, decltype(delInvmt)> // 如果是C++14，返回值可以用auto，delInvmt的定义也可放入makeInvestment函数内
makeInvestment(Ts&&... params) {
	std::unique_ptr<Investment, decltype(delInvmt)> pInv(nullptr, delInvmt);
  if ( /* a Stock object should be created */ ){
		pInv.reset(new Stock(std::forward<Ts>(params)...)); 
  }
  else if ( /* a Bond object should be created */ ){
		pInv.reset(new Bond(std::forward<Ts>(params)...)); 
  }
  else if ( /* a RealEstate object should be created */ ){
		pInv.reset(new RealEstate(std::forward<Ts>(params)...)); 
  }
  return pInv;
}

```

代码细节：

* `std::forward`: 为了在使用右值引用参数的函数模板中解决参数的perfect-forward问题。(Item 25), [reference](https://blog.csdn.net/zhangsj1007/article/details/81149719?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-6.edu_weight&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-6.edu_weight)
* unique_ptr可以作为函数返回值
* 不能直接给unique_ptr赋值裸指针，`reset(new)`是标准用法
* 引入deleter后unique_ptr的size变化：如果是函数指针，1 word -> 2 word；如果是函数对象，取决于函数内部的state



##### Item 19: Use std::shared_ptr for shared-ownership resource management.

dynamically allocated control blocks, arbitrarily large deleters and allocators, virtual function machinery, and atomic reference count manipulations

[shared_ptr的使用](https://blog.csdn.net/qq_33266987/article/details/78784852), [一篇看不太懂的shared_ptr使用技巧](https://gist.github.com/BruceChen7/8cccb33ea6fbc73a0651c4bce3166806)

As with garbage collection, clients need not concern themselves with managing the life‐ time of pointed-to objects, but as with destructors, the timing of the objects’ destruc‐ tion is deterministic.

reference count的影响：

* std::shared_ptrs are twice the size of a raw pointer
* Memory for the reference count must be dynamically allocated，除非用`std::make_shared`
* Increments and decrements of the reference count must be atomic

=> move assignment is faster than copy assignment

![shared-ptr](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/C++/shared-ptr.png)

对于custom deleter，和unique_ptr的区别在于deleter不影响shared_ptr的类型

* 这样便于函数传参、放入同一容器
* 始终two words size：

```c++
auto loggingDel = [](Widget *pw){
                    makeLogEntry(pw);
										delete pw; 
									};
std::unique_ptr<Widget, decltype(loggingDel) > upw(new Widget, loggingDel);
std::shared_ptr<Widget> spw(new Widget, loggingDel);
// deleter type is not part of ptr type
```

关于control block的rules

* std::make_shared (see Item 21) always creates a control block
* A control block is created when a std::shared_ptr is constructed from a unique-ownership pointer (i.e., a std::unique_ptr or std::auto_ptr).
* When a std::shared_ptr constructor is called with a raw pointer, it creates a control block

=> 

* 用同一个裸指针创建两个shared_ptr是undefined behaviour
  * 正确写法：

```c++
std::shared_ptr<Widget> spw1(new Widget, // direct use of new loggingDel);
std::shared_ptr<Widget> spw2(spw1);
```

* 类似地，不要用this指针创建shared_ptr

  * 正确写法：*The Curiously Recurring Template Pattern (CRTP)*
  * 为了防止process一个不存在shared_ptr的对象，常把ctors设成private

```c++
class Widget: public std::enable_shared_from_this<Widget> {
public:
  // factory function that perfect-forwards args to a private ctor
	template<typename... Ts>
	static std::shared_ptr<Widget> create(Ts&&... params);
	void process(){
    processedWidgets.emplace_back(shared_from_this());
  }
private:
  Widget Widget();
  Widget(Widget &&other);
  Widget(const Widget &other);
  TCPConnection() = delete;
};
```



**Warn:**

* `std::vector<std::shared_ptr<A> > a_vec(n, std::make_shared<A>());`这个用法是错的，会给里面元素赋值同一个智能指针，正确写法如下：

```c++
std::vector<std::shared_ptr<A>> a(size);
std::for_each(std::begin(a), std::end(a),[](std::shared_ptr<A> &ptr) {
                    ptr = std::make_shared<A>();
```

* shared_ptr没有array的版本。一律建议用std::vector
* shared_ptr不能循环指向，会永不析构内存泄漏



##### Item 20: Use std::weak_ptr for std::shared_ptr-like pointers that can dangle.

weak_ptr指示shared_ptr是否dangle，但不能直接dereference，需要一个原子操作(lock)，衔接判断和取值

* Potential use cases for `std::weak_ptr` include caching, observer lists, and the prevention of std::shared_ptr cycles.

```c++
auto spw = std::make_shared<Widget>();
std::weak_ptr<Widget> wpw(spw);
spw = nullptr;
if (wpw.expired()) ...
auto spw2 = wpw.lock();
std::shared_ptr<Widget> spw3(wpw); // if wpw's expired, throw std::bad_weak_ptr
```

应用：读写cache，自动删去不再使用的对象

```c++
std::unique_ptr<const Widget> loadWidget(WidgetID id);
```

=>

```c++
std::shared_ptr<const Widget> fastLoadWidget(WidgetID id) {
static std::unordered_map<WidgetID, std::weak_ptr<const Widget>> cache;
auto objPtr = cache[id].lock();
  if (!objPtr) {
    objPtr = loadWidget(id);
    cache[id] = objPtr;
}
  return objPtr;
}

//优化空间：删除cache内不用的weak_ptr
```



##### Item 21: Prefer **std::make_unique** and **std::make_shared** to direct use of **new**.

C++14没有`make_unique`

```c++
template<typename T, typename... Ts> std::unique_ptr<T> make_unique(Ts&&... params) {
return std::unique_ptr<T>(new T(std::forward<Ts>(params)...)); }
```

```c++
std::allocate_shared
auto spw1(std::make_shared<Widget>());   //能用auto
```



`make_shared`优点

* 不用的话，有潜在的内存泄漏风险，和异常安全性有关
  * 先new再创建shared_ptr，如果在这中间computePriority抛出异常，则内存泄漏

```c++
processWidget(std::shared_ptr<Widget>(new Widget), computePriority());
```

* `make_shared`更高效，一次分配对象和control block的内存

`make_shared`的缺点

* deleter
* `make_shared`的perfect forwarding code用`()`，而非`{}`; `{}`只能用new，除非参考Item 2
```c++
// create std::initializer_list
auto initList = { 10, 20 };
// create std::vector using std::initializer_list ctor
auto spv = std::make_shared<std::vector<int>>(initList);
```

* using make functions to create objects of types with class-specific versions of operator new and operator delete is typically a poor idea. 因为只能new/delete本对象长度的内存，而非加上control block的
* shared_ptr的场景，必须要所有相关weak_ptr全destroy，这块内存才会释放

因为上述原因，想用new
```c++
std::shared_ptr<Widget> spw(new Widget, cusDel);
processWidget(std::move(spw), computePriority()); // both efficient and exception safe
```

##### Item 22: When using the Pimpl Idiom, define special member functions in the implementation file.

设计模式：减少client（只include .h文件）的相关编译

`widget.h`

```c++
class Widget {
public:
	Widget(); 
	~Widget(); 
	...
private:
  struct Impl;
  Impl *pImpl;
};
```

`widget.cpp`

```c++
#include "widget.h" 
#include "gadget.h" 
#include <string> 
#include <vector>
struct Widget::Impl {
  std::string name;
  std::vector<double> data;
  Gadget g1, g2, g3;
};
Widget::Widget() : pImpl(new Impl) {}
Widget::~Widget() { delete pImpl; }
```

**新的用`unique_ptr`的版本**

* 不要在.h文件里generate析构函数

`widget.h`

```c++
class Widget {
public:
	Widget(); 
  ~Widget();
  Widget(Widget&& rhs);
  Widget& operator=(Widget&& rhs);
private:
  struct Impl;
  std::unique_ptr<Impl> pImpl;
};
```

`widget.cpp`

```c++
#include "widget.h" 
#include "gadget.h" 
#include <string> 
#include <vector>
struct Widget::Impl {
  std::string name;
  std::vector<double> data;
  Gadget g1, g2, g3;
};
Widget::Widget(): pImpl(std::make_unique<Impl>()) {}
Widget::~Widget() = default;
Widget::Widget(Widget&& rhs) = default;
Widget& Widget::operator=(Widget&& rhs) = default;
Widget::Widget(const Widget& rhs)
: pImpl(std::make_unique<Impl>(*rhs.pImpl)) {}
Widget& Widget::operator=(const Widget& rhs)
{
	*pImpl = *rhs.pImpl;
  return *this;
}
```

如果是shared_ptr，则不需要做任何操作，本质上是因为unique_ptr的deleter在类型定义中，为了更高效的runtime行为，隐性地generate一些function





#### chpt 5 Rvalue References, Move Semantics, and Perfect Forwarding

Rvalue references => Move Semantics and Perfect Forwarding

it’s especially important to bear in mind that a parameter is always an lvalue, even if its type is an rvalue reference

##### Item 23: Understand **std::move** and **std::forward**.

`std::move` doesn’t move anything. `std::forward` doesn’t forward anything

`std::move` unconditionally casts its argument to an rvalue, while `std::forward` performs this cast only if a particular condition is fulfilled

```c++
template<typename T> 
decltype(auto) move(T&& param) {
	using ReturnType = remove_reference_t<T>&&;
	return static_cast<ReturnType>(param); 
} // C++14
// remove_reference是因为Item 1 => T有可能被推断为lvalue reference
```

First, don’t declare objects const if you want to be able to move from them. Move requests on const objects are silently transformed into copy operations. 

Second, std::move not only doesn’t actually move anything, it doesn’t even guarantee that the object it’s casting will be eligible to be moved. The only thing you know for sure about the result of applying std::move to an object is that it’s an rvalue.

```c++
void process(const Widget& lvalArg);
void process(Widget&& rvalArg);
template<typename T>
void logAndProcess(T&& param) {
	auto now = std::chrono::system_clock::now();
  makeLogEntry("Calling 'process'", now);
	process(std::forward<T>(param));
}

Widget w;
logAndProcess(w);                  // call with lvalue
logAndProcess(std::move(w));       // call with rvalue
```

`std::forward`: a *conditional* cast, it casts to an rvalue only if its argument was initialized with an rvalue. 原理参考Item 28



##### Item 24: Distinguish universal references from rvalue references.

T&&不一定是rvalue references，称其为universal references，场景如下：

* 模板: 

```c++
template<typename T>
void f(T&& param); // param is a universal reference
```

* auto: `auto&& var2 = var1; // var2 is a universal reference`

universal reference的场景有限制：

* 必须是T&&，加const也不行
* 必须有type deduction，e.g. `push_back` 's caller explicitly specifies the type，type deduction，而`emplace_back`有

```c++
template<class T, class Allocator = allocator<T>>  // from C++
class vector {                                     // Standards
public:
	void push_back(T&& x);
	...
};
template<class T, class Allocator = allocator<T>>  
class vector {
public:
	template <class... Args>
	void emplace_back(Args&&... args); ...
};
```

universal reference的应用：

```c++
auto timeFuncInvocation = [](auto&& func, auto&&... params){
  start timer;
  std::forward<decltype(func)>(func)( std::forward<decltype(params)>(params)... );
  stop timer and record elapsed time;
};
```



##### Item 25: Use **std::move** on rvalue references, **std::forward** on universal references.





### C++

#### Debug

* 参考我的[Debugging and Profiling笔记]()

#### 编译相关

不同操作系统的编译:

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

  * LD_PRELOAD的应用
      * when two libraries export the same symbol and our program links with the wrong one
      * when an optimized or custom implementation of a library function should be preferred
      * various profiling and monitoring tools widely use LD_PRELOAD for instrumenting code

```shell
LD_PRELOAD="/data/preload/lib/malloc_interpose.so:/data/preload/lib/free_interpose.so" ls -lh
```




#### C

STDERR_FILENO: [文件描述符的概念](http://guoshaoguang.com/blog/tag/stderr_fileno/)

volatile特性：这条代码不要被编译器优化，`volatile int counter = 0`，避免出现意料之外的情况，常用于全局变量

C不能用for(int i=0; ; ) 需要先定义i

负数取余是负数，如果想取模：a = (b%m+m)%m;

Dijkstra反对goto的[文章](http://www.cs.utexas.edu/users/EWD/ewd02xx/EWD215.PDF)

getopt函数处理参数，用法参照[tsh.c](https://github.com/huangrt01/CSAPP-Labs/blob/master/shlab-handout/tsh.c)

##### 结构体

结构体内存分配问题：内存对齐
* 起始地址为该变量的类型所占的整数倍，若不足则不足部分用数据填充至所占内存的整数倍。
* 该结构体所占内存为结构体成员变量中最大数据类型的整数倍。
* e.g.: 1+4+1+8->4+4+8+8=24

#### C++的特性
面向对象、构造函数、析构函数、动态绑定、内存管理

* 从`int *p=(int *)malloc(sizeof(int));`到`int *p=new int[10]`
* 二维数组的定义：`TYPE(*p)[N] = new TYPE[][N];`，要指出行数
* 复制构造函数：常引用
* 析构函数
  * 对象的destructor不被call的情形：Most stem from abnormal program termination. If an exception propagates out of a thread’s primary function (e.g., main, for the program’s initial thread) or if a noexcept specifi‐ cation is violated (see Item 14), local objects may not be destroyed, and if `std::abort` or an exit function (i.e., `std::_Exit`, `std::exit`, or `std::quick_exit`) is called, they definitely won’t be.
* 虚析构函数    =>对象内有虚函数表，指向虚函数表的指针：32位系统4字节，64位系统8字节
* 虚基类偏移量表指针

##### namespace

https://www.runoob.com/cplusplus/cpp-namespaces.html

```c++
namespace XXX{
	//
}
using namespace XXX;
using XXX::object;

//可嵌套
using namespace namespace_name1::namespace_name2;
```





#### C++11
`void DUMMY_CODE(Targs &&... /* unused */) {}`的[应用](https://blog.csdn.net/xs18952904/article/details/85221921)

`std::optional<>`

* 方法：has_value(), value(), value_or(XX)

#### 操作符重载
**特点**
* 既不能改变原运算符的运算优先级和结合性，也不能改变操作数的个数
*“.”、“.*”(成员指针运算符)、“::”（作用域分辨符）、“ ? : ”（三目运算符） 、sizeof 以及typeid这6个操作符不能重载。否则会带来难以琢磨的问题
* 和类的概念紧密联系，操作数中至少有一个是自定义的类类型。 这一点也和编译原理有联系

**成员函数和友元函数**

* 非成员函数不一定要作为类的友元，如果只通过外部接口访问类，当类的内部结构改变时，不用改操作符
```c++
inline WrappingInt32 operator++(WrappingInt32 &a, int) {
    uint32_t r = a.raw_value();
    a = a + 1;
    return WrappingInt32(r);
}
```

* `=, [], (), ->`只能重载为成员函数
* 友元函数实现
```c++
#include <iostream>
using namespace std;
class pwr {
public: 
	pwr(int i) { num = i; }
	friend int operator ^(pwr, pwr);
private: int num;
};
int operator ^(pwr b, pwr e)
{
	int t, temp;
	temp = b.num;
	for (t = e.num - 1; t; t--)
		b.num *= temp;
	return b.num;
}
```

* 二目操作符的成员函数：参数常引用
* 二目操作符的友员函数
* 单目操作符的成员函数
  * 前置重载`++a`视为无形参的成员函数，后置`a++`具有一个int类型形参
  * 前置重载`++a`返回值带引用，而后置重载返回值不带引用
* 单目操作符的友元函数：见上面`WrappingInt32`的例子

**其它操作符重载**

1.string类的赋值运算符函数
* 经典解法：考虑[返回引用](https://bbs.csdn.net/topics/100000589?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task)、连续赋值、等号两边相同的情形
```c++
CMyString& CMyString::operator=(const CMyString &str){
	if(this==&str)
		return *this;
	delete []m_pData;
	m_pData = new char[strlen(str.m_pData)+1];
	strcpy(m_pData, str.m_pData);
	return *this;
}
```
* 考虑异常安全性：上面的解法在new分配之前先delete，违背了Exception Safety原则，我们需要保证分配内存失败时原先的实例不会被修改，因此可以先复制，或者创造临时实例。(临时实例利用了if语句，在if的大括号外会自动析构)
```c++
CMyString& CMyString::operator=(const CMyString &str){
	if(this!=&str){
		CMyString strTemp(str);
		swap(m_pData,strTemp.m_pData);
	}
	return this;
}
```

2.<< 操作符的重载
* `ostream &operator << (ostream &output, 自定义类型&); //istream也可`
* 只能重载为类的友元函数（否则必须修改系统的ostream类，不被允许） 
* 为了级联输出，重载函数的形参和返回值必须是I / O对象的引用
* 对于 `cout << c1 << c2 << c3;`语句，编译器需要一次性生成可执行代码，因为C++是编译型语言，不是BASIC那种边编译边执行的解释型语言。为了按照c1，c2，c3的顺序输出，就需要把对象按照c3，c2，c1的次序依此入栈，便于执行时的边出栈边打印，因此一次性传递完参数然后再从左向右执行。

3.类型操作符的重载强制类型转换

```c++
operator 类型名()
{
	return 与类型名类型相同的转换结果;
}
```

* 此函数前没有返回类型，函数也没有参数。函数将会自动返回与类型转换函数类型名相同的类型。
* 这里的类型名是基本数据类型int，double等。
* 类型转换函数只能作为相应类的成员函数。
* 总的来看，类中的类型转换函数优先级很高，导致类的对象在运算时，不按常规方式操作，而是先进行类型强制转换，导致结果不可琢磨，因此类型转换函数要慎用！

#### 继承与派生

[Access Modifiers in C++](https://www.geeksforgeeks.org/access-modifiers-in-c/)

* protected: 类似private，但可以被派生类调用



#### C++的其它可扩展性

* [类的成员指针函数](https://www.cnblogs.com/zhoug2020/p/11394408.html)
```c++
class Solution;
typedef bool (Solution::*FP_func)(int n);
class Solution {
public:
    FP_func m_pfn_func;
    bool isEven(int n){
        return (n&1)==0;
    }
    vector<int> exchange(vector<int>& nums) {
        m_pfn_func=&Solution::isEven;
        if(nums.empty())return nums;
        int i=-1,j=nums.size();
        while(1){
            while(!(this->*m_pfn_func(nums[++i]))if(i==nums.size()-1)break;
            while((this->*m_pfn_func)(nums[--j])) if(j==0)break;
            if(i>=j)break;
            swap(nums[i],nums[j]);
        }
        return nums;
    }
};
```
* 类的静态成员函数指针

#### 编程习惯
RAII原则：Resource acquisition is initialization
* [Google Style](https://google.github.io/styleguide/cppguide.html)
* [Google: Developer Documentation Style Guide](https://developers.google.com/style)
* [CppCoreGuidelines](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md)
* 用[CppCheck](http://cppcheck.net/)诊断，`make cppcheck`

format

* `/llvm-6.0/bin/clang-format -i *.cpp`



#### 输入输出

##### 文件

```c++
static std::string read_file(const std::string& file_name) {
  std::string str = "";
  std::ifstream infile;
  infile.open(file_name);
  infile.seekg(0, std::ios::end);
  str.reserve(infile.tellg());
  infile.seekg(0, std::ios::beg);

  str.assign((std::istreambuf_iterator<char>(infile)),
              std::istreambuf_iterator<char>());
  infile.close();
  return str;
}
```



##### 输入用逗号间隔的数据

* 方法一：[混合使用cin>>和cin.get()](http://c.biancheng.net/view/1346.html)
* 方法二：利用string和strtok
```c++
class cplus_input{
    public:
        vector<int> num;
        void input_comma(){
            int a;
            while (cin >> a)
            {
                num.push_back(a);
                if (cin.get() == '\n')
                    break;
            }

            // string s;
            // cin >> s;
            // char *str = (char *)s.c_str(); //string --> char
            // const char *split = ",";
            // char *p = strtok(str, split); //逗号分隔依次取出
            
            // while (p != NULL)
            // {
            //     sscanf(p, "%d", &a); //char ---> int
            //     num.push_back(a);
            //     p = strtok(NULL, split); //s为空值NULL，则函数保存的指针SAVE_PTR在下一次调用中将作为起始位置。
            // }
            return;
        }
};


```



LOG代码

* `#、__FILE__、__LINE__` 的运用

```c++
#define CHECK(x)                                                \
  if (!(x))                                                       \
  LogMessageFatal(__FILE__, __LINE__).stream() \
      << "Check failed: " #x << ": "

namespace euclid {
namespace rosetta {

class LogMessageFatal {
 public:
  LogMessageFatal(const char* file, int line) {
    log_stream_ << file << ":" << line << ": ";
  }
  std::ostringstream& stream() {
    return log_stream_;
  }
  ~LogMessageFatal() noexcept(false) {
    throw std::runtime_error(log_stream_.str());
  }

 private:
  std::ostringstream log_stream_;
};
```

* `__VA_ARGS__` 的[应用](https://stackoverflow.com/questions/26053959/what-does-va-args-in-a-macro-mean)



#### multi-thread programming

* 读写锁

```c++
#include <boost/thread/thread.hpp>
#include <shared_mutex>

//读锁
//灵活使用：用{}包起来，控制释放锁的时机
{
	std::shared_lock<boost::shared_mutex> lock(filter_mutex_);
}

//写锁
std::unique_lock<boost::shared_mutex> lock(filter_mutex_);
```

* 大量读，少量更新，可以用tbb::concurrent_hash_map<key_type, value_type>;

```c++
{
	typename map_type::const_accessor cit;
	bool found = _map.find(cit, key);
}
{
  typename map_type::accessor it;
  _map.insert(it, key);
}

using queue_type = tbb::concurrent_bounded_queue<timed_key>;
_queue.try_push(tk);
while(_queue.try_pop(tk)){
  ...
}
```



```c++
std::mutex mutex_;
std::condition_variable cond_;

cond_.notify_one();
cond_.notify_all();
```



* Shared Store

参考 `shared_store.h`



#### STL

十三大头文件：\<algorithm>、\<functional>、\<deque>、、\<iterator>、\<array>、\<vector>、\<list>、\<forward_list>、\<map>、\<unordered_map>、\<memory>、\<numeric>、\<queue>、\<set>、\<unordered_set>、\<stack>、\<utility>

##### \<algorithm>
sort，自己定义cmp函数，注意cmp的定义：类内静态，传参引用
* `static bool cmp1(vector<int> &a, vector<int> &b)`


##### \<deque>
* deque，两端都能进出，双向队列，[用法详解](https://blog.csdn.net/u011630575/article/details/79923132)
* [STL之deque实现详解]( https://blog.csdn.net/u010710458/article/details/79540505?depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-6&utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-6)
* deque的pop_front相当于queue的pop()

##### \<list>
* 参考[LRU cache](https://leetcode-cn.com/problems/lru-cache/)，类似双向链表的实现
  * map<int,list<pair<int,int>>::iterator> m;
* r.push_front(…), r.begin(), r.back()

##### \<vector>
* 初始化，可以用列表
  
  * 也可以 `int a[10]={…};vector<int> b(a,a+10); `       左闭右开
* [动态扩容](https://www.cnblogs.com/zxiner/p/7197327.html)，容量翻倍，可以用reserve()预留容量
* 方法：
  * reverse(nums.begin(),nums.end());
  * reserve(size_type n) 预先分配内存
* [关于vector的内存释放问题](https://www.cnblogs.com/jiayouwyhit/p/3878047.html)
  * 方法一：clear 
  * 方法二：`vector<int>().swap(nums);`
  * 方法三：利用代码块和临时变量
`
{
    vector<int> tmp = curLevel;   
    curLevel.swap(tmp); 
}
`
* clear虽然不会deallocate释放空间，但是会destroy执行析构函数，所以可以用同一个空间构造节点，如果swap了就要重新分配空间再构造节点。因此对于同一个vector的重复利用，可以直接用clear();
  * 如果要每次都释放空间，也可以用`res.emplace_back(std::move(curLevel))`，涉及[emplace_back](https://www.cnblogs.com/ChrisCoder/p/9919646.html), [std::move](https://blog.csdn.net/p942005405/article/details/84644069/), [左值、右值引用](https://blog.csdn.net/p942005405/article/details/84644101), [这是一篇有关类定义的总结](https://blog.csdn.net/zzhongcy/article/details/86747794)
* [无法构造有atomic类型成员的对象的vector](https://stackoverflow.com/questions/13193484/how-to-declare-a-vector-of-atomic-in-c)




#### 其它的库

##### \<exception>
https://blog.csdn.net/qq_37968132/article/details/82431775

`throw invalid_argument("Invalid input.");`

##### \<pthread.h>
* 注意使用封装好的线程操作接口

##### \<string>
* [和字符数组的相互转换](https://www.cnblogs.com/fnlingnzb-learner/p/6369234.html)
  * 字符数组->str：构造函数
  * str->字符数组
```c++
char buf[10];
string str("ABCDEFG");
length = str.copy(buf, 9);
buf[length] = '\0';

char buf[10];
string str("ABCDEFG");
strcpy(buf, str.c_str());//strncpy(buf, str.c_str(), 10);
```

* [用find查找字串](https://www.cnblogs.com/wkfvawl/p/9429128.html)
  * 截取子串
    * s.substr(pos, n) 截取s中从pos开始（包括0）的n个字符的子串，并返回
    * s.substr(pos) 截取s中从从pos开始（包括0）到末尾的所有字符的子串，并返回
  * 替换子串
    * s.replace(pos, n, s1)    用s1替换s中从pos开始（包括0）的n个字符的子串
  * 查找子串
    * s.find(s1) 查找s中第一次出现s1的位置，并返回（包括0）
    * s.rfind(s1) 查找s中最后次出现s1的位置，并返回（包括0）
    * s.find_first_of(s1) 查找在s1中任意一个字符在s中第一次出现的位置，并返回（包括0）
    * s.find_last_of(s1) 查找在s1中任意一个字符在s中最后一次出现的位置，并返回（包括0）
    * s.fin_first_not_of(s1) 查找s中第一个不属于s1中的字符的位置，并返回（包括0）
    * s.fin_last_not_of(s1) 查找s中最后一个不属于s1中的字符的位置，并返回（包括0）

```c++
# Interprets an unsigned integer value in the string str.
size_t idx = 0;
unsigned long ul = std::stoul(str,&id,0);
```

##### \<sys.h>

sche.h

`pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &m->set)`
* [CPU亲和性](https://blog.csdn.net/ma950924/article/details/81773719)

```c++
#ifndef __common_h__
#define __common_h__

#include <sys/time.h>
#include <sys/stat.h>
#include <assert.h>

double GetTime() {
    struct timeval t;
    int rc = gettimeofday(&t, NULL);
    assert(rc == 0);
    return (double) t.tv_sec + (double) t.tv_usec/1e6;
}

void Spin(int howlong) {
    double t = GetTime();
    while ((GetTime() - t) < (double) howlong)
    ; // do nothing in loop
}

struct timeval start, end;
gettimeofday(&start, NULL);
gettimeofday(&end, NULL);
printf("Time (seconds): %f\n\n", (float) (end.tv_usec - start.tv_usec + (end.tv_sec - start.tv_sec) * 1000000) / 1000000);


#endif // __common_h__
```



```c++
static thread_local std::mt19937 rng(std::random_device{}());
std::uniform_int_distribution<int> distribution(0, RAND_MAX-1);
int32_t sleep_time = static_cast<int32_t>(
        _expire_delta * ((double) distribution(rng)/ (RAND_MAX)) * _expire_num);
    std::this_thread::sleep_for(std::chrono::seconds(sleep_time));
```








