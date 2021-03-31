// Copy a range of elements
std::vector<int> to_vector;
std::copy(from_vector.begin(), from_vector.end(),
          std::back_inserter(to_vector));
std::cout << "to_vector contains: ";
std::copy(to_vector.begin(), to_vector.end(),
          std::ostream_iterator<int>(std::cout, " "));

// Back inserter
std::fill_n(std::back_inserter(v), 3, -1);

// Count occurrences of value in a range
std::vector<int> numbers = {1, 2, 3, 5, 6, 3, 4, 1};
int count = std::count(std::begin(numbers),
                       std::end(numbers),
                       3);

// Sort a range of elements
std::array<int, 5> arr = {3, 4, 1, 5, 2};
std::sort(std::begin(arr), std::end(arr));
std::sort(std::begin(arr), std::end(arr),
        std::greater<int>{});

// Swap containers
std::vector<int> v1 = { 1, 3, 5, 7 };
std::vector<int> v2 = { 2, 4, 6, 8 };
v1.swap(v2);
using std::swap;
swap(v1, v2);
// element by element swap of values
std::swap_ranges(std::begin(v1), std::end(v1), std::begin(v2));

// ADL: https://en.cppreference.com/w/cpp/language/adl
// "using std::swap" allows a user-defined specialization of swap 
// to be found via argument-dependent lookup (ADL), which may 
// provide a more efficient implementation of the swap operation, 
// before falling back to the generic std::swap function. This 
// approach is particularly useful when swapping two generic objects 
// (such as in a template).


// Copy-and-swap idiom
#include <utility>
class resource {
  int x = 0;
};

class foo
{
  public:
    foo()
      : p{new resource{}}
    { }

    foo(const foo& other)
      : p{new resource{*(other.p)}}
    { }

    foo(foo&& other)
      : p{other.p}
    {
      other.p = nullptr;
    }

    foo& operator=(foo other)
    {
      swap(*this, other);
      return *this;
    }

    ~foo()
    {
      delete p;
    }

    friend void swap(foo& first, foo& second)
    {
      using std::swap;
      swap(first.p, second.p);
    }

  private:
    resource* p;
};
// The copy-and-swap idiom has inherent strong exception safety because all allocations 
// (if any) occur when copying into the other argument, before any changes have been made to *this.
// It is generally, however, less optimized than a more custom implementation of the assignment operators.


// Delegate behavior to derived classes
template<typename derived>
class base
{
  public:
    void do_something()
    {
      // ...
      static_cast<derived*>(this)->do_something_impl();
      // ...
    }
  private:
    void do_something_impl()
    {
      // Default implementation
    }
};
class foo : public base<foo>
{
  public:
    void do_something_impl()
    {
      // Derived implementation
    }
};
class bar : public base<bar>
{ };

template<typename derived>
void use(base<derived>& b)
{
// 能识别出foo::do_something_impl() 
  b.do_something();
}
// Curiously Recurring Template Pattern (CRTP)
// https://en.wikipedia.org/wiki/Virtual_method_table


// Lexicographic ordering
struct S {
    int n;
    std::string s;
    float d;
    bool operator<(const S& rhs) const
    {
        // compares n to rhs.n,
        // then s to rhs.s,
        // then d to rhs.d
        return std::tie(n, s, d) < std::tie(rhs.n, rhs.s, rhs.d);
    }
};

int main()
{
    std::set<S> set_of_s; // S is LessThanComparable
 
    S value{42, "Test", 3.14};
    std::set<S>::iterator iter;
    bool inserted;
 
    // unpacks the return value of insert into iter and inserted
    std::tie(iter, inserted) = set_of_s.insert(value);
 
    if (inserted){
        std::cout << "Value was inserted successfully\n";
    }
}

// Non-member non-friend interfaces
// ADL->find non_member() in namespace ns
namespace ns
{
  class foo
  {
    public:
      void member()
      {
        // Uses private data
      }
    private:
      // Private data
  };
  void non_member(foo obj)
  {
    obj.member();
  }
}
int main()
{
  ns::foo obj;
  non_member(obj);
}


// The PIMPL idiom
// 嵌套类：https://blog.csdn.net/Poo_Chai/article/details/91596538
// 嵌套类的其它应用：Builder Pattern
// foo.h - header file
#include <memory>
class foo
{
  public:
    foo();
    ~foo();
    foo(foo&&);
    foo& operator=(foo&&);
  private:
    class impl;
    std::unique_ptr<impl> pimpl;
};
// foo.cpp - implementation file
class foo::impl
{
  public:
    void do_internal_work()
    {
      internal_data = 5;
    }
  private:
    int internal_data = 0;
};
foo::foo()
  : pimpl{std::make_unique<impl>()}
{
  pimpl->do_internal_work();
}
foo::~foo() = default;
foo::foo(foo&&) = default;
foo& foo::operator=(foo&&) = default;


// The rule of five
#include <utility>
class resource {
  int x = 0;
};

class foo
{
  public:
    foo()
      : p{new resource{}}
    { }
    foo(const foo& other)
      : p{new resource{*(other.p)}}
    { }
    foo(foo&& other)
      : p{other.p}
    {
      other.p = nullptr;
    }
    foo& operator=(const foo& other)
    {
      if (&other != this) {
        delete p;
        p = nullptr;
        p = new resource{*(other.p)};
      }
      return *this;
    }
    foo& operator=(foo&& other)
    {
      if (&other != this) {
        delete p;
        p = other.p;
        other.p = nullptr;
      }
      return *this;
    }
    ~foo()
    {
      delete p;
    }
  private:
    resource* p;
};

// 另一种the-copy-and-swap idiom写法
foo& operator=(const foo& other)
{
  if (&other != this) {
  	foo temp{other};
  	swap(p, temp.p)
  }
  return *this;
}

// The rule of zero
// foo support copy and move semantics, bar support move semantics
class foo
{
  private:
    int x = 10;
    std::vector<int> v = {1, 2, 3, 4, 5};
};
class bar
{
  public:
    std::unique_ptr<int> p = std::make_unique<int>(5);
};

// Virtual Constructors
// Create a copy of an object through a pointer to its base type.
class Base
{
public:
  virtual ~Base() {}
  virtual Base* clone() const = 0;
};
class Derived : public Base
{
public:
  Derived* clone() const override
  {
    return new Derived(*this);
  }
};
void foo(std::unique_ptr<Base> original)
{
  std::unique_ptr<Base> copy{original->clone()};
}
// NOTE: clone()返回智能指针
// https://stackoverflow.com/questions/6924754/return-type-covariance-with-smart-pointers/6925201#6925201


// Create a thread
#include <thread>
#include <string>
#include <functional>
void func(std::string str, int& x);
void do_something();
int main()
{
  std::string str = "Test";
  int x = 5;
  std::thread t{func, str, std::ref(x)};
  do_something();
  t.join();
}

// ASYNC: 接口都有，性能很差，用folly
#include <future>
int func()
{
  int some_value = 0;
  // Do work...
  return some_value;
}
int main()
{
  std::future<int> result_future = std::async(func);
  // Do something...
  int result = result_future.get();
}
// std::async的第一个参数，用std::launch::async/std::launch::deferred决定是否执行方式


// Pass values between thread
void func(std::promise<int> result_promise) noexcept
{
  result_promise.set_value(42);
}
int main()
{
  std::promise<int> result_promise;
  std::future<int> result_future = result_promise.get_future();
  std::thread t{func, std::move(result_promise)};
  int result = result_future.get();
  t.join();
}

// Check existence of a key
std::map<std::string, int> m = {{"a", 1}, {"b", 2}, {"c", 3}};
if (m.count("b"))
{
// We know "b" is in m
}

// Remove elements from a container
int main()
{
  std::vector<int> v = {1, 2, 3, 4, 2, 5, 2, 6};
  v.erase(std::remove(std::begin(v), std::end(v), 2),
          std::end(v));
  v.erase(std::remove_if(std::begin(v), std::end(v),
                         [](int i) { return i%2 == 0; }),
          std::end(v));
}

// Remove duplicate elements
std::sort(v.begin(), v.end());
v.erase(unique(v.begin(), v.end()), v.end());


// Apply tuple to a function
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>
template<typename F, typename Tuple, size_t ...S >
decltype(auto) apply_tuple_impl(F&& fn, Tuple&& t, std::index_sequence<S...>)
{
  return std::forward<F>(fn)(std::get<S>(std::forward<Tuple>(t))...);
}

template<typename F, typename Tuple>
decltype(auto) apply_from_tuple(F&& fn, Tuple&& t)
{
  std::size_t constexpr tSize
    = std::tuple_size<typename std::remove_reference<Tuple>::type>::value;
  return apply_tuple_impl(std::forward<F>(fn),
                          std::forward<Tuple>(t),
                          std::make_index_sequence<tSize>());
}

int do_sum(int a, int b)
{
  return a + b;
}

int main()
{
  int sum = apply_from_tuple(do_sum, std::make_tuple(10, 20));
}

// std::optional 
// https://en.cppreference.com/w/cpp/utility/optional


// Return multiple values
// C++17
#include <tuple>
std::tuple<int, bool, float> foo()
{
  return {128, true, 1.5f};
}
int main()
{
  std::tuple<int, bool, float> result = foo();
  int value = std::get<0>(result);
  auto [value1, value2, value3] = foo();
}
// C++11
std::tuple<int, bool, float> foo()
{
  return std::make_tuple(128, true, 1.5f);
}
int main()
{
  std::tuple<int, bool, float> result = foo();
  int value = std::get<0>(result);
  int obj1;
  bool obj2;
  float obj3;
  std::tie(obj1, obj2, obj3) = foo();
}


// sstream
#include <sstream>
#include <string>
int main()
{
  std::istringstream stream{"This stream\n"
                            "contains many\n"
                            "lines.\n"};
  std::string line;
  while (std::getline(stream, line)) {
    // Process line
  }

  std::istringstream stream{"4 36 72 8"};
  std::vector<int> values;
  std::copy(std::istream_iterator<int>{stream},
            std::istream_iterator<int>{},
            std::back_inserter(values));
}

// 这样写效率很低，不知道为啥。。。
static inline std::string sort_and_concat_strings(
    std::vector<std::string>& str_list, const char* delimiter = "") {
  std::ostringstream oss;
  std::sort(str_list.begin(), str_list.end());
  std::copy(str_list.begin(), str_list.end(),
            std::ostream_iterator<std::string>(oss, delimiter));
  return oss.str();
}

// Validate multiple reads
int main()
{
  std::istringstream stream{"Chief Executive Officer\n"
                            "John Smith\n"
                            "32"};
  std::string position;
  std::string first_name;
  std::string family_name;
  int age;
  if (std::getline(stream, position) &&
      stream >> first_name >> family_name >> age) {
    // Use values
  }
}

// Weak reference
class bar;
class foo
{
public:
  foo(const std::shared_ptr<bar>& b)
    : forward_reference{b}
  { }
private:
  std::shared_ptr<bar> forward_reference;
};

class bar
{
public:
  void set_back_reference(const std::weak_ptr<foo>& f)
  {
    this->back_reference = f;
  }
  void do_something()
  {
    std::shared_ptr<foo> shared_back_reference = this->back_reference.lock();
    if (shared_back_reference) {
      // Use *shared_back_reference
    }
  }
private:
  std::weak_ptr<foo> back_reference;
};

// Overload operator<<
class foo
{
  public:
    friend std::ostream& operator<<(std::ostream& stream,
                                    foo const& f);
  private:
    int x = 10;
};
std::ostream& operator<<(std::ostream& stream,
                         foo const& f)
{
  return stream << "A foo with x = " << f.x;
}

// Write data in columns
#include <iostream>
#include <iomanip>
int main()
{
  std::cout << std::left << std::setw(12) << "John Smith"
            << std::right << std::setw(3) << 23
            << '\n';
  std::cout << std::left << std::setw(12) << "Sam Brown"
            << std::right << std::setw(3) << 8
            << '\n';
}

// Range-based algorithms
template <typename ForwardRange>
void algorithm(ForwardRange& range)
{
  using std::begin;
  using std::end;
  using iterator = decltype(begin(range));
  iterator it_begin = begin(range);
  iterator it_end = end(range);
  // Now use it_begin and it_end to implement algorithm
}

// Class template SFINAE
// SFINAE(Substitution Failure Is Not An Error)
// https://en.wikipedia.org/wiki/Substitution_failure_is_not_an_error
#include <type_traits>
template <typename T, typename Enable = void>
class foo;

template <typename T>
class foo<T, typename std::enable_if<std::is_integral<T>::value>::type>
{ };

template <typename T>
class foo<T, typename std::enable_if<std::is_floating_point<T>::value>::type>
{ };


// Function template SFINAE
#include <type_traits>
#include <limits>
#include <cmath>
template <typename T>
typename std::enable_if<std::is_integral<T>::value, bool>::type
  equal(T lhs, T rhs)
{
  return lhs == rhs;
}
template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, bool>::type
  equal(T lhs, T rhs)
{
  return std::abs(lhs - rhs) < 0.0001;
}


// Observer
// Notify generic observer objects when an event occurs.
// We use std::reference_wrapper for the elements of the std::vector (line 33), 
// because the standard containers require the element type to be assignable, which normal reference types are not.
#include <functional>
class observer
{
  public:
    virtual void notify() = 0;
};
class observer_concrete : public observer
{
  public:
    virtual void notify() override
    { }
};
class subject
{
  public:
    void register_observer(observer& o)
    {
      observers.push_back(o);
    }
    void notify_observers()
    {
      for (observer& o : observers) {
        o.notify();
      }
    }
  private:
    std::vector<std::reference_wrapper<observer>> observers;
};


// Visitor
// Separate generic algorithms from the elements or structure on which they operate.
// “double dispatch”: visitor and element 均可派生
// The visitor pattern is particularly useful when the elements are part of a larger structure, 
// in which case the accept function can call itself recursively down the structure.
class element_concrete_1;
class element_concrete_2;
class visitor
{
  public:
    virtual void visit(element_concrete_1& el) = 0;
    virtual void visit(element_concrete_2& el) = 0;
};
class visitor_concrete : public visitor
{
  public:
    virtual void visit(element_concrete_1& el) override
    {
      // Do something with el
    };
    virtual void visit(element_concrete_2& el) override
    {
      // Do something with el
    };
};
class element
{
  public:
    virtual void accept(visitor& v) = 0;
};
class element_concrete_1 : public element
{
  public:
    virtual void accept(visitor& v) override
    {
      v.visit(*this);
    }
};
class element_concrete_2 : public element
{
  public:
    virtual void accept(visitor& v) override
    {
      v.visit(*this);
    }
};



// Builder
// Separate the complex construction of an object from its representation.
#include <vector>
class foo
{
  public:
    class builder;
    foo(int prop1, bool prop2, bool prop3, std::vector<int> prop4)
      : prop1{prop1}, prop2{prop2}, prop3{prop3}, prop4{prop4}
    { }
    int prop1;
    bool prop2;
    bool prop3;
    std::vector<int> prop4;
};
class foo::builder
{
  public:
    builder& set_prop1(int value) { prop1 = value; return *this; };
    builder& set_prop2(bool value) { prop2 = value; return *this; };
    builder& set_prop3(bool value) { prop3 = value; return *this; };
    builder& set_prop4(std::vector<int> value) { prop4 = value; return *this; };
    foo build() const
    {
      return foo{prop1, prop2, prop3, prop4};
    }
  private:
    int prop1 = 0;
    bool prop2 = false;
    bool prop3 = false;
    std::vector<int> prop4 = {};
};
int main()
{
  foo f = foo::builder{}.set_prop1(5)
                        .set_prop3(true)
                        .build();
}



// Decorator
// Extend the functionality of a class.
class foo
{
  public:
    virtual void do_work() = 0;
};
class foo_concrete : public foo
{
  public:
    virtual void do_work() override
    { }
};
class foo_decorator : public foo
{
  public:
    foo_decorator(foo& f)
      : f(f)
    { }
    virtual void do_work() override
    {
      // Do something else here to decorate
      // the do_work function
      f.do_work();
    }
  private:
    foo& f;
};
void bar(foo& f)
{
  f.do_work();
}
int main()
{
  foo_concrete f;
  foo_decorator decorated_f{f};
  bar(decorated_f);
}

