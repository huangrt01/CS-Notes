### C++
[toc]
#### debug
[macOS上用VSCode](https://zhuanlan.zhihu.com/p/106935263?utm_source=wechat_session)

[lldb的使用](https://www.jianshu.com/p/9a71329d5c4d)
*  breakpoint set -n main, run, print, next

#### C

* volatile特性：这条代码不要被编译器优化，`volatile int counter = 0`，避免出现意料之外的情况，常用于全局变量

* C不能用for(int i=0; ; ) 需要先定义i

* 负数取余是负数，如果想取模：a = (b%m+m)%m;

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
* 虚析构函数    =>对象内有虚函数表，指向虚函数表的指针：32位系统4字节，64位系统8字节
* 虚基类偏移量表指针

##### 可扩展性
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


#### 输入输出

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

#### STL
十三大头文件：\<algorithm>、\<functional>、\<deque>、、\<iterator>、\<array>、\<vector>、\<list>、\<forward_list>、\<map>、\<unordered_map>、\<memory>、\<numeric>、\<queue>、\<set>、\<unordered_set>、\<stack>、\<utility>

##### \<algorithm>
sort，自己定义cmp函数，注意cmp的定义：类内静态，传参引用
* `static bool cmp1(vector<int> &a, vector<int> &b)`


##### \<deque>
* deque，两端都能进出，双向队列
* [STL之deque实现详解]( https://blog.csdn.net/u010710458/article/details/79540505?depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-6&utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-6)
* deque的pop_front相当于queue的pop()

##### \<list>
* 参考[LRU cache](https://leetcode-cn.com/problems/lru-cache/)，类似双向链表的实现
  * map<int,list<pair<int,int>>::iterator> m;
* r.push_front(…), r.begin(), r.back()

##### \<vector>
* 初始化，可以用列表
  * 也可以 `int a[10]={…};vector<int> b(a,a+10); `       左闭右开
* 方法：
  * reverse(nums.begin(),nums.end());
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

#### 其它的库

##### \<exception>
https://blog.csdn.net/qq_37968132/article/details/82431775

throw invalid_argument("Invalid input.");

##### \<pthread.h>
* [使用封装好的线程操作接口，mythreads.h]()：

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






