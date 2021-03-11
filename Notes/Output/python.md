### Python

[toc]

#### 基础操作

list操作

```python
list.extend(list)
list.append(item)
```



[浅拷贝与深拷贝](https://zhuanlan.zhihu.com/p/25221086)，[copy.py](https://docs.python.org/3/library/copy.html)

核心思想：

* 赋值是将一个对象的地址赋值给一个变量，让变量指向该地址（ 旧瓶装旧酒 ）。
* 修改不可变对象（str、tuple）需要开辟新的空间
* 修改可变对象（list等）不需要开辟新的空间

=> 所以说改 list 会牵连复制的对象，而改 str 等不会互相影响



#### 类

[@staticmethod v.s. @classmethod]([https://stackoverflow.com/questions/136097/difference-between-staticmethod-and-classmethod#:~:text=%40staticmethod%20function%20is%20nothing%20more,not%20Parent%20class%2C%20via%20inheritance.](https://stackoverflow.com/questions/136097/difference-between-staticmethod-and-classmethod#:~:text=%40staticmethod function is nothing more,not Parent class%2C via inheritance.))



#### ipython

重启kernel，释放GPU内存
```python
import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```



```python
counters = [Counter.remote() for _ in range(10)]  # 创建10个Counter实例
```



[Using the Python zip() Function for Parallel Iteration](https://realpython.com/python-zip-function/)
* python2返回list，python3返回iterator
	* python2耗内存，可以用`iterator.izip(*iterables)`
* 支持sorted函数和sort方法
```python
list(zip(numbers, letters))

try:
    from itertools import izip as zip
except ImportError:
    pass
    
pairs = [(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]
numbers, letters = zip(*pairs)
```



```python
#!/usr/bin/env python
def fib0(): return 0
def fib1(): return 1

s = """def fib{}(): return fib{}() + fib{}()"""

if __name__ == '__main__':

    for n in range(2, 10):
       exec(s.format(n, n-1, n-2))
    from functools import lru_cache
    for n in range(10):
   		exec("fib{} = lru_cache(1)(fib{})".format(n, n))
    print(eval("fib9()"))
```


```python
import sys
import os
for arg in reversed(sys.argv[1:]):
    print(arg)
    
# 当前路径插入系统路径最高优先级，有效解决部分import问题
sys.path.insert(0,os.getcwd())
```

```python
os.path.join(dir,file)
```

#### imp
[What does the first argument of the imp.load_source method do?
](https://stackoverflow.com/questions/31773310/what-does-the-first-argument-of-the-imp-load-source-method-do)
```python
import imp
var_file = imp.load_source('var', file_path)
object = var_file.inside_object()

import var
object = var.inside_object()
```

#### mpl_toolkits

```python
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.scatter(w_array, b_array, loss_array)
ax.set_xlabel('w', size=16)
ax.set_ylabel('b', size=16)
ax.tick_params(labelsize=12)

plt.show()
```

#### pandas

https://pandas.pydata.org/pandas-docs/stable/index.html

#### random

```python
import random
index = [i for i in range(X_train.shape[0])]
random.choice(a)    随机取一个
random.sample(a, n) 随机取n个
random.shuffle(index)
random.randint(0,n)

# 保证同一shuffle顺序
randnum = random.randint(0,100)
random.seed(randnum)
random.shuffle(train_x)
random.seed(randnum)
random.shuffle(train_y)
```



#### 网络

* python传文件

```
tar -cvf file.tar file
python -m SimpleHTTPServer 99**
# python3 -m http.server 99**
wget **.**.**.**:99**/tf_model.tar
tar -xvf file.tar
```



#### 代码风格
* format

```
pip install autopep8
python -m autopep8 -i -r $folder
```