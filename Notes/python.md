### Python

[toc]

#### 基础数据
```python
bool(int(str(3))) -> True
bool(int(str(0))) -> False
```

#### 数据结构

```python
list.extend(list)
list.append(item)

# split不加参数，默认为所有的空字符，包括空格、换行(\n)、制表符(\t)等
list = [x.strip() for x in list_string.split() if x.strip()]
list = filter(lambda x: x != 2, iterable)

# dict: 同一dict中存储的value类型可以不一样

# 普通dict的default插入方式（类似于C++的[]）
obj = dict.setdefault(key, default=None)

# set
unique_elements = list(set([2,1,2])) # Remove duplicates

# set operations: https://www.linuxtopia.org/online_books/programming_books/python_programming/python_ch16s03.html
&, |, -, ^

```
#### collections

collections.defaultdict(list)、collections.defaultdict(set)

* defaultdict相比普通dict的区别在于：使用索引时，如果未查找到，会自动插入默认值
* dict 可以用 tuple 作 key

```python
d = defaultdict(lambda: 0)
d[key1, key2] = val
if (key, key2) in d:
  ...
for k in d.iteritems(): # 不支持直接用 for k, v 遍历
  v = d[k]
  ...
for k,v in d.items(): # python3.6+
  ...
```

[collections.counter](https://docs.python.org/3/library/collections.html#collections.Counter)



```python
class Example(collections.namedtuple('Example', ['aid', 'bid', 'cid', 'did']))

	@classmethod
	def from_abcd(cls, a, b, c, d):
    return cls(a, b, c, d)
```



##### [浅拷贝与深拷贝](https://zhuanlan.zhihu.com/p/25221086)，[copy.py](https://docs.python.org/3/library/copy.html)

核心思想：

* 赋值是将一个对象的地址赋值给一个变量，让变量指向该地址（ 旧瓶装旧酒 ）。
* 修改不可变对象（str、tuple）需要开辟新的空间
* 修改可变对象（list等）不需要开辟新的空间

=> 所以说改 list 会牵连复制的对象，而改 str 等不会互相影响

```python
import copy
copy.deepcopy(dict)
```



##### queue

```python
# queue.py
task_done()
join()
put(data)
get()

queue的利用：新线程prefetch内容塞进queue里，可以拿到遍历queue的更快的生成器（yield结尾）
```









一些细节

* 运行文件乱码问题，在文件开头加 `# coding=utf-8`




#### 正则表达式

教程：https://www.runoob.com/regexp/regexp-tutorial.html

python正则：https://www.runoob.com/python3/python3-reg-expressions.html

```python
import re
tmp = re.sub("pattern", "", line.strip('\n'))
matchObj = re.match(r'', line.strip('\n'), re.M|re.I)
```

#### argparse

```python
import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_xxx', type=int, default=10, help='number of xxx')
    args = parser.parse_args()
    
# 设计模式：可以直接把args传入程序中的各种类作为self._args成员

# subparser
subparsers = parser.add_subparsers(help='sub-command help')
#添加子命令 add
parser_a = subparsers.add_parser('add', help='add help')
parser_a.add_argument('-x', type=int, help='x value')
parser_a.add_argument('-y', type=int, help='y value')
#设置默认函数
parser_a.set_defaults(func=add)
#添加子命令 sub
parser_s = subparsers.add_parser('sub', help='sub help')
parser_s.add_argument('-x', type=int, help='x value')
parser_s.add_argument('-y', type=int, help='y value')
#设置默认函数
parser_s.set_defaults(func=sub)

args = parser.parse_args()
args.func(args)

$python subc.py add -x 1 -y 2
x + y =  3
$python subc.py sub -x 1 -y 2
x - y =  -1
```





#### module

* [__all\_](https://zhuanlan.zhihu.com/p/54274339)_
   * 控制 from xxx import 的行为
   * 为 lint 等代码检查工具提供辅助

#### 关键词

##### for else

* https://book.pythontips.com/en/latest/for_-_else.html

```python
for item in container:
    if search_something(item):
        # Found it!
        process(item)
        break
else:
    # Didn't find anything..
    not_found_in_container()
```


##### with

* [with 的用法和原理](https://www.cnblogs.com/sddai/p/14411906.html)

#### class

* [@staticmethod v.s. @classmethod](https://stackoverflow.com/questions/136097/difference-between-staticmethod-and-classmethod#:~:text=%40staticmethod)

* @property

```python
class Student(object):

    @property         
    def birth(self):           # 读写属性
        return self._birth

    @birth.setter
    def birth(self, value):
        self._birth = value

    @property
    def age(self):            # 只读属性
        return 2015 - self._birth
```

* [一文让你彻底搞懂Python中\__ str\__和\__repr\__?](https://segmentfault.com/a/1190000022266368)



#### 函数修饰decorator

* @func 作为类或者函数的修饰符
  * 当解释器读到@修饰符之后，会先解析@后的内容，直接就把@下一行的函数**或者类**作为@后边的函数的参数，然后将返回值赋值给下一行修饰的函数对象
  * @func 修饰类，常用于 register

```
@model_registry.register('model_type')
```

* `@functools.wraps(func)` : https://stackoverflow.com/questions/308999/what-does-functools-wraps-do
  * 主要作用是继承一些函数信息
  * kwargs要自己拿

```python
@functools.wraps(func)
def wrapper(*args, **kwargs):
	arg1 = kwargs.get('arg1')
  nonlocal arg1
  ...
  return func(*args, **kwargs)
```

#### 函数

* callable

```python
def call_with_retry(fn: Callable,
                    check_fn: Union[Callable, None] = None,
                    retry_limit: int = 5,
                    retry_interval: int = 60):
    retry = 0
    while True:
        try:
            res = fn()
            if check_fn:
                check_fn(res)
            return res
        except Exception as e:
            logging.warning(f'Function `{fn.__name__}` encountered exception: {repr(e)}.')
            retry += 1
            if retry >= retry_limit > 0:
                break
            logging.warning(f'Retrying {retry} out of {retry_limit} times in {retry_interval} secs.')
            time.sleep(retry_interval)
    return None
```

* @修饰器

```python
#funA 作为装饰器函数
def funA(fn):
    #...
    fn() # 执行传入的fn参数
    #...
    return '...'
@funA
def funB():
    #...
    
---> funB = funA(funB)

# 装饰器嵌套参数函数
def funA(fn):
    def say(*args,**kwargs):
        fn(*args,**kwargs)
    return say
@funA
def funB(arc):
    print("A: ",arc)
@funA
def other_funB(name,arc):
    print(name,arc)
funB("a")
other_funB("B: ","b")
```

* 函数名之前加类名

#### virtualenv

```
virtualenv .myenv --python=python3.8
source .myenv/bin/activate
deactivate
```

[virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/)

```shell
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple virtualenvwrapper

# 加入 zshrc
export WORKON_HOME=~/Envs
mkdir -p $WORKON_HOME
source /usr/local/bin/virtualenvwrapper.sh
# or
source ~/.local/bin/virtualenvwrapper.sh

mkvirtualenv env1
ls $WORKON_HOME
lssitepackages
workon env1

echo 'pip install sphinx' >> $WORKON_HOME/postmkvirtualenv
mkvirtualenv env3
```



#### conda

```shell
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
conda create -n py37 -c anaconda python=3.7
conda activate py37
pip3 install --upgrade pip
pip3 install -r requirements.txt
conda deactivate
```

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

#### os, sys

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

```python
os.getenv('ABC', 'abc') # 注意返回值是str
```



#### datetime

```python
import datetime as dt
dt.datetime.now()
dt.timedelta(hours=1)
dt.timedelta(days=1)

# max datetime (普通调用其timestamp方法可能溢出)
dt.datetime.max.replace(tzinfo=datetime.timezone.utc).timestamp()

# parse timestamp
object = dt.datetime.fromtimestamp(timestamp)

# timezone
import pytz
a = dt.datetime(2022,1,1,tzinfo=pytz.timezone('UTC'))
b = a.tzinfo.localize(
            dt.datetime.combine(object2.date(), dt.time.min)
        )
assert(a == b)

# local timezone
from tzlocal import get_localzone # $ pip install tzlocal
local_tz = get_localzone() 
```

#### fstring

https://www.datacamp.com/tutorial/f-string-formatting-in-python

```python
person = {"name": "John", "age": 19}
## 双引号套单引号
print(f"{person['name']} is {person['age']} years old.")
```

#### functools

```python
from functools import reduce
nparam = reduce(lambda x, y: int(x) * int(y), shape, 1)
```


#### imp

[What does the first argument of the imp.load_source method do?](https://stackoverflow.com/questions/31773310/what-does-the-first-argument-of-the-imp-load-source-method-do)

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

#### multiprocessing

* 关于spawn：https://superfastpython.com/multiprocessing-start-method/

```python
import multiprocessing as mp
mp.set_start_method('spawn')
queue = mp.Queue(num_parallel*2)
producer_proc = mp.Process(name="Producer-0", target=producer, args=(args, queue, sys.stdin.fileno()), daemon=True)

p = mp.Process(...)
# p.setDaemon(True),守护进程，如果主进程结束，则强行让p结束
p.start()
while True:
  if not p.is_alive():
    if p.exitcode != 0:
      raise RuntimeError(...)
    else:
      break
  time.sleep(0.1)

# p.join()

```

#### 多线程编程

```python
class PeriodicRunner(object):
    def __init__(self, name, cond, callback, interval=5, daemon=True):
        self._mutex = threading.Lock()
        self._name = name
        self._cond = cond
        self._callback = callback
        self._interval = interval
        self._running = True
        if not callable(cond) or not callable(callback):
            self._running = False
            return
        self._thread = threading.Thread(target=self.run, name=name, args=(), daemon=daemon)
        self._thread.start()

    def is_alive(self):
        return self._thread.is_alive()

    def run(self):
        try:
            while True:
                with self._mutex:
                    if not self._running:
                        return
                if self._cond():
                    self._callback()
                time.sleep(self._interval)
        except Exception as e:
            logging.info('PeriodicRunner[{}] caught exception[{}]'.format(self._name, repr(e)))
        finally:
            logging.info('PeriodicRunner[{}] exit!'.format(self._name))

    def stop(self):
        logging.info('PeriodicRunner[{}] stop!'.format(self._name))
        with self._mutex:
            self._running = False
        self._thread.join()
```



#### pandas

https://pandas.pydata.org/pandas-docs/stable/index.html

`crosstab` 统计分组频率

`drop`

`get_dummies` convert categorical variables to sets of indicator

```python
data['not_working'] = np.where(np.in1d(data['job'], ['student', 'retired', 'unemployed']), 1, 0)
```

#### pdb

https://docs.python.org/zh-cn/3/library/pdb.html

```python
import pdb

pdb.set_trace()
```




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

#### schedule

https://stackoverflow.com/questions/15088037/python-script-to-do-something-at-the-same-time-every-day

```python
pip install schedule

import schedule
import time

class Scheduler:
  def job(self, t):
    logging.info(t)
    print(t)
  def func(self):
    t = 'Done'
    schedule.every().minutes.at(":17").do(self.job, t) 
    while True:
        schedule.run_pending()
        time.sleep(1) # wait one second
    
nohup python2.7 MyScheduledProgram.py &
```



#### struct

https://docs.python.org/3/library/struct.html

```shell
id_size = fd.read(8)
# id_size = struct.pack('>Q', len(proto))[::-1]
id_size = struct.unpack('<Q', id_size)[0]
```



#### subprocess

```python
import subprocess


def runCommand(cmd):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    res = p.stdout.read()
    return res.strip("\n")
```

#### unittest

```python
import unittest

class MyTestCase(unittest.TestCase):
  def testFunction(self):
    a= 1
    b= 2 
    self.assertEqual(a, b)

if __name__ == '__main__':
	unittest.main()
```

```python
# mock sth
import contextlib
self._exit_stack = contextlib.ExitStack()
# context_manager: contextlib.AbstractContextManager
self._exit_stack.enter_context(context_manager)

mock_sleep = self._exit_stack.enter_context(mock.patch('time.sleep'))
mock_sleep.return_value = True
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
format

* Autopep8

```
pip install autopep8
python -m autopep8 -i -r $folder
```

* Flake8
  * `pip install yapf`
  * VSCode setting.json 添加以下字段，文件页面 `Alt+Shift+F` 自动格式化代码

```json
"python.linting.flake8Enabled": true,
"python.formatting.provider": "yapf",
"python.linting.flake8Args": ["--max-line-length=120"],  
"python.linting.pylintEnabled": false
```

