### Python

[toc]

#### 基础操作

##### 数据结构

```python
list.extend(list)
list.append(item)

list = [x.strip() for x in list_string.split() if x.strip()]

# dict: 同一dict中存储的value类型可以不一样
```
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



[浅拷贝与深拷贝](https://zhuanlan.zhihu.com/p/25221086)，[copy.py](https://docs.python.org/3/library/copy.html)

核心思想：

* 赋值是将一个对象的地址赋值给一个变量，让变量指向该地址（ 旧瓶装旧酒 ）。
* 修改不可变对象（str、tuple）需要开辟新的空间
* 修改可变对象（list等）不需要开辟新的空间

=> 所以说改 list 会牵连复制的对象，而改 str 等不会互相影响

```python
import copy
copy.deepcopy(dict)
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


#### 类

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

* @func 作为类或者函数的修饰符
  * 当解释器读到@修饰符之后，会先解析@后的内容，直接就把@下一行的函数**或者类**作为@后边的函数的参数，然后将返回值赋值给下一行修饰的函数对象
  * @func 修饰类，常用于 register

```
@model_registry.register('model_type')
```

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

#### datetime

```python
from datetime import datetime
datetime.now()
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

#### pandas

https://pandas.pydata.org/pandas-docs/stable/index.html

`crosstab` 统计分组频率

`drop`

`get_dummies` convert categorical variables to sets of indicator

```python
data['not_working'] = np.where(np.in1d(data['job'], ['student', 'retired', 'unemployed']), 1, 0)
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

#### subprocess

```python
import subprocess


def runCommand(cmd):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    res = p.stdout.read()
    return res.strip("\n")
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

