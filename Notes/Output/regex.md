* []表示字符
  * [a-z]
* 预定义字符
  * `.`表示任意一个字符
    * `.*` 表示任意多个字符
  * `\d`表示任意一个数字
  * `\b`表示word前后的空格
  * `+` 代表的是匹配加号(+)之前的正则表达式1次或多次
    * `[ab]+`和`[ab][ab]*`等价
* \w+ 匹配word

* 模式
  * (?i) 不区分大小写，`(?i)(invoke|call|exec|run).*function`


* 匹配行首尾：行首`^`，行尾`$`
  * `egrep "[45]..$"`

![img](https://images2018.cnblogs.com/blog/733013/201809/733013-20180912234030307-1579497375.png)

## 例子

* 解释一下 `PAT = re.compile(
  r'^/(?P<bzid>[-_0-9A-Za-z]+)/service/(?P<base_name>[-_0-9A-Za-z]+)(/(?P<idc>[-_0-9A-Za-z]+):(?P<cluster>[-_0-9A-Za-z]+))?/(?P<server_type>\w+):(?P<index>\d+)(/(?P<replica_id>\d+))?$'
  )`

  * `^`：表示字符串的开头。

  * `/`：表示斜杠字符。

  * `(?P<bzid>[-_0-9A-Za-z]+)`：使用命名捕获组的语法，匹配一个或多个由字母、数字、下划线、短划线组成的字符序列，并将其命名为"bzid"。

  * `/service/`：表示字面字符串"/service/"。

  * `(?P<base_name>[-_0-9A-Za-z]+)`：匹配一个或多个由字母、数字、下划线、短划线组成的字符序列，并将其命名为"base_name"。

  * `(/(?P<idc>[-_0-9A-Za-z]+):(?P<cluster>[-_0-9A-Za-z]+))?`：使用括号和问号表示的可选部分，匹配一个可选的分组，包含一个由字母、数字、下划线、短划线组成的字符序列（命名为"idc"），后跟冒号、再后跟一个由字母、数字、下划线、短划线组成的字符序列（命名为"cluster"）。

  * `/`：表示斜杠字符。

  * `(?P<server_type>\w+)`：匹配一个或多个字母、数字、下划线字符，并将其命名为"server_type"。

  * `:`：表示冒号字符。

  * `(?P<index>\d+)`：匹配一个或多个数字字符，并将其命名为"index"。

  * `(/(?P<replica_id>\d+))?`：使用括号和问号表示的可选部分，匹配一个可选的分组，包含一个或多个数字字符（命名为"replica_id"）。

  * `?`：表示前面的可选部分是可选的，可以出现零次或一次。

  * `$`：表示字符串的结尾。

## Python

```python
PAT = re.compile(
r'^/(?P<bzid>[-_0-9A-Za-z]+)/service/(?P<base_name>[-_0-9A-Za-z]+)(/(?P<idc>[-_0-9A-Za-z]+):(?P<cluster>[-_0-9A-Za-z]+))?/(?P<server_type>\w+):(?P<index>\d+)(/(?P<replica_id>\d+))?$'
)

path: str
matched = PAT.match(path)
group_dict = matched.groupdict()
```

## 技巧

### 捕获组

```python
import re

string = "Hello, World!"
match = re.search(r'(\w+), (\w+)!', string)

if match:
    print(match.group())      # 输出完整的匹配结果 "Hello, World!"
    print(match.group(1))     # 输出第一个捕获组的值 "Hello"
    print(match.group(2))     # 输出第二个捕获组的值 "World"
```

