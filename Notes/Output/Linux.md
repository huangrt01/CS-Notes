### Linux

* [哪些命令行工具让你相见恨晚？ - Jackpop的回答 - 知乎](https://www.zhihu.com/question/41115077/answer/624385012)

#### MIT 6.NULL课程
https://missing.csail.mit.edu/ ，介绍了如何利用工具提升效率

##### Lecture1. Course overview + the shell
* shell：空格分割输入，`~` is short for "home"
* environment variable: `echo $PATH`; `vim ~/.zshrc`
  * `$PATH`可以作为输入
* connecting programs：
  * <和>：rewire the input and output streams; >>可append
  * `cat < hello.txt > hello2.txt`
  * wire: `ls -l | tail -n1`，``
  * `curl --head --silent baidu.com | grep --ignore-case content-length | cut -f2 -d ' '`

* sudo: super user，linux系统可改/sys下面的sysfs

`echo 1 | sudo tee /sys/class/leds/input6::scrolllock/brightness`

* [shell中不同类型quotes的含义](https://www.gnu.org/software/bash/manual/html_node/Quoting.html)

##### Lecture2. Shell Tools and Scripting
shell scripting
* foo=bar, \$foo	注意等号前后不能有space，否则被当成参数
* 单引号和双引号的区别：同样套在\$foo上，前者是literal meaning，而" "会替换成变量值
* shell scripting也有if、case、while、for、function特性
  * source mcd.sh后即可使用。cd如果在function内部使用，针对的是子shell，不影响外部，因此直接用./mcd.sh不合适
```shell
#!/bin/bash
mcd(){
	mkdir -p "$1"
	cd "$1"
}
```

* **special variables**
  * \$0 - Name of the script
  * \ <img src="https://www.zhihu.com/equation?tex=1%20to%20%5C" alt="1 to \" class="ee_img tr_noresize" eeimg="1"> 9 - Arguments to the script. $1 is the first argument and so on.
  * \$@ - All the arguments
  * \$# - Number of arguments
  * \$? - Return code of the previous command
  * \ <img src="https://www.zhihu.com/equation?tex=%5C" alt="\" class="ee_img tr_noresize" eeimg="1">  - Process Identification number for the current script
  * !! - Entire last command, including arguments. A common pattern is to execute a command only for it to fail due to missing permissions, then you can quickly execute it with sudo by doing sudo !!
  * \$_ - Last argument from the last command. If you are in an interactive shell, you can also quickly get this value by typing Esc followed by .

* ||和&& operator：机制和error code联系，true和false命令返回固定的error code
```shell
false || echo "Oops, fail"
# Oops, fail

true || echo "Will not be printed"
#

true && echo "Things went well"
# Things went well

false && echo "Will not be printed"
#

false ; echo "This will always run"
# This will always run

```

* command substitution: `for file in $(ls)`
* process substitution: 生成返回temporary file，`diff <(ls foo) <(ls bar)`

```shell
#!/bin/bash

echo "Starting program at $(date)" # Date will be substituted

echo "Running program  <img src="https://www.zhihu.com/equation?tex=0%20with%20" alt="0 with " class="ee_img tr_noresize" eeimg="1"> # arguments with pid $$"

for file in $@; do
    grep foobar $file > /dev/null 2> /dev/null
    # When pattern is not found, grep has exit status 1
    # We redirect STDOUT and STDERR to a null register since we do not care about them
    if [[ $? -ne 0 ]]; then
        echo "File $file does not have any foobar, adding one"
        echo "# foobar" >> "$file"
    fi
done

```
* 2>重定向stderr；引申：>&2，重定向到stderr
* -ne，更多的查看man test
* “test command”， \[\[和\[的区别：http://mywiki.wooledge.org/BashFAQ/031 ，[[是compound command，存在special parsing context，寻找reserved words or control operators 

**shell globbing 通配**

* wildcard通配符：?和* 	`ls *.sh`
* {}: `mv *{.py,.sh} folder`
* `touch {foo,bar}/{a..h}`

* 利用[shellcheck](https://github.com/koalaman/shellcheck)检查shell scripts的错误

* [shebang](https://en.wikipedia.org/wiki/Shebang_(Unix))line 进行解释，可以利用env命令
  * `#!/usr/bin/env python`
  * ` #!/usr/bin/env -S /usr/local/bin/php -n -q -dsafe_mode=0`

**shell函数和scripts的区别：**

- Functions have to be in the same language as the shell, while  scripts can be written in any language. This is why including a shebang  for scripts is important.
- Functions are loaded once when their definition is read. Scripts  are loaded every time they are executed. This makes functions slightly  faster to load but whenever you change them you will have to reload  their definition.
- Functions are executed in the current shell environment whereas  scripts execute in their own process. Thus, functions can modify environment variables, e.g. change your current directory, whereas scripts can’t. Scripts will be passed by value environment variables  that have been exported using [`export`](http://man7.org/linux/man-pages/man1/export.1p.html)
  * 比如cd只能在function中影响到外界shell
- As with any programming language functions are a powerful  construct to achieve modularity, code reuse and clarity of shell code.  Often shell scripts will include their own function definitions.

** shell tools**
帮助文档：
* XX -h
* man XX
* :help 或 ? (interactive)
* [tldr](https://tldr.sh/)：比man好用！

**shell中的查找**
* 寻找文件：find, fd, locate
* 寻找代码：grep, [ack](https://beyondgrep.com/), [ag](https://github.com/ggreer/the_silver_searcher) and [rg](https://github.com/BurntSushi/ripgrep)
  * grep -R can be improved in many ways, such as ignoring .git folders, using multi CPU support, &c

```shell
# Find all python files where I used the requests library
rg -t py 'import requests'
# Find all files (including hidden files) without a shebang line
rg -u --files-without-match "^#!"
# Find all matches of foo and print the following 5 lines
rg foo -A 5
# Print statistics of matches (# of matched lines and files )
rg --stats PATTERN
```

* 寻找shell指令
  * `history | grep find`
  
  * ctrl+R，可结合[fzf](https://github.com/junegunn/fzf/wiki/Configuring-shell-key-bindings#ctrl-r)，[教程](https://www.jianshu.com/p/d64553a37d69)：高效查找，手动选择
  
  * [zsh-history-substring-search](https://github.com/zsh-users/zsh-history-substring-search): 键盘上下键寻找历史
  
  * [zsh-autosuggestions](https://github.com/zsh-users/zsh-autosuggestions)：键盘右键快速键入
  * 如果输入命令有leading space，不会记入历史数据；如果不慎记入，可修改`.bash_history`或`.zsh_history`

* 寻找目录
  * [fasd](https://github.com/clvv/fasd): 用[frecency](https://developer.mozilla.org/en/The_Places_frecency_algorithm)(frequency+recency)这个指标排序，这一指标最早用于火狐浏览器
  * [autojump](https://www.baidu.com/link?url=mmPr58MUREjyOpep_Bjba3FyOvqmlUlHSjwpit3kmUPWMWCrvvrUjx1-MKzWeBCsFBiJoXKF-A3Qk23C07rCTa&wd=&eqid=c4204f66000031cb000000065ebf6b15)


#### zsh
* [oh-my-zsh](https://github.com/ohmyzsh/ohmyzsh)
* [zsh的10个优点](https://blog.csdn.net/rapheler/article/details/51505003)，[zsh介绍](https://www.cnblogs.com/dhcn/p/11666845.html)
* [MacOS配置iTerm2+zsh+powerline](https://www.jianshu.com/p/2e8c340c9496)
* autojump: j, jc, jo, jco, j A B
* [zsh-autosuggestions](https://github.com/zsh-users/zsh-autosuggestions)
* [zsh-history-substring-search](https://github.com/zsh-users/zsh-history-substring-search)

Aliases
* pyfind
* pyclean [dirs]
* pygrep \<text\> 
```shell
alias python3="/Users/huangrt01/anaconda3/bin/python3"
# alias base
alias ll='ls -alGh'
alias la='ls -a'
alias l='ls -CF'
alias cls='clear'
alias gs='git status'
alias gc='git commit'
alias gqa='git add .'
# alias docker
alias dkst="docker stats"
alias dkps="docker ps"
alias dklog="docker logs"
alias dkpsa="docker ps -a"
alias dkimgs="docker images"
alias dkcpup="docker-compose up -d"
alias dkcpdown="docker-compose down"
alias dkcpstart="docker-compose start"
alias dkcpstop="docker-compose stop"
```

#### a
#### b
#### c
* cat
* cd
* chmod：sudo chmod 777 文件修改为可执行
* curl
  * -I/--head: 只显示传输文档，经常用于测试连接本身
`curl --head --silent baidu.com | grep --ignore-case content-length | cut -f2 -d ' '`
* cut
  * 使用 -f 选项提取指定字段：`cut -f2,3 test.txt`
* cp
#### d
* d: zsh的特点，可显示最近10个目录，然后`cd -数字`进入
* date：日期
* diff：[Linux中diff的渊源](https://www.cnblogs.com/moxiaopeng/articles/4853352.html)

#### e
* echo: 输出输入，空格分割
* env: 进入环境
  * 读man env，` #!/usr/bin/env -S /usr/local/bin/php -n -q -dsafe_mode=0`，利用env来传参。（FreeBSD 6.0之后不能直接传参，解释器会把多个参数合成一个参数）
* [export](http://man7.org/linux/man-pages/man1/export.1p.html)

#### f
* [fd](https://github.com/sharkdp/fd)：作为find的替代品
  * colorized output, default regex matching, Unicode support, more intuitive syntax
* find：1）寻找文件； 2）机械式操作
  * -iname：大小写不敏感
```shell
# Find all directories named src
find . -name src -type d
# Find all python files that have a folder named test in their path
find . -path '**/test/**/*.py' -type f
# Find all files modified in the last day
find . -mtime -1
# Find all zip files with size in range 500k to 10M
find . -size +500k -size -10M -name '*.tar.gz'

# Delete all files with .tmp extension
find . -name '*.tmp' -exec rm {} \;
# Find all PNG files and convert them to JPG
find . -name '*.png' -exec convert {} {.}.jpg \;

```
#### g
#### h
#### i
#### j
#### k
#### l
* locate
  * Most would agree that `find` and `fd` are good but some of you might be wondering about the efficiency of  looking for files every time versus compiling some sort of index or  database for quickly searching. That is what [`locate`](http://man7.org/linux/man-pages/man1/locate.1.html) is for. `locate` uses a database that is updated using [`updatedb`](http://man7.org/linux/man-pages/man1/updatedb.1.html). In most systems `updatedb` is updated daily via [`cron`](http://man7.org/linux/man-pages/man8/cron.8.html). Therefore one trade-off between the two is speed vs freshness. Moreover `find` and similar tools can also find files using attributes such as file size, modification time or file permissions while `locate` just uses the name. A more in depth comparison can be found [here](https://unix.stackexchange.com/questions/60205/locate-vs-find-usage-pros-and-cons-of-each-other).
* ls
  * -l: long listing format; drwxr-xr-x，d代表文件夹，后面3*3代表owner、owning group、others的权限
  * r：read，w：modify，x：execute
#### m
* man: q退出
* mkdir
* mv
#### n
#### o 
#### p
* pwd: print cwd
#### q
#### r
#### s
#### t
* tail
  * `ls -l | tail -n1`
  * -f：不断读最新内容，实时监视
#### u
#### v
#### w
* which：找到程序路径
#### x
#### y
#### z