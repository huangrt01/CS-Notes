[toc]

###  Shell

### MIT 6.NULL课程
https://missing.csail.mit.edu/ ，介绍了如何利用工具提升效率

#### Lecture1. Course overview + the shell
* shell：空格分割输入，`~` is short for "home"，`.`表示当前文件夹
  * `.`在UNIX系统的[遗留问题](https://plus.google.com/101960720994009339267/posts/R58WgWwN9jp)
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

#### Lecture2. Shell Tools and Scripting
##### shell scripting
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
  * for特性的实用例子
```shell
POLICIES=("FIFO" "LRU" "OPT" "UNOPT" "RAND" "CLOCK")
for policy in "${POLICIES[@]}"
do
    for i in 1 2 3 4
    do
        ./paging-policy.py -c -f ./vpn.txt -p "$policy" -C "$i"
    done
    echo ""
done
```

##### special variables
  * `$0` - Name of the script
  * `$1 to \$9` - Arguments to the script. $1 is the first argument and so on.
  * `$@` - All the arguments
  * `$#` - Number of arguments
  * `$?` - Return code of the previous command
  * `$$` - Process Identification number for the current script
  * `!!` - Entire last command, including arguments. A common pattern is to execute a command only for it to fail due to missing permissions, then you can quickly execute it with sudo by doing sudo !!
  * `$_` - Last argument from the last command. If you are in an interactive shell, you can also quickly get this value by typing Esc followed by .
  * `$!` - last backgrounded job

* ||和&& operator：机制和error code联系，true和false命令返回固定的error code
  * [linux中，&和&&, |和|| ,&> 与 >的区别](https://blog.csdn.net/sunfengye/article/details/78973831)
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

##### [Linux-shell中各种替换的辨析](https://www.cnblogs.com/chengd/p/7803664.html)

* variable substitution：`$var, ${var}`
* command substitution: `for file in $(ls)`，可以用`' '`代替`$( )`，但后者辨识度更高
* process substitution: 生成返回temporary file，`diff <(ls foo) <(ls bar)`

```shell
#!/bin/bash

echo "Starting program at $(date)" # Date will be substituted

echo "Running program $0 with $# arguments with pid $$"

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

##### shell globbing 通配
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

##### shell tools

**帮助文档**

* XX -h
* man XX
* :help 或 ? (interactive)
* [tldr](https://tldr.sh/)：比man好用！

**shell中的查找**

* 查找文件：find, fd, locate，见底部命令解释
* 查找代码：grep, [ack](https://beyondgrep.com/), [ag](https://github.com/ggreer/the_silver_searcher) and [rg](https://github.com/BurntSushi/ripgrep)
  * grep -R can be improved in many ways, such as ignoring .git folders, using multi CPU support, &c
  * 代码行数统计工具[cloc](https://github.com/AlDanial/cloc)

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

* 查找shell指令
  * `history | grep find`
  
  * Ctrl-r，可结合[fzf](https://github.com/junegunn/fzf/wiki/Configuring-shell-key-bindings#ctrl-r)，[教程](https://www.jianshu.com/p/d64553a37d69)：高效查找，手动选择
  
  * [zsh-history-substring-search](https://github.com/zsh-users/zsh-history-substring-search): 键盘上下键寻找历史
  
  * [zsh-autosuggestions](https://github.com/zsh-users/zsh-autosuggestions)：键盘右键快速键入
  * 如果输入命令有leading space，不会记入历史数据；如果不慎记入，可修改`.bash_history`或`.zsh_history`

* 查找目录
  * [fasd](https://github.com/clvv/fasd): 用[frecency](https://developer.mozilla.org/en/The_Places_frecency_algorithm)(frequency+recency)这个指标排序，这一指标最早用于火狐浏览器
  * [autojump](https://www.baidu.com/link?url=mmPr58MUREjyOpep_Bjba3FyOvqmlUlHSjwpit3kmUPWMWCrvvrUjx1-MKzWeBCsFBiJoXKF-A3Qk23C07rCTa&wd=&eqid=c4204f66000031cb000000065ebf6b15)
  * More complex tools exist to quickly get an overview of a directory structure [`tree`](https://linux.die.net/man/1/tree), [`broot`](https://github.com/Canop/broot) or even full fledged file managers like [`nnn`](https://github.com/jarun/nnn) or [`ranger`](https://github.com/ranger/ranger)

**Shell编辑**
* `Ctrl-a`光标移动到行前
* ESC进入Vim-mode，ESC-v进入Vim直接编辑


**Exercises**

1. `alias ll='ls -aGhlt'`

2. marco记录directory，polo前往
```shell
#!/bin/bash
marco(){
        foo=$(pwd)
        export MARCO=$foo
}
polo(){
        cd "$MARCO" || echo "cd error"
}
```

3. 实用小工具，比如可以抢实验室GPU（实现的功能相对原题有改动）
```shell
#!/usr/bin/env bash
debug(){
        echo "start capture the program failure log"
        cnt=-1
        ret=1
        while [[ $ret -eq 1 ]]; do
                sh "$1" 2>&1
                ret=$?
                cnt=$((cnt+1))
                # let cnt++
                if [[ $# -eq 2 ]];then
                        sleep "$2"
                fi
        done
        echo "succeed after ${cnt} times"
}
```

4.`fd -e html -0 | xargs -0 zip output.zip`

5.返回文件夹下最近修改的文件：`fd . -0 -t f | xargs -0 stat -f '%m%t%Sm %N' | sort -n | cut -f2- | tail -n 1` (设成了我的fdrecent命令)

  * [stackoverflow讨论](https://stackoverflow.com/questions/5566310/how-to-recursively-find-and-list-the-latest-modified-files-in-a-directory-with-s)
  *  `find . -exec stat -f '%m%t%Sm %N' {} + | sort -n | cut -f2- | tail -n 1`
  *  `find . -type f -print0 | xargs -0 stat -f '%m%t%Sm %N' | sort -n | cut -f2- | tail -n 1`

##### zsh
* [oh-my-zsh](https://github.com/ohmyzsh/ohmyzsh)
* [zsh的10个优点](https://blog.csdn.net/rapheler/article/details/51505003)，[zsh介绍](https://www.cnblogs.com/dhcn/p/11666845.html)
* [MacOS配置iTerm2+zsh+powerline](https://www.jianshu.com/p/2e8c340c9496)
* autojump: j, jc, jo, jco, j A B
* [zsh-autosuggestions](https://github.com/zsh-users/zsh-autosuggestions)
* [zsh-history-substring-search](https://github.com/zsh-users/zsh-history-substring-search)
* [zsh-completions](https://github.com/zsh-users/zsh-completions): tab自动补全，比如输cd按tab会自动识别文件夹；输git add会自动识别需要add的文件

Aliases
* pyfind
* pyclean [dirs]
* pygrep \<text\> 

#### Lecture3. Editors(Vim)
* [Editor War](https://en.wikipedia.org/wiki/Editor_war)
* [Stack Overflow survey](https://insights.stackoverflow.com/survey/2019/#development-environments-and-tools)
*  [Vim emulation for VS code](https://github.com/VSCodeVim/Vim)：个人觉得Vim+VSCode是绝配，Vim负责文本编辑，VSCode负责插件等高级功能
*  [在VSCode中使用Vim的正确方式](https://zhuanlan.zhihu.com/p/141248420)


**Vim的设计逻辑：a modal editor，多模态的编辑器**
- Normal (ESC): for moving around a file and making edits，ESC很重要，我用[Karabiner](https://mac.softpedia.com/get/System-Utilities/KeyRemap4MacBook.shtml)把MacOS的右Command键设成了ESC
- Insert (i): for inserting text
- Replace (R): for replacing text，无需删除，在文本上覆盖编辑；`r`替换字符
- Visual (plain (v), line (V), block (C-v)) mode: for selecting blocks of text
  * `^V = Ctrl-v = <C-v>`
- Command-line (:): for running a command

**Vim基础**
* 插入，按i进入，Esc退出
* 概念：Buffers, tabs, and windows
  *  buffer和window关系：一对多
* Command-line
  * :q   quit (close window)
  * :w   save (“write”)
  * :wq   save and quit    = `ZZ`
  * :e {name of file}   open file for editing   **利用这一命令和:sp在文件间复制粘贴**
  * :ls   show open buffers
  * :help {topic}   open help，`Ctrl-D`显示补全命令列表
  * `:r`提取和合并文件；`:r !ls`可读取存放外部命令输出

##### Vim’s interface is a programming language
**Movement**：也称作“nouns”，因为指代chunks of text

* Basic movement: `hjkl`左下上右
* Words: `w` (下一个词开头), `b` (本词或上一个词开头), `e` (本词或下一个词末尾，常和`a`搭配)
* Lines: `0` (beginning of line), `^` (first non-blank character), `$` (end of line)
* Paragraph (原文没写): `{ and }` 
* Screen: `H` (top of screen), `M` (middle of screen), `L` (bottom of screen)
* Scroll: `Ctrl-U (up), Ctrl-D (down)`
* File: `gg` (beginning of file), `G` (end of file), `Ctrl-G`显示行号信息, `数字+G`移动到某一行
* Line numbers: `:{number}` or `{number}G` (line {number})
* Misc: `%` (corresponding item，比如括号匹配)
* Find: `f{character}`, `t{character}`, `F{character}`, `T{character}`
  - find/to forward/backward {character} on the current line
  - `,` / `;` for navigating matches
* [Search](https://www.cnblogs.com/wayneliu007/p/10322453.html): `/{regex}`向后搜索, `n` / `N` for navigating matches
  * `？{regex}`向前搜索
  * 退出查找高亮状态：`:nohl`
  * `:set ic`忽略大小写；`:set hls is`; 选项前加no可关闭选项`:set noic`

**Selection**：Visual modes 

* plain (v) 
* line (V)
* block (Ctrl-v) 
* `v`键提取后按`: ... w ABC`可以保存文件

**Edits**: "Verbs"

* `i`进入insert模式
* `o` / `O` insert line below / above
* `d{motion}` delete {motion}    
  - e.g. `dw` is delete word, `d$` is delete to end of line, `d0` is delete  to beginning of line
* `c{motion}` change {motion}    
  - e.g. `cw` is change word
  - like `d{motion}` followed by `i`
* `x` delete character (equal do `dl`)
* `s` substitute character (equal to `xi`)
* visual mode + manipulation    
  - select text, `d` to delete it or `c` to change it
* `u` to undo, `<C-r>` to redo，`U`撤销行内命令
* `y` to copy / “yank” (some other commands like `d` also copy)
* `p` to paste
* Lots more to learn: e.g. `~` flips the case of a character
* d, y, c均可双写，表示单行操作
* `A a S s `附加操作，相当于操作后移一格
* `y`操作符命令会把文本复制到一个寄存器3中。然后可以用`p`命令把它取回。因为`y`是一个操作符命令，所以可以用`yw`来复制一个word. 同样可以使用counting, 如用`y2w`命令复制两个word，`yy`命令复制一整行，`Y`也是复制整行的内容，复制当前光标至行尾的命令是`y$`
  { }段首段尾

**Counts**:

- `3w` move 3 words forward
- `5j` move 5 lines down
- `7dw` delete 7 words

**Modifiers**: 接在nouns后面，i=inside，a=around，t=to

- `ci(` change the contents inside the current pair of parentheses
- `ci[` change the contents inside the current pair of square brackets
- `da'` delete a single-quoted string, including the surrounding single quotes
- `d2a` 删除到a之前

##### Vim拓展
* ./vimrc: [课程推荐config](https://missing.csail.mit.edu/2020/files/vimrc), instructors’ Vim configs ([Anish](https://github.com/anishathalye/dotfiles/blob/master/vimrc), [Jon](https://github.com/jonhoo/configs/blob/master/editor/.config/nvim/init.vim) (uses [neovim](https://neovim.io/)), [Jose](https://github.com/JJGO/dotfiles/blob/master/vim/.vimrc))
* plugin: [推荐网站](https://vimawesome.com/)，git clone到`~/.vim/pack/vendor/start/`
  * [ctrlp.vim](https://github.com/ctrlpvim/ctrlp.vim): fuzzy file finder
  * [ack.vim](https://github.com/mileszs/ack.vim): code search
  * [nerdtree](https://github.com/scrooloose/nerdtree): file explorer
    * 手动输`vim -u NONE -c "helptags ~/.vim/pack/my_plugs/start/nerdtree/doc" -c q`激活帮助插件，配置在了我的dotfiles里
  * [vim-easymotion](https://github.com/easymotion/vim-easymotion): magic motions


##### Vim-mode的其它应用
* Shell：If you’re a Bash user, use `set -o vi`. If you use Zsh, `bindkey -v`. For Fish, `fish_vi_key_bindings`. Additionally, no matter what shell you use, you can `export EDITOR=vim`. This is the environment variable used to decide which editor is launched when a program wants to start an editor. For example, `git` will use this editor for commit messages.

* Readline: Many programs use the [GNU Readline](https://tiswww.case.edu/php/chet/readline/rltop.html) library for their command-line interface. 
  * Readline supports (basic) Vim emulation too, which can be enabled by adding the following line to the `~/.inputrc` file: `set editing-mode vi`
  * With this setting, for example, the Python REPL will support Vim bindings.

* Others:

  There are even vim keybinding extensions for web [browsers](http://vim.wikia.com/wiki/Vim_key_bindings_for_web_browsers), some popular ones are [Vimium](https://chrome.google.com/webstore/detail/vimium/dbepggeogbaibhgnhhndojpepiihcmeb?hl=en) for Google Chrome and [Tridactyl](https://github.com/tridactyl/tridactyl) for Firefox. You can even get Vim bindings in [Jupyter notebooks](https://github.com/lambdalisue/jupyter-vim-binding).

##### Vim的其它特性积累

* buffer操作：` :ls,:b num, :bn（下一个）, :bp（前一个）, :b#(上次的buffer) `
* window操作： `:sp / :vsp` split window，`C-w + hjkl`切换
* tab操作：`gt`切换tab
* `Ctrl-O/I` 进入更旧/新的位置

查找替换：
* `:%s/foo/bar/g`
  * replace foo with bar globally in file
  * `%`表示修改全文件而不是第一个匹配串，`/g`表示全行/全文件匹配，`/gc`会提示每个匹配串是否替换 
  * `:#,#s/...`表示对行号之间的内容操作
* ` :%s/\[.*\](\(.*\))/\1/g`
  - replace named Markdown links with plain URLs
* `:g/pattern/command`，对匹配行执行命令

* `.`复制操作
* 外部命令：`:!`


##### Macros
- `q{character}` to start recording a macro in register `{character}`

- `q` to stop recording

- `@{character}` replays the macro

- Macro execution stops on error

- `{number}@{character}` executes a macro {number} times

- Macros can be recursive    

  - first clear the macro with `q{character}q`
  - record the macro, with `@{character}` to invoke the macro recursively  (will be a no-op until recording is complete)

- Example: convert xml to json   ([file](https://missing.csail.mit.edu/2020/files/example-data.xml))    

  - Array of objects with keys “name” / “email”

  - Use a Python program?

  - Use sed / regexes        

    - `g/people/d`
    - `%s/<person>/{/g`
    - `%s/<name>\(.*\)<\/name>/"name": "\1",/g`
    - …

  - Vim commands / macros        

    - `Gdd`, `ggdd` delete first and last lines

    - Macro to format a single element (register `e`)            
      - Go to line with `<name>`
      - `qe^r"f>s": "<ESC>f<C"<ESC>q`

    - Macro to format a person            

      - Go to line with `<person>`
      - `qpS{<ESC>j@eA,<ESC>j@ejS},<ESC>q`

    - Macro to format a person and go to the next person            

      - Go to line with `<person>`
      - `qq@pjq`

    - Execute macro until end of file            

      - `999@q`

    - Manually remove last `,` and add `[` and `]` delimiters
  - [在vimrc中存宏](https://stackoverflow.com/questions/2024443/saving-vim-macros) 

<img src="Shell-MIT-6-NULL/vim.png" alt="vim" style="zoom:100%;" />


##### Resources

- `vimtutor` is a tutorial that comes installed with Vim
- [Vim Adventures](https://vim-adventures.com/) is a game to learn Vim
- [Vim Tips Wiki](http://vim.wikia.com/wiki/Vim_Tips_Wiki)
- [Vim Advent Calendar](https://vimways.org/2019/) has various Vim tips
- [Vim Golf](http://www.vimgolf.com/) is [code golf](https://en.wikipedia.org/wiki/Code_golf), but where the programming language is Vim’s UI
- [Vi/Vim Stack Exchange](https://vi.stackexchange.com/)
- [Vim Screencasts](http://vimcasts.org/)
- [Practical Vim](https://pragprog.com/book/dnvim2/practical-vim-second-edition) (book)

#### Lecture 4.Data Wrangling

#### Lecture 5.Command-line Environment

##### Job Control
杀进程
* signals: software interrupts

* `Ctrl-C`:`SIGINT`; `Ctrl-\`:`SIGQUIT`; `kill -TERM PID` :`SIGTERM`
* [SIGINT、SIGQUIT、 SIGTERM、SIGSTOP区别](https://blog.csdn.net/pmt123456/article/details/53544295)

```python
#!/usr/bin/env python
import signal, time

def handler(signum, time):
    print("\nI got a SIGINT, but I am not stopping")

signal.signal(signal.SIGINT, handler)
i = 0
while True:
    time.sleep(.1)
    print("\r{}".format(i), end="")
    i += 1
```

**Pausing and backgrounding processes**

* 暂停并放入后台：`Ctrl-Z`,`SIGTSTP`
* 继续暂停的job：`fg和bg`；`jobs`搭配`pgrep`
* `$!` - last backgrounded job
* 命令行后缀`&`在背景运行命令
* 关闭终端发出`SIGHUP`信号，使子进程终止，解决方案：
  1. 运行前`nohup` 
  2. 运行后`disown` 
  3. `tmux`
* `SIGKILL`和`SIGSTOP`都不能被相关的系统调用阻塞，因此`SIGKILL`不会触发父进程的清理部分，可能导致子进程成为孤儿进程；如果是`SIGINT`，可能会有handler处理资源，比如有些数据还在内存，需要刷新到磁盘上。

##### tmux: terminal multiplexer

- Sessions - a session is an independent workspace with one or more windows    

  - `tmux` starts a new session.
  - `tmux new -s NAME` starts it with that name. `tmux rename-session -t 0 database` 重命名
  - `tmux ls` lists the current sessions
  - Within `tmux` typing `<C-b> d/D`  detaches the current session
  - `tmux a` attaches the last session. You can use `-t` flag to specify which

- Windows

   \- Equivalent to tabs in editors or browsers, they are visually separate parts of the same session    

  - `<C-b> c` Creates a new window. To close it you can just terminate the shells doing `<C-d> / exit`
  - `<C-b> N` Go to the *N* th window. Note they are numbered
  - `<C-b> p` Goes to the previous window
  - `<C-b> n` Goes to the next window
  - `<C-b> ,` Rename the current window
  - `<C-b> w` List current windows

- Panes

   \- Like vim splits, panes let you have multiple shells in the same visual display.    

  - `<C-b> "` Split the current pane horizontally
  - `<C-b> %` Split the current pane vertically
  - `<C-b> <direction>` Move to the pane in the specified *direction*. Direction here means arrow keys.
  - `<C-b> z` make a pane go full screen. Hit `<C-b> z` again to shrink it back to its previous size
  - `<C-b> [` Start scrollback. You can then press `<space>` to start a selection and `enter` to copy that selection.
  - `<C-b> <space> ` Cycle through pane arrangements.

* For further reading, [here](https://www.hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/) is a quick tutorial on `tmux` and [this](http://linuxcommand.org/lc3_adv_termmux.php) has a more detailed explanation that covers the original `screen` command. You might also want to familiarize yourself with [`screen`](http://man7.org/linux/man-pages/man1/screen.1.html), since it comes installed in most UNIX systems.
* tmux是client-server的实现模式
* [tmux customizations](https://www.hamvocke.com/blog/a-guide-to-customizing-your-tmux-conf/)

##### Aliases
* `alias ll`可print出alias的对象
* `unalias ll`可解除alias
```shell
# alias base
alias v='vim'
alias ll='ls -aGhlt'
alias la='ls -a'
alias l='ls -CF'
alias cls='clear'

alias gs='git status'
alias gc='git commit'
alias gqa='git add .'

alias v="vim"
alias mv="mv -i" # -i prompts before overwrite
alias mkdir="mkdir -p" # -p make parent dirs as needed
alias df="df -h" # -h prints human readable format

alias vfzf='vim $(fzf)' #vim打开搜索到的结果文件
alias cdfzf='cd $(find * -type d | fzf)'
alias gitfzf='git checkout $(git branch -r | fzf)'


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

##### Dotfiles
* [很详尽的tutorial](https://www.anishathalye.com/2014/08/03/managing-your-dotfiles/)
* [有关dotfile的种种](https://dotfiles.github.io/)
* [shell-startup的机理](https://blog.flowblok.id.au/2013-02/shell-startup-scripts.html)

e.g.
- `bash` - `~/.bashrc`, `~/.bash_profile`
- `git` - `~/.gitconfig`
- `vim` - `~/.vimrc` and the `~/.vim` folder
- `ssh` - `~/.ssh/config`
- `tmux` - `~/.tmux.conf`

管理方法：单独的文件夹，版本控制，**symlinked** into place using a script
  * 用git的submodule
  * [Dotbot](https://github.com/anishathalye/dotbot) 

* **Easy installation**: if you log in to a new machine, applying your customizations will only take a minute.
* **Portability**: your tools will work the same way everywhere.
* **Synchronization**: you can update your dotfiles anywhere and keep them all in sync.
* **Change tracking**: you’re probably going to be maintaining your dotfiles for your entire programming career, and version history is nice to have for long-lived projects.

一些有用的构造代码块：
```shell
# $HOME/dotfiles
BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$(uname)" == "Linux" ]]; then {do_something}; fi

# Check before using shell-specific features
if [[ "$SHELL" == "zsh" ]]; then {do_something}; fi

# You can also make it machine-specific
if [[ "$(hostname)" == "myServer" ]]; then {do_something}; fi

# Test if ~/.aliases exists and source it
if [ -f ~/.aliases ]; then
    source ~/.aliases
fi
```

在`~/.gitconfig`里加

```
[include]
    path = ~/.gitconfig_local
```

##### Remote Machines
* [装虚拟机](https://hibbard.eu/install-ubuntu-virtual-box/)
  * `sudo apt-get install --reinstall lightdm && sudo systemctl start lightdm`图形界面
  * `Ctrl+Alt+A`打开终端
  * 自动/手动设置共享文件夹：`sudo mkdir -p /media/sf_<FolderName> && 
sudo mount -t vboxsf -o rw,gid=vboxsf FolderName /media/sf_FolderName `
* ssh可执行命令
  * `ssh foobar@server ls | grep PATTERN` 
  * `ls | ssh foobar@server grep PATTERN`
* [用SSH连GitHub](https://help.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh)
```shell
ssh-keygen -t rsa -b 4096 -C "huangrt01@163.com"
eval "$(ssh-agent -s)"
ssh-add -K ~/.ssh/id_rsa
pbcopy < ~/.ssh/id_rsa.pub  #适合MacOS , Linux用xclip
# 上github添加SSH Key

ssh -T git@github.com

ssh-keygen -y -f ~/.ssh/id_rsa
```
* [ssh agent及其forwarding特性](https://www.ssh.com/ssh/agent#ssh-agent-forwarding)
* ssh连虚拟机
```shell
ssh -p 2222 cs144@localhost
# ssh will look into .ssh/authorized_keys to determine which clients it should let in.
ssh-copy-id -i ~/.ssh/id_rsa.pub -p 2222 cs144@localhost
# or
cat .ssh/id_ed25519.pub | ssh foobar@remote 'cat >> ~/.ssh/authorized_keys'
```
* ssh传文件
  - `ssh+tee`, the simplest is to use `ssh` command execution and STDIN input by doing `cat localfile | ssh remote_server 'tee serverfile'`. Recall that [`tee`](http://man7.org/linux/man-pages/man1/tee.1.html) writes the output from STDIN into a file.
  - [`scp`](http://man7.org/linux/man-pages/man1/scp.1.html) when copying large amounts of files/directories, the secure copy `scp` command is more convenient since it can easily recurse over paths. The syntax is `scp -P 2075 -r path/to/local_file remote_host:path/to/remote_file`
  - [`rsync`](http://man7.org/linux/man-pages/man1/rsync.1.html) improves upon `scp` by detecting identical files in local and remote, and preventing  copying them again. It also provides more fine grained control over  symlinks, permissions and has extra features like the `--partial` flag that can resume from a previously interrupted copy. `rsync` has a similar syntax to `scp`.

* Port Forwarding: `localhost:PORT or 127.0.0.1:PORT`
  * Local Port Forwarding: ssh端口重定向：`-L 9999:127.0.0.1:8097，比如在服务器开`jupyter notebook`
    * <img src="Shell-MIT-6-NULL/001.png" alt="Local Port Forwarding" style="zoom:100%;" />
  * Remote Port Forwarding
    * <img src="Shell-MIT-6-NULL/002.png" alt="Remote Port Forwarding" style="zoom:100%;" />

* ssh configuration: `~/.ssh/config`，server side: `/etc/ssh/sshd_config`，调端口、X11 forwarding等
```shell
Host vm
    User foobar
    HostName 172.16.174.141
    Port 2222
    IdentityFile ~/.ssh/id_ed25519
    LocalForward 9999 localhost:8888

# Configs can also take wildcards
Host *.mit.edu
    User foobaz
```

* 其它
  * openconnect: `sudo openconnect --juniper  https://sslvpn.tsinghua.edu.cn -u 2015010356` 
  
  * vscode remote-ssh中[ssh_config的配置细节](https://linux.die.net/man/5/ssh_config) 
  
  * [Mosh](https://mosh.org/), the mobile shell, improves upon ssh, allowing roaming connections, intermittent connectivity and  providing intelligent local echo.
  
  * [sshfs](https://github.com/libfuse/sshfs) can mount a folder on a remote server locally, and then you can use a local editor.

##### Shells & Frameworks

zsh的新特性
- Smarter globbing, `**`: `**/README.md`可递归地列出相应文件
- Inline globbing/wildcard expansion
- Spelling correction
- Better tab completion/selection (`XXX -`加`tab`会列出说明，很贴心)
- Path expansion (`cd /u/lo/b` will expand as `/usr/local/bin`)


##### Terminal Emulators
[comparison](https://anarc.at/blog/2018-04-12-terminal-emulators-1/)

重点：

- Font choice
- Color Scheme
- Keyboard shortcuts
- Tab/Pane support
- Scrollback configuration
- Performance (some newer terminals like [Alacritty](https://github.com/jwilm/alacritty) or [kitty](https://sw.kovidgoyal.net/kitty/) offer GPU acceleration).

**Exercises**
* pidwait，用于跨终端的控制
```shell
#!/bin/bash
pidwait(){
	try=0
	while [[ $try -eq 0 ]]; do
		kill -0 "$1" || try=1
		sleep 1
	done
}
```


#### Lecture 6.Version Control (Git)
* 见[我的Git笔记](https://github.com/huangrt01/CS-Notes/blob/master/Notes/Output/git.md)

#### Lecture 7.Debugging and Profiling

### Linux命令按字母分类
#### a
#### b
* bg: resume后台暂停的命令
#### c
* cat
* cd
* chmod：sudo chmod 777 文件修改为可执行
* cloc: 代码行数统计
* curl
  * -I/--head: 只显示传输文档，经常用于测试连接本身
`curl --head --silent baidu.com | grep --ignore-case content-length | cut -f2 -d ' '`
* cut
  * 使用 -f 选项提取指定字段：`cut -f2,3 test.txt`
* cp
#### d
* d: zsh的特点，可显示最近10个目录，然后`cd -数字`进入
* date：日期
* df: disk情况
* disown
* diff：[Linux中diff的渊源](https://www.cnblogs.com/moxiaopeng/articles/4853352.html)

#### e
* echo: 输出输入，空格分割
* env: 进入环境
  * 读man env，` #!/usr/bin/env -S /usr/local/bin/php -n -q -dsafe_mode=0`，利用env来传参。（FreeBSD 6.0之后不能直接传参，解释器会把多个参数合成一个参数）
* [export](http://man7.org/linux/man-pages/man1/export.1p.html)

#### f
* [fd](https://github.com/sharkdp/fd)：作为find的替代品
  * colorized output, default regex matching, Unicode support, more intuitive syntax
* [fg](https://blog.csdn.net/carolzhang8406/article/details/51314894): Run jobs in foreground
* find：1）寻找文件； 2）机械式操作
  * -iname：大小写不敏感

* [fuck](https://github.com/nvbn/thefuck#the-fuck-----):流行的纠正工具
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
* [icdiff](): 分屏比较文档
  * `icdiff button-{a,b}.css`
* ifconfig
#### j
* jobs
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
* [nc(netcat)](https://zhuanlan.zhihu.com/p/83959309): TCP/IP 的瑞士军刀
  * 端口测试
  * 
* nohup
#### o 
#### p
* [pandoc](https://github.com/jgm/pandoc)
  * `pandoc test1.md -f markdown -t html -s -o test1.html
`
  * `pandoc -s --toc -c pandoc.css -A footer.html MANUAL.txt -o example3.html`
* pbcopy: 复制到剪贴板 `pbcopy < file`
* ping
* pgrep: 配合jobs
  * `pgrep -f 100` 全命令行匹配
* pkill = pgrep + kill
  *`pkill -9 -f 100` 
* ps aux
* pwd: print cwd
#### q
#### r
#### s
* script
  * 记录终端操作记录，按`C-d`退出
  * 可用于demo演示终端操作
#### t
* tail
  * `ls -l | tail -n1`
  * -f：不断读最新内容，实时监视
* tee: Read from standard input and write to standard output and files (or commands).
  * 和`xargs`有相似之处，都是转化stdin用于他用
  * `echo "example" | tee /dev/tty | xargs printf "[%s]"`
* tig
  
  * >tig是一个基于ncurses的git文本模式接口。它的功能主要是作为一个Git存储库浏览器，但也可以帮助在块级别上分段提交更改，并充当各种Git命令输出的分页器。
* tmux 
* traceroute: -w 1
* tree: 显示树形文件结构, `-L`设置层数
#### u
#### v
#### w
* wait：`wait pid`，不加pid则等待所有进程
* which：找到程序路径
* [wget](https://blog.csdn.net/wangshuminjava/article/details/79916655): 断点续传-c    后台-b

#### x
* xargs：[解决命令的输入来源问题](https://blog.csdn.net/vanturman/article/details/84325846)：命令参数有标准输入和命令行参数两大来源，有的命令只接受命令行参数，需要xargs来转换标准输入
  * e.g. ` ls | xargs rm`
#### y
* youget
  * 自动补全：`curl -fLo ~/.zplug/repos/zsh-users/zsh-completions/src/_youget https://raw.githubusercontent.com/soimort/you-get/develop/contrib/completion/_you-get`
  * 
#### z