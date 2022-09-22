### Vim

[Editor War](https://en.wikipedia.org/wiki/Editor_war)

[Stack Overflow survey](https://insights.stackoverflow.com/survey/2019/#development-environments-and-tools)

[Vim emulation for VS code](https://github.com/VSCodeVim/Vim)：个人觉得Vim+VSCode是绝配，Vim负责文本编辑，VSCode负责插件等高级功能

[在VSCode中使用Vim的正确方式](https://zhuanlan.zhihu.com/p/141248420)

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
  * `:E` 打开netrw文件浏览器
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
    * `/\cword`不区分大小写查找

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
  * vim plugin: undotree[https://github.com/mbbill/undotree] 
  * [how to persistently undo](https://vim.fandom.com/wiki/Using_undo_branches#Persistent_undo)
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
./vimrc: [课程推荐config](https://missing.csail.mit.edu/2020/files/vimrc), instructors’ Vim configs ([Anish](https://github.com/anishathalye/dotfiles/blob/master/vimrc), [Jon](https://github.com/jonhoo/configs/blob/master/editor/.config/nvim/init.vim) (uses [neovim](https://neovim.io/)), [Jose](https://github.com/JJGO/dotfiles/blob/master/vim/.vimrc))

plugin: [推荐网站](https://vimawesome.com/)，git clone到`~/.vim/pack/vendor/start/`
* [ctrlp.vim](https://github.com/ctrlpvim/ctrlp.vim): fuzzy file finder

* [ack.vim](https://github.com/mileszs/ack.vim): code search

* [nerdtree](https://github.com/scrooloose/nerdtree): file explorer
  
  * 手动输`vim -u NONE -c "helptags ~/.vim/pack/my_plugs/start/nerdtree/doc" -c q`激活帮助插件，配置在了我的dotfiles里
  
* [vim-easymotion](https://github.com/easymotion/vim-easymotion): magic motions

* [ale](https://github.com/dense-analysis/ale#installation-with-vim-plug)

  * `:ALEGoToDefinition`
  * `:ALEFindReferences`
  * `:ALEHover`
  * `:ALESymbolSearch`


##### Vim-mode的其它应用
* Shell：If you’re a Bash user, use `set -o vi`. If you use Zsh, `bindkey -v`. For Fish, `fish_vi_key_bindings`. Additionally, no matter what shell you use, you can `export EDITOR=vim`. This is the environment variable used to decide which editor is launched when a program wants to start an editor. For example, `git` will use this editor for commit messages.

* Readline: Many programs use the [GNU Readline](https://tiswww.case.edu/php/chet/readline/rltop.html) library for their command-line interface. 
  * Readline supports (basic) Vim emulation too, which can be enabled by adding the following line to the `~/.inputrc` file: `set editing-mode vi`
  * With this setting, for example, the Python REPL will support Vim bindings.

* Others:

  There are even vim keybinding extensions for web [browsers](http://vim.wikia.com/wiki/Vim_key_bindings_for_web_browsers), some popular ones are [Vimium](https://chrome.google.com/webstore/detail/vimium/dbepggeogbaibhgnhhndojpepiihcmeb?hl=en) for Google Chrome and [Tridactyl](https://github.com/tridactyl/tridactyl) for Firefox. You can even get Vim bindings in [Jupyter notebooks](https://github.com/lambdalisue/jupyter-vim-binding).

##### Vim的其它特性积累

Q: 如果文件太大打不开怎么办？

A: 先`grep -nr $target $FILE`获取行号，然后`vim $FILE +N`进入定位所在。



* 缩进操作：
  * 先 `v` 进入visual模式选定要缩进的内容，再 `shift + </>` 进行整体的左右缩进
  * 对齐缩进：`v`选中第二、三行，然后`=`，与第一行对齐缩进



buffer操作：` :ls,:b num, :bn（下一个）, :bp（前一个）, :b#(上次的buffer) `

window操作： `:sp / :vsp` split window，`C-w + hjkl`切换

tab操作：`gt`切换tab

跳转操作：

* `Ctrl-O/I` 进入更旧/新的位置

* Marks - In vim, you can set a mark doing `m<X>` for some letter `X`. You can then go back to that mark doing `'<X>`. This let’s you quickly navigate to specific locations within a file or even across files.

查找替换：
* `:%s/foo/bar/g`
  * replace foo with bar globally in file
  * `%`表示修改全文件而不是第一个匹配串，`/g`表示全行/全文件匹配，`/gc`会提示每个匹配串是否替换 
  * `:#,#s/...`表示对行号之间的内容操作
* ` :%s/\[.*\](\(.*\))/\1/g`
  
  - replace named Markdown links with plain URLs
  

命令操作：
* `:g/pattern/command`，对匹配行执行命令
* `.`复制操作
* 外部命令：`:!`

Advanced Text Objects - Text objects like searches can also be composed with vim commands. E.g. `d/<pattern>` will delete to the next match of said pattern or `cgn` will change the next occurrence of the last searched string.





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

<img src="Vim/vim.png" alt="vim" style="zoom:100%;" />


##### Resources

- `vimtutor` is a tutorial that comes installed with Vim
- [Vim Adventures](https://vim-adventures.com/) is a game to learn Vim
- [Vim Tips Wiki](http://vim.wikia.com/wiki/Vim_Tips_Wiki)
- [Vim Advent Calendar](https://vimways.org/2019/) has various Vim tips
- [Vim Golf](http://www.vimgolf.com/) is [code golf](https://en.wikipedia.org/wiki/Code_golf), but where the programming language is Vim’s UI
- [Vi/Vim Stack Exchange](https://vi.stackexchange.com/)
- [Vim Screencasts](http://vimcasts.org/)
- [Practical Vim](https://pragprog.com/book/dnvim2/practical-vim-second-edition) (book)

### Sublime Text

Ctrl + M 括号匹配

Shift + Ctrl + M 选中该级括号内容