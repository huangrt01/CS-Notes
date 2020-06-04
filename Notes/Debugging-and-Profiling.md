## Debugging and Profiling

[MIT 6.NULL - Debugging and Profiling](https://missing.csail.mit.edu/2020/debugging-profiling/)

### Debugging

#### Printf Debugging and Logging
脑力劳动debug，借助打印信息思考和推断问题所在

信息除了Printf，还可以Logging，更灵活（可以输出到文件、sockets、remote servers），并且可复用

[Here](https://missing.csail.mit.edu/static/files/logger.py) is an example code that logs messages:

```shell
$ python logger.py
# Raw output as with just prints
python logger.py log
# Log formatted output
$ python logger.py log ERROR
# Print only ERROR levels and above
$ python logger.py color
# Color formatted output
```
利用颜色信息： [ANSI escape codes](https://en.wikipedia.org/wiki/ANSI_escape_code)

```shell
#!/usr/bin/env bash
for R in $(seq 0 20 255); do
    for G in $(seq 0 20 255); do
        for B in $(seq 0 20 255); do
            printf "\e[38;2;${R};${G};${B}m█\e[0m";
        done
    done
done
```

#### Third party logs

* UNIX系统中第三方库的log常存在`/var/log`
  * the [NGINX](https://www.nginx.com/) webserver places its logs under `/var/log/nginx`
* `systemd`, a system daemon that controls many things in your system such as which services are enabled and running 
  * `/var/log/journal`，可用`journalctl`显示
  * macOS上`var/log/system.log`，但更多人用system log，用 [`log show`](https://www.manpagez.com/man/1/log/) 显示
  * 参考[data wrangling]()，需要对log信息处理，也可以用[lvav](http://lnav.org/)，体验更好

```shell
logger "Hello Logs"
# On macOS
log show --last 1m | grep Hello
# On Linux
journalctl --since "1m ago" | grep Hello
```

#### Debuggers
* 共同命令：l(ist), s(tep), n(ext), b(reak), p(rint), r(eturn), q(uit), c(continue)

**Python**: [`ipdb`](https://pypi.org/project/ipdb/) is an improved `pdb` that uses the [`IPython`](https://ipython.org) REPL enabling tab completion, syntax highlighting, better tracebacks,  and better introspection while retaining the same interface as the `pdb` module.
* ipdb特有命令: `p locals()`, j(ump), pp([`pprint`](https://docs.python.org/3/library/pprint.html)), restart

**C++**: [`gdb`](https://www.gnu.org/software/gdb/) (and its quality of life modification [`pwndbg`](https://github.com/pwndbg/pwndbg)) and [`lldb`](https://lldb.llvm.org/)
* gdb特有命令: start, finish
* `gdb --args sleep 20`

[CS107 GDB and Debugging教程](https://web.stanford.edu/class/archive/cs/cs107/cs107.1202/resources/gdb)

[CS107 Software Testing Strategies](https://web.stanford.edu/class/archive/cs/cs107/cs107.1202/testing.html)

[lldb的使用](https://www.jianshu.com/p/9a71329d5c4d)

[macOS上配置VSCode的gdb调试环境](https://zhuanlan.zhihu.com/p/106935263?utm_source=wechat_session)  

#### Specialized Tools

Even if what you are trying to debug is a black box binary there are tools that can help you with that. Whenever programs need to perform actions that only the kernel can, they use [System Calls](https://en.wikipedia.org/wiki/System_call). There are commands that let you trace the syscalls your program makes. In Linux there’s [`strace`](https://www.man7.org/linux/man-pages/man1/strace.1.html) and macOS and BSD have [`dtrace`](http://dtrace.org/blogs/about/). `dtrace` can be tricky to use because it uses its own `D` language, but there is a wrapper called [`dtruss`](https://www.manpagez.com/man/1/dtruss/) that provides an interface more similar to `strace` (more details [here](https://8thlight.com/blog/colin-jones/2015/11/06/dtrace-even-better-than-strace-for-osx.html)).

* [strace入门](https://blogs.oracle.com/linux/strace-the-sysadmins-microscope-v2)

```shell
# On Linux
sudo strace (-e lstat) ls -l > /dev/null
# On macOS
sudo dtruss -t lstat64_extended ls -l > /dev/null
```
Under some circumstances, you may need to look at the network packets to figure out the issue in your program. Tools like [`tcpdump`](https://www.man7.org/linux/man-pages/man1/tcpdump.1.html) and [Wireshark](https://www.wireshark.org/) are network packet analyzers that let you read the contents of network packets and filter them based on different criteria.

For web development, the Chrome/Firefox developer tools are quite handy. They feature a large number of tools, including:

- Source code - Inspect the HTML/CSS/JS source code of any website.
- Live HTML, CSS, JS modification - Change the website content,  styles and behavior to test (you can see for yourself that website  screenshots are not valid proofs).
- Javascript shell - Execute commands in the JS REPL.
- Network - Analyze the requests timeline.
- Storage - Look into the Cookies and local application storage.

#### Static Analysis


**内存泄露问题**

* `valgrind`
* `gdb`

### Profiling