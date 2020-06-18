## Metaprogramming

[MIT 6.NULL - Metaprogramming](https://missing.csail.mit.edu/2020/metaprogramming/)

### Build systems

概念：dependencies, targets, rules

`make`和`Makefile`
* 根据modify time确定什么文件需要regenerate
```shell
target: prerequisite1 prerequisite2 ...
	command1
	command2 (需要tab)
```
```shell
paper.pdf: paper.tex plot-data.png
	pdflatex paper.tex

plot-%.png: %.dat plot.py
	./plot.py -i  <img src="https://www.zhihu.com/equation?tex=%2A.dat%20-o%20" alt="*.dat -o " class="ee_img tr_noresize" eeimg="1"> @
```


```shell
# specify all source files here
SRCS = hw.c helper.c

# specify target here (name of executable)
TARG = hw

# specify compiler, compile flags, and needed libs
CC   = gcc
OPTS = -Wall -O
LIBS = -lm

# this translates .c files in src list to .o’s
OBJS = $(SRCS:.c=.o)

# all is not really needed, but is used to generate the target, default directive
all: $(TARG)

# this generates the target executable
 <img src="https://www.zhihu.com/equation?tex=%28TARG%29%3A%20" alt="(TARG): " class="ee_img tr_noresize" eeimg="1"> (OBJS)
	 <img src="https://www.zhihu.com/equation?tex=%28CC%29%20-o%20" alt="(CC) -o " class="ee_img tr_noresize" eeimg="1"> (TARG)  <img src="https://www.zhihu.com/equation?tex=%28OBJS%29%20" alt="(OBJS) " class="ee_img tr_noresize" eeimg="1"> (LIBS)
	
# this is a generic rule for .o files
%.o: %.c
   <img src="https://www.zhihu.com/equation?tex=%28CC%29%20" alt="(CC) " class="ee_img tr_noresize" eeimg="1"> (OPTS) -c  <img src="https://www.zhihu.com/equation?tex=%3C%20-o%20" alt="< -o " class="ee_img tr_noresize" eeimg="1"> @

# and finally, a clean line
.PHONY: clean
clean:
	rm -f  <img src="https://www.zhihu.com/equation?tex=%28OBJS%29%20" alt="(OBJS) " class="ee_img tr_noresize" eeimg="1"> (TARG)
```
* 第一个directive是default goal
* indent后面的命令是建立依赖的语句
* [.PHONY](https://www.gnu.org/software/make/manual/html_node/Phony-Targets.html)避免clean和名为clean的文件冲突

```shell
SUBDIRS = foo bar baz

.PHONY: subdirs $(SUBDIRS)

subdirs: $(SUBDIRS)

$(SUBDIRS):
         <img src="https://www.zhihu.com/equation?tex=%28MAKE%29%20-C%20" alt="(MAKE) -C " class="ee_img tr_noresize" eeimg="1"> @

foo: baz
```

* 配合[git ls-files](https://git-scm.com/docs/git-ls-files)，写make的[标准targets](https://www.gnu.org/software/make/manual/html_node/Standard-Targets.html#Standard-Targets)
* 可以利用`.git/hooks`中的[`pre-commit`](https://git-scm.com/docs/githooks#_pre_commit)在每次commit之前make特定的文件

### Dependency management
概念：repository, versioning, version number, [semantic versioning](https://semver.org/)

- If a new release does not change the API, increase the patch version.
- If you *add* to your API in a backwards-compatible way, increase the minor version.
- If you change the API in a non-backwards-compatible way, increase the major version.
- [Rust's build system](https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html)，可帮助理解版本号以及dependency管理

lock file: a file that lists the exact version you are *currently* depending on of each dependency

* vendoring: copy all the code of your dependencies into your own project

makedepend工具能帮助寻找依赖

### Continuous integration(CI) systems

“stuff that runs whenever your code changes”

* e.g. Travis CI, Azure Pipelines, and GitHub Actions
* Pages is a CI action that runs the Jekyll blog software on every push to `master` and makes the built site available on a particular GitHub domain

**testing**

- Test suite: a collective term for all the tests
- Unit test: a “micro-test” that tests a specific feature in isolation
- Integration test: a “macro-test” that runs a larger part of the system to check that different feature or components work *together*.
- Regression test: a test that implements a particular pattern that *previously* caused a bug to ensure that the bug does not resurface.
- Mocking: the replace a function, module, or type with a fake implementation to avoid testing unrelated functionality. For example, you might “mock the network” or “mock the disk”.

### Github Pages
1. Set up a simple auto-published page using [GitHub Pages](https://help.github.com/en/actions/automating-your-workflow-with-github-actions). Add a [GitHub Action](https://github.com/features/actions) to the repository to run `shellcheck` on any shell files in that repository (here is [one way to do it](https://github.com/marketplace/actions/shellcheck)). Check that it works!
2. [Build your own](https://help.github.com/en/actions/automating-your-workflow-with-github-actions/building-actions) GitHub action to run [`proselint`](http://proselint.com/) or [`write-good`](https://github.com/btford/write-good) on all the `.md` files in the repository. Enable it in your repository, and check that it works by filing a pull request with a typo in it.



个人blog的建立：

调研之后决定用动态blog，[halo](https://halo.run/)是傻瓜式操作，[在阿里云上部署](https://blog.csdn.net/weixin_43160252/article/details/104864279)




