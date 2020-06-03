### Git

* [Pro Git](https://git-scm.com/book/en/v2)
* [MIT 6.NULL - git](https://missing.csail.mit.edu/2020/version-control/)
* [Learn Git Branching](https://learngitbranching.js.org/?locale=en_US)


#### VCS(version control system)
##### Git的理念
* distributed version control system (DVCS) 
  * VCS: 简单的管理，例如RCS
  * CVCS(centralized VCS)
  * DVCS: 每个clone都是一个backup，能处理不同的remote
* 操作模式
  *  本地操作、集中下载和上传
  *  The  Platonic  ideal  is  that  each  commit should compile and should move steadily towards more and more tests passing. 
*  ugly interface and beautiful design -> bottom-up地理解git

This [XKCD comic](https://xkcd.com/1597/) captures Git’s reputation:

<img src="git/git.png" alt="git" style="zoom:100%;" />

##### Git's data model
* Snapshots: 文件是blob，文件夹是tree，snapshot是top-level tree 
* Modeling history: relating snapshots
  * a history is a directed acyclic graph (DAG) of snapshots
  * 一个snapshot可能有多个parent，比如merge
  * snapshot被称作commit，是Git的精髓所在，不像其它VCS针对每个文件储存changes
* Data model, as pseudocode
* Objects and content-addressing 
  *  id由SHA-1 hash生成，40个十六进制字符
  * `git cat-file -p` 显示对象信息
* References：照顾可读性，和对象不同，它是mutable的
  * `master`表示主分支的最近commit
  * `HEAD`表示“where we currently are”
* repositories = objects + references 
```python
// a file is a bunch of bytes
type blob = array<byte>

// a directory contains named files and directories
type tree = map<string, tree | blob>

// a commit has parents, metadata, and the top-level tree
type commit = struct {
	parent: array<commit>
	author: string
	message: string
	snapshot: tree
}

type object = blob | tree | commit
objects = map<string, object>
def store(object):
    id = sha1(object) 
    objects[id] = object

def load(id):
    return objects[id]
```
##### Staging area
<img src="git/areas.png" alt="areas" style="zoom:80%;" />
* staging area也称作index
* 和`git add`相关
  * Git tracks changes to a developer’s codebase, but it’s necessary to stage and take a snapshot of the changes to include them  in the project’s history. `git add` performs staging, the first part  of that two-step process. Any changes that are staged will become a part of the next snapshot and a part of the project’s history. Staging and  committing separately gives developers complete control over the history of their project without changing how they code and work. 
* 意义在于摆脱snapshot和当前状态的绝对联系，使commit操作更灵活
* checkout在working directory之间切换；reset从staging area回复到working directory



#### Git command-line interface

##### Basics

- `git help <command>`: get help for a git command
  - `git command -h`: concise help 
  - [freenode](https://freenode.net/) 的`#git`和`#github`频道寻求帮助
- `git init`: creates a new git repo, with data stored in the `.git` directory
- `git status`: tells you what's going on
  - `git status -s`: 左列是staging area，右列是working tree 
- `git add <filename>`: adds files to staging area
  - 对于同一文件，add之后有新改动要重新add
- `git commit`: creates a new commit
  - Write [good commit messages](https://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html) and Even more reasons to write [good commit messages](https://chris.beams.io/posts/git-commit/): 大写开头，祈使句，简短
  - saves the snapshot to the project history and completes the  change-tracking process. In short, a commit functions like taking a photo. Anything that’s been staged with `git add` will become a part of the snapshot with `git commit`.
  - `git commit -am "m"`可以先add再commit，但前提是commit的文件都是tracked状态
- `git log`: shows a flattened log of history
  - `git log --pretty=format:"%h %s" --all --graph --decorate --no-merges`: visualizes history as a DAG
  - `git shortlog/ git log --oneline`: 只显示标题
  - `-p`显示全部信息，`-3`显示三条
  - `--stat`显示统计信息
  - `-S function_name --since=2.weeks --before="2008-11-01" --grep --author --committer --no-merges`
- `git diff`: 比较working directory和staging area
  - `git diff --staged'：比较staging area和last commit
  - `git diff <filename>`: show differences since the last commit
  - `git diff <revision> <filename>`: shows differences in a file between snapshots
  - `git difftool`，图形界面
- `git checkout <revision>`: updates HEAD and current branch
- `git rm file`
  - `git rm --cached`，只删除staging areas，不删除working tree
  - `git rm log/\*.log` ，通配符，注意要加`\`，Git有自己的文件名拓展
* `git mv file_from file_to`
```shell
mv README.md README
git rm README.md
git add README
```


##### Branching and merging
- `git branch`: shows branches
- `git branch <name>`: creates a branch
- `git checkout -b <name>`: creates a branch and switches to it
  - same as `git branch <name>; git checkout <name>`
- `git merge <revision>`: merges into current branch
- `git mergetool`: use a fancy tool to help resolve merge conflicts
- `git rebase`: rebase set of patches onto a new base

##### Remotes
- `git remote`: list remotes
- `git remote add <name> <url>`: add a remote
- `git push <remote> <local branch>:<remote branch>`: send objects to remote, and update remote reference
  * `git push origin lab1:lab1`
  * `git push --set-upstream origin my-branch`，本地关联远程分支，用来省略上面一行的分支标注
- `git branch --set-upstream-to=<remote>/<remote branch>`: set up correspondence between local and remote branch
- `git fetch`: retrieve objects/references from a remote
```shell
git fetch origin master:tmp
git diff tmp
git merge tmp
git branch -d tmp
```
- `git pull`: same as `git fetch; git merge`
- `git clone`: download repository from remote
  - 在最后可加文件夹名参数 

##### Undo
- `git commit --amend`: edit a commit's contents/message
- `git reset HEAD <file>`: unstage a file
  * `git reset --hard` 回到上次commit的版本，配合`git pull/push`
  * [Github如何回退敏感信息](https://help.github.com/en/github/authenticating-to-github/removing-sensitive-data-from-a-repository) 
```shell
git log
git reset --hard XXXXXXXX
git push origin HEAD --force # 回退remote敏感信息
```
- `git checkout -- <file>`: discard changes

##### Advanced Git
- `git cat-file -p`: 显示对象信息
  * 40位Hash值，前2位是文件夹，后38位是文件名
  * 存在`.git/objects/`中 
  * [理解git常用命令原理](http://www.cppblog.com/kevinlynx/archive/2014/09/09/208257.html)
- `git config`: Git is [highly customizable](https://git-scm.com/docs/git-config)
  - `/etc/gitconfig`对应`--system` 
  - `~/.gitconfig`或`~/.config/git/config`对应 `--global`
  - `path/.git/config`对应`--local`
  - `git config --list (--show-origin)`显示所有config
  - 设置Identity，见下面Github部分
- `git clone --depth=1`: shallow clone, without entire version history
- `git add -p`: interactive staging
- `git rebase -i`: interactive rebasing
- `git blame`: show who last edited which line
  - `git blame -L :collection _config.yml`
- `git stash`: temporarily remove modifications to working directory
	- `git stash show -p | git apply -R`
- `git bisect`: binary search history (e.g. for regressions)
- `git submodule add <url> /path`
    * clone之后初始化：`git submodule update --init --recursive`
    * 更新：`git submodule update --init --remote`
    * 如果报错already exists in the index ，用`git rm -r --cached /path`解决此问题 
    * 这个特性很适合和[dotfiles](https://github.com/huangrt01/dotfiles)搭配，但如果用在项目里可能[出现问题](https://codingkilledthecat.wordpress.com/2012/04/28/why-your-company-shouldnt-use-git-submodules/)，尤其是需要commit模块代码的时候
    * [使用时可能遇到的坑的集合](https://blog.csdn.net/a13271785989/article/details/42777793)
    * commit的时候有坑，需要先commit子模块，再commit主体，参考：https://stackoverflow.com/questions/8488887/git-error-changes-not-staged-for-commit
- `.gitignore`: [specify](https://git-scm.com/docs/gitignore) intentionally untracked files to ignore
  - `.gitignore_global`，[我的设定](https://github.com/huangrt01/dotfiles/blob/master/gitignore_global)

```
# ignore all .a files
*.a

# but do track lib.a, even though you're ignoring .a files above
!lib.a

# only ignore the TODO file in the current directory, not subdir/TODO
# /TODO

# ignore all files in any directory named build
# build/

# ignore doc/notes.txt, but not doc/server/arch.txt
# doc/*.txt

# ignore all .pdf files in the doc/ directory and any of its subdirectories
# doc/**/*.pdf
```

#### Miscellaneous
- **GUIs**: there are many [GUI clients](https://git-scm.com/downloads/guis)
out there for Git. We personally don't use them and use the command-line
interface instead.
- **Shell integration**: it's super handy to have a Git status as part of your
shell prompt ([zsh](https://github.com/olivierverdier/zsh-git-prompt),
[bash](https://github.com/magicmonty/bash-git-prompt)). Often included in
frameworks like [Oh My Zsh](https://github.com/ohmyzsh/ohmyzsh).
- **Editor integration**: similarly to the above, handy integrations with many
features. [fugitive.vim](https://github.com/tpope/vim-fugitive) is the standard
one for Vim.
- **Workflows**: we taught you the data model, plus some basic commands; we
didn't tell you what practices to follow when working on big projects (and
there are [many](https://nvie.com/posts/a-successful-git-branching-model/)
[different](https://www.endoflineblog.com/gitflow-considered-harmful)
[approaches](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow)).
- **GitHub**: Git is not GitHub. GitHub has a specific way of contributing code
to other projects, called [pull
requests](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests).
- **Other Git providers**: GitHub is not special: there are many Git repository
hosts, like [GitLab](https://about.gitlab.com/) and
[BitBucket](https://bitbucket.org/).

#### 和Github联动
* GitHub is a Git hosting repository that provides developers with tools to ship better code through command line features, issues (threaded discussions), pull requests, code review, or the use of a collection of free and for-purchase apps in the GitHub Marketplace. 
* [用SSH连GitHub](https://help.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh)
```shell
git config --global user.name "John Doe"
git config --global user.email johndoe@example.com

#MacOs
ssh-keygen -t rsa -b 4096 -C "huangrt01@163.com"
eval "$(ssh-agent -s)"
ssh-add -K ~/.ssh/id_rsa
pbcopy < ~/.ssh/id_rsa.pub  #适合MacOS , Linux用xclip
# 上github添加SSH Key
ssh -T git@github.com
#ssh-keygen -y -f ~/.ssh/id_rsa

#Linux

```
* 如果设置ssh key后，git push仍然要求输入邮箱密码
  * `git remote -v`查看origin使用的是https还是ssh
  * 如果是https，替换成ssh即可 `git remote set-url origin git@github.com:huangrt01/XXX.git`

* 建立仓库
```
git init
git remote add origin git@github.com:huangrt01@163.com/dotfiles.git
git pull --rebase origin master
git push --set-upstream origin master
```

#### Other Resources
- [Oh Shit, Git!?!](https://ohshitgit.com/) is a short guide on how to recover
from some common Git mistakes.
- [Git for Computer
Scientists](https://eagain.net/articles/git-for-computer-scientists/) is a
short explanation of Git's data model, with less pseudocode and more fancy
diagrams than these lecture notes.
- [Git from the Bottom Up](https://jwiegley.github.io/git-from-the-bottom-up/)
is a detailed explanation of Git's implementation details beyond just the data
model, for the curious.
- [How to explain git in simple
words](https://smusamashah.github.io/blog/2017/10/14/explain-git-in-simple-words)

* [git handbook](https://guides.github.com/introduction/git-handbook/)，里面有一些资源
* [完整doc文档](https://git-scm.com/docs)
* [resources to learn Git](https://try.github.io/)
* [如何fork一个私库](https://stackoverflow.com/questions/10065526/github-how-to-make-a-fork-of-public-repository-private)
* [tig](https://jonas.github.io/tig/doc/manual.html)：图形化git历史
  * >Tig是一个基于ncurses的git文本模式接口。它的功能主要是作为一个Git存储库浏览器，但也可以帮助在块级别上分段提交更改，并充当各种Git命令输出的分页器。
  * 先[安装ncurses](https://blog.csdn.net/weixin_40123831/article/details/82490687)
  * [使用指南](https://www.jianshu.com/p/d9f60c0abbf7)
