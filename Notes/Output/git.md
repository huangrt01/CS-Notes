### Git
* Git的理念
  * distributed version control system (DVCS) 
  *  The  Platonic  ideal  is  that  each  commit should compile and should move steadily towards more and more tests passing. 
* [git handbook](https://guides.github.com/introduction/git-handbook/)，里面有一些资源
* [完整doc文档](https://git-scm.com/docs)
* [resources to learn Git](https://try.github.io/)
* [如何fork一个私库](https://stackoverflow.com/questions/10065526/github-how-to-make-a-fork-of-public-repository-private)

#### 重要命令
* `git clone` creates a local copy of a project that already exists remotely. The  clone includes all the project’s files, history, and branches.
  * --depth=1：clone速度慢的时候只clone最后一次commit 

* `git add` stages a change. Git tracks changes to a developer’s codebase, but it’s necessary to stage and take a snapshot of the changes to include them  in the project’s history. This command performs staging, the first part  of that two-step process. Any changes that are staged will become a part of the next snapshot and a part of the project’s history. Staging and  committing separately gives developers complete control over the history of their project without changing how they code and work.

* `git commit` saves the snapshot to the project history and completes the  change-tracking process. In short, a commit functions like taking a  photo. Anything that’s been staged with `git add` will become a part of the snapshot with `git commit`.
  * `git commit -am "m"`可以先add再commit，但前提是commit的文件都是tracked状态

* `git status` shows the status of changes as untracked, modified, or staged.

* `git branch` shows the branches being worked on locally.

* `git merge` merges lines of development together. This command is typically used to combine changes made on two distinct branches. For example, a developer would merge when they want to combine changes from a feature branch  into the master branch for deployment.

* `git pull` updates the local line of development with updates from its remote  counterpart. Developers use this command if a teammate has made commits  to a branch on a remote, and they would like to reflect those changes in their local environment.

* `git push` updates the remote repository with any commits made locally to a branch.
  * `git push origin lab1:lab1`
  * `git push --set-upstream origin my-branch`，本地关联远程分支，用来省略上面一行的分支标注

* `git submodule add <url> /path`
  * clone之后初始化：`git submodule update --init --recursive`
  * 更新：`git submodule update --init --remote`
  * 如果报错already exists in the index ，用`git rm -r --cached /path`解决此问题 
  * 这个特性很适合和[dotfiles](https://github.com/huangrt01/dotfiles)搭配，但如果用在项目里可能[出现问题](https://codingkilledthecat.wordpress.com/2012/04/28/why-your-company-shouldnt-use-git-submodules/)，尤其是需要commit模块代码的时候

#### 和Github联动
* GitHub is a Git hosting repository that provides developers with tools to ship better code through command line features, issues (threaded discussions), pull requests, code review, or the use of a collection of free and for-purchase apps in the GitHub Marketplace. 
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

* 回退敏感信息
```shell
git log
git reset --hard XXXXXXXX
git push origin HEAD --force
```

#### 其它
* [tig](https://jonas.github.io/tig/doc/manual.html)：图形化git历史
  * >Tig是一个基于ncurses的git文本模式接口。它的功能主要是作为一个Git存储库浏览器，但也可以帮助在块级别上分段提交更改，并充当各种Git命令输出的分页器。
  * 先[安装ncurses](https://blog.csdn.net/weixin_40123831/article/details/82490687)
  * [使用指南](https://www.jianshu.com/p/d9f60c0abbf7)
