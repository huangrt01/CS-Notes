[toc]
### CS144-Lab-Computer-Networking

#### Lab0: networking warmup
##### 1.配环境
设虚拟机，[实验指导书](https://stanford.edu/class/cs144/vm_howto/vm-howto-image.html#connect) ，可参考[Shell笔记](https://github.com/huangrt01/Markdown-Transformer-and-Uploader/blob/master/Notes/Output/Shell-MIT-6-NULL.md)，迁移dotfiles

```shell
sudo apt-get update
sudo apt install zsh
zsh --version

# log out and login back
echo $SHELL
$SHELL --version

sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

source ~/.zshrc

```

##### 2.Networking by Hand
2.1 Fetch a Web page

```shell
telnet cs144.keithw.org http
GET /hello HTTP/1.1 # path part,第三个slash后面的部分
Host: cs144.keithw.org # host part,`https://`和第三个slash之间的部分
```
* 返回的有[ETag](https://www.cnblogs.com/happy4java/p/11206015.html), 减少服务器带宽压力

```
HTTP/1.1 200 OK
Date: Sat, 23 May 2020 12:00:46 GMT
Server: Apache
X-You-Said-Your-SunetID-Was: huangrt01
X-Your-Code-Is: 582393
Content-length: 113
Vary: Accept-Encoding
Content-Type: text/plain

Hello! You told us that your SUNet ID was "huangrt01". Please see the HTTP headers (above) for your secret code.
```

```shell
netcat -v -l -p 9090
telnet localhost 9090
```

##### 3.Writing a network program using an OS stream socket
OS stream socket： ability to create areliable bidirectional in-order byte stream between two programs
* turn “best-effort datagrams” (the abstraction the Internet provides) into“reliable byte streams” (the abstraction that applications usually want)
