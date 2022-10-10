[toc]

# JAVA

### Environments

```shell
brew update
brew upgrade
brew install openjdk@17
brew install openjdk@11
brew install openjdk@8

java -version
```

* OpenJDK
  * Homebrew默认会将OpenJDK的包解压在 `/usr/local/Cellar/`下，并且在`/usr/local/opt/`创建软链接。
  * `/usr/local/opt/openjdk` 默认指向最新版本
  * [JAVA_HOME vs. PATH Environment Variables](https://tomgregory.com/java-home-vs-path-environment-variables/)
    * 最好都设，比如maven依赖JAVA_HOME

```shell
# 指定jdk版本，for maven
export JAVA_HOME=/usr/local/opt/openjdk@8
export PATH="/usr/local/opt/openjdk@8/bin:$PATH"
```

```shell
# /usr/bin/java 默认读取 /Library/Java/JavaVirtualMachines 中最高版本的 JDK

For the system Java wrappers to find this JDK, symlink it with
  sudo ln -sfn /usr/local/opt/openjdk@17/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk-17.jdk


openjdk@17 is keg-only, which means it was not symlinked into /usr/local,
because this is an alternate version of another formula.

# 通过环境变量设置
If you need to have openjdk@17 first in your PATH, run:
  echo 'export PATH="/usr/local/opt/openjdk@17/bin:$PATH"' >> ~/.zshrc

For compilers to find openjdk@17 you may need to set:
  export CPPFLAGS="-I/usr/local/opt/openjdk@17/include"
```

* IDE: IntelliJ IDEA Ultimate https://www.jetbrains.com/idea/
  * 调整 memory heap：https://www.jetbrains.com/help/idea/increasing-memory-heap.html，建议 2G 以上
  * 语法版本：File | Project Structure | Project | Language level：8
* IDE Plugin
  - 代码质量：[SonarLint](https://plugins.jetbrains.com/plugin/7973-sonarlint)
  - [Maven Helper](https://plugins.jetbrains.com/plugin/7179-maven-helper)
    * Right click in Editor | Run Maven 
      Right click in Project View Toolbar | (Run|Debug) Maven 
      CTRL + ALT + R - "Run Maven Goal" popup (you can use Delete key in the popup) 
      CTRL + SHIFT + ALT + R - "Run Maven Goal on Root Module" popup (you can use Delete key in the popup)  
      Customize goals: Settings | Other Settings | Maven Helper 
      Define shortcuts: Settings | Keymap | Plug-ins | Maven Helper  
      Open pom file, click on 'Dependency Analyzer' tab, right click in the tree for context actions.
* maven

```shell
brew install maven
# edit ~/.m2/settings.xml 全局配置maven
```

```shell
export MAVEN_OPTS="-Xms256m -Xmx512m -Xss10m"
# 堆内存初始值为256MB，最大值512MB
# xss: thread stack size
# https://stackoverflow.com/questions/4967885/jvm-option-xss-what-does-it-do-exactly

mvn package -DskipTests -pl my_target -am
# -am: If project list is specified, also build projects required by the list
# -e: 显示errors
```

* gradle
  * https://gradle.org/releases/

```shell
brew install gradle
```





### Basics

```shell
java -classpath
```

* [Extends vs Implements in Java](https://www.geeksforgeeks.org/extends-vs-implements-in-java/)
* [final Keyword in Java](https://www.geeksforgeeks.org/final-keyword-in-java/)
  * 用于 [create immutable class](https://www.geeksforgeeks.org/create-immutable-class-java/)

### Data Structures

#### Data Types

[convert string to long](https://howtodoinjava.com/java/string/convert-string-to-long/)

```java
String num = xxx;
long val = Long.parseLong(num);

String num1 = Long.toString(val);
```


```java
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

private final ReadWriteLock readWriteLock
            = new ReentrantReadWriteLock();
private final Lock writeLock
  					= readWriteLock.writeLock();
private final Lock readLock = readWriteLock.readLock();
```

