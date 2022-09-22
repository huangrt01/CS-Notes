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

### Flink 

streaming依赖kafka的exactly once 保障。用的是water mark机制来确定消费进度

#### 《Serving Machine Learning Models》chpt 4

http://kb.sites.apiit.edu.my/files/2018/12/ebook-serving-machine-learning-models.pdf

* 介绍了flink的特点：scalabality、checkpointing、state support、window semantics
* Process Function -> basic building blocks
  * Events (individual records within a stream)
  * State (fault-tolerant, consistent)
  * Timers (event time and processing time)
* Flink provides two ways of implementing low-level joins, key-based joins implemented by **CoProcessFunction**, and partition-based joins implemented by **RichCoFlatMapFunction**
  * Key-based Joins：类似于按模型分组
  * Partition-Based Joins：一个instance能serving所有model
    * 一个任务多个instances执行该任务的不同input data subset



```java
// Partition-Based Joins
public class DataProcessMap extends AbstractRichFunction implements CoFlatMapFunction<byte[], ModelConfig, Double> {
  
  public double map(byte[] rawValue) throws RuntimeException {
    
  }
  
  @Override
  public void flatMap1(byte[] bytes, Collector<double> collector) throws RuntimeException {
    readLock.lock();
    double score = map(bytes);
    readLock.unlock();
    collector.collect(score);
  }
  
  @Override
  public void flatMap2(ModelConfig modelConfig, Collector<double> collector) {
    writeLock.lock();
    this.modelConfig = modelConfig;
    logger.info("Use new model config: " + modelConfig.toString());
    writeLock.unlock();
  }
}

DataStream<byte[]> kafkaInstanceStream = env.addSource(kafka_source);
DataStream<double> myStream = kafkaInstanceStream.connect(configDataStream.broadcast())
                .flatMap(new DataProcessMap(curConfig));


```





```java
// DataStream.class
StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.getConfig().setGlobalJobParameters(params);
FlinkKafkaConsumer010<byte[]> kafka_source = Kafka010Utils.customFlinkKafkaConsumer010(
                cluster, topic, groupId, properties, new ByteArraySchema());
kafka_source.setXXX(...);
...
DataStream<byte[]> kafkaInstanceStream = env.addSource(kafka_source);

```



```java
// source function
import my_config.Config
import org.apache.flink.streaming.api.checkpoint.CheckpointedFunction;
import org.apache.flink.streaming.api.functions.source.SourceFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ConfigSource implements SourceFunction<Config>, CheckpointedFunction {
  private static final Logger logger = LoggerFactory.getLogger(ConfigSource.class);
  private Config config;
  private volatile boolean isRunning = true;
  private transient ListState<Config> checkpointed;
  public ConfigSource(Config config) {
    this.config = config;
  }
  
  @Override
  public void run(SourceContext<Config> sourceContext) {
    while (isRunnning) {
      // this synchronized block ensures that state checkpointing,
      // internal state updates and emission of elements are an atomic operation
      synchronized (sourceContext.getCheckpointLock()) {
				Set<Long> oldList = null;
        if (config.XXX() != null) {
          oldList = new HashSet<>(config.XXX());
        }
        boolean changed = false;
        Set<Long> list = GetXXX();
        if (!list.equals(oldList)) {
          changed = true;
        }
        if (changed) {
          sourceContext.collect(config);
        }
        
      }
      try {
        Thread.sleep(60000);
      } catch (InterruptedException e) {
        logger.error("Sleep interrupted: " + e);
      }
    }
  }
  
  @Override
  public void cancel() {
    isRunning = false;
  }
  
  @Override
  public void snapshotState(FunctionSnapshotContext functionSnapshotContext) throws Exception {
    this.checkpointed.clear();
    this.checkpointed.add(config);
  }
  
  @Override
  public void initializeState(FunctionInitializationContext functionInitializationContext) throws Exception {
    this.checkpointed = functionInitializationContext
      .getOperatorStateStore()
      .getListState(new ListStateDescriptor<>("config", Config.class));

    if (functionInitializationContext.isRestored()) {
      for (Config config: this.checkpointed.get()) {
        logger.info("Old config: " + config.toString());
      }
    }
  }
}
```



### 踩坑

* Compile error: error: scala.reflect.internal.MissingRequirementError: object java.lang.Object in compiler mirror not found.
  * solution: scala版本2.11，需要使用jdk8
  * Idea scale compiler: https://www.jetbrains.com/help/idea/compile-and-build-scala-projects.html

