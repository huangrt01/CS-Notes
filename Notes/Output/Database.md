[toc]

## Database

没学过db。。。极速入门满足日常简单需求

https://www.runoob.com/sql/sql-tutorial.html

https://www.w3schools.com/sql/default.asp

```mysql
CREATE TABLE Persons (
    Personid int NOT NULL AUTO_INCREMENT,
    LastName varchar(255) NOT NULL,
    FirstName varchar(255),
    Age int,
    PRIMARY KEY (Personid)
);
```

* NOT NULL
  * [为什么数据库字段要使用 NOT NULL？](https://mp.weixin.qq.com/s/XyOU6dimeZNjzIYyTv_fPA)

* AUTO INCREMENT



* SELECT
  * max, min, avg
  * GROUP BY
    * [not a group by expression error](https://learnsql.com/blog/not-a-group-by-expression-error/)

```mysql
# 判断范围
select min(record_id),max(record_id) from service_status where service_type='abc' limit 10;
# 精确查询规模
explain select * from  service_status where service_type='abc' and (record_id between 1 and 100000) 
# 从后往前
select * from service_status order by id desc
# 不等于
<>
# 非空
is NOT NULL
```

e.g.

```mysql
SELECT log.uid, info.source, log.action, sum(log.action), COUNT(1) AS count
FROM info, log
WHERE (log.time in LAST_7D) 
    and (log.id = info.id) 
    and (log.action=show)
GROUP by log.uid, info.source
ORDER by log.uid, info.source DESC
LIMIT 1000
```

* WITH AS
* JOIN
  * https://stackoverflow.com/questions/354070/sql-join-where-clause-vs-on-clause

```mysql
with temp as(
	select a,b,c
  where a=.. AND b=.. AND c=..
)
select A.a
from temp A join temp B
on A.a = B.a
group by A.a
```

```mysql
# 高级 JOIN 技巧
# Using the Same Table Twice
SELECT a.account_id, e.emp_id, b_a.name open_branch, b_e.name emp_branch 
FROM account AS a 
INNER JOIN branch AS b_a ON a.open_branch_id = b_a.branch_id 
INNER JOIN employee AS e ON a.open_emp_id = e.emp_id 
INNER JOIN branch b_e ON e.assigned_branch_id = b_e.branch_id WHERE a.product_cd = 'CHK';

# Self-Joins
SELECT e.fname, e.lname, e_mgr.fname mgr_fname, e_mgr.lname mgr_lname
FROM employee AS e INNER JOIN employee AS e_mgr
ON e.superior_emp_id = e_mgr.emp_id;

# Non-Equi-Joins
SELECT e1.fname, e1.lname, 'VS' vs, e2.fname, e2.lname
FROM employee AS e1 INNER JOIN employee AS e2
ON e1.emp_id < e2.emp_id WHERE e1.title = 'Teller' AND e2.title = 'Teller';
```



* ALTER TABLE

```mysql
ALTER TABLE
  service_status
ADD
  error_count_avg decimal(10, 5) NOT NULL DEFAULT 0 COMMENT "服务错误量",
MODIFY
  error_code int(11) NOT NULL COMMENT "请求错误类型（取最多的）",
MODIFY
  is_error int(5) NOT NULL COMMENT "是否是错误请求";
  
ALTER TABLE test TABLE (c1 char(1),c2 char(1));
```



* TIMESTAMP

```mysql
`record_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '记录产生时间'
```



### db client

#### Redis 常见接口

- put
- get key or keys
- mget
  - Input: keys
  - Output: indices, results
- hset
  - e.g. Redis: https://redis.io/docs/reference/optimization/memory-optimization/#use-hashes-when-possible
  - 存储结构化数据
- remove key or keys
- zcard
  - Returns the sorted set cardinality (number of elements) of the sorted set stored at `key`.
- zremrangebyscore(key, start, end)
  - Removes all elements in the sorted set stored at `key` with a score between `min` and `max` (inclusive).
- sadd、zadd、hset
  - sadd：set
  - zadd：sorted set (element with score)
  - hset：https://redis.io/commands/hset/

### Hive

* hadoop 生态下的OLAP引擎，将 SQL 查询翻译为 map reduce 任务，特点是稳定，成功率高，但是查询速度慢

* API v.s. mysql

  * `percentile(a, array(0.5,0.9,0.99))` 求分位数（mysql没有这个函数）
  * [Hive Aggregate Functions](http://hadooptutorial.info/hive-aggregate-functions/)

* 用 Spark 做 mysql to Hive 的同步

  * ```
    spark.mysql.remain_delete = true
    ```

* 细节
  * p_date和p_hour是hive分区字段

### 特征工程

[滴滴特征工程](https://mp.weixin.qq.com/s/vUP4LAA7gAYDo91Wd5rSQQ)


### OLAP

DSL like ES -> (forward/aggregation/post-aggregation) -> view -> physical table

* 难点是异构数据源查询，使用 query engine 封装收口了所有 olap 引擎的查询，同时用服务聚合多数据源数据，格式化&聚合 后返回给用户，带来的挑战是研发同学需要熟悉各个数据源特性时延精确去重，异构数据源聚合等
* 指标杂乱，一张物理表几万个指标，配置分散在上十个元文件中
* [浅谈数据治理、数据管理、数据资源与数据资产管理内涵及差异点](https://mp.weixin.qq.com/s/B9t1ZdNEl8u0mhxDLyS8-A)

* [滴滴指标体系](https://mp.weixin.qq.com/s/-pLpLD_HMiasyyRxo5oTRQ)

### 论文

##### Spitfire: A Three-Tier Buffer Manager for Volatile and Non-Volatile Memory, SIGMOD 2021

介绍视频：https://www.bilibili.com/video/BV1164y1d7HD

![image-20210808000446554](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/Database/hymem.png)

![dram-nvm-ssd](https://raw.githubusercontent.com/huangrt01/Markdown-Transformer-and-Uploader/mynote/Notes/Database/dram-nvm-ssd.png)




nvm读写性能介于dram和ssd之间，更接近dram

2.Hymem

* 对比Spitfire：Hymem是single-threaded，且是NVM-aware的，对emulation有依赖
* clock algo逐出nvm、ssd
* the cache line-grained loading and mini-page optimizations must be tailored for a real NVM device. We also illustrate that the choice of the data migration policy is significantly more important than these auxiliary optimizations.
* 行为
  * DRAM admission: 如果没在DRAM中找到，eagerly SSD->DRAM，跳过 SSD->NVM
  * DRAM eviction: 用是否在 recent queue 中决定是否进 nvm
  * cache-line-grained page (256 cache lines): resident + dirty bitmap; 指向nvm page的指针
  * mini page (<16 cache lines): dirty bitmap + slots; count; full page

3.NVM-Aware Data Migration

* 目标：利用NVM提供的数据新通路，minimize the performance impact of NVM and to extend the lifetime of the NVM and SSD devices
  * lazy data migration from NVM to DRAM ensures that only hot data is promoted to DRAM.
  * 延续CLOCK策略，引入概率插入来决定migrate到哪一层存储
* 实现策略
  * Bypass DRAM during reads
    * lazily migrate data from NVM to DRAM
      while serving read operations.
    * ensures that warm pages on NVM do not evict hot pages in DRAM
  * bypass DRAM during writes
    * DBMSs use the group commit optimization to reduce this I/O overhead [9]. The DBMS first batches the log records for a group of transactions in the DRAM buffer (❹) and then flushes them together with a single write to SSD
    * Dw，减少hot pages在DRAM上的eviction
  * Bypass NVM during reads
    * a lazy policy for migrating data from NVM to DRAM (Dr = 0.01), and a comparatively eager policy while moving data from SSD to NVM (Nr = 0.2). While this scheme increases the number of writes to NVM compared to the lazy policy, it enables Spitfire to deliver higher performance than Hymem (§6.5)
    * This design reduces data duplication in the NVM buffer.
  * Bypass NVM during writes
    * 不同于hymem用recent queue，spitfire用随机插入来决定进入nvm的页

4.Adaptive Data Migration

* 相关论文：On multi-level exclusive caching: offline optimality and why promotions are better than demotions, FAST 2008
* 用模拟退火来搜索参数

5.System Architecture

5.1 Multi-Tier Buffer Management

* When a page is requested, Spitfire performs a table lookup that returns a shared page descriptor containing the locations (if any) of the logical page in the DRAM and NVM buffers.

* 数据结构：
  * {page_index -> shared_page_descriptor}
    * shared_page_descriptor: {latch_dram, latch_nvm, latch_ssd, dram_pd, nvm_pd}
      * dram_pd: {num_of_users, is_dirty, physical_pointer}

5.2 Concurrency Control and Recovery

* To support concurrent operations, we leverage the following data structures and protocols:
  * a concurrent hash table for managing the mapping from logical page identifiers to shared page descriptors [17]
  * a concurrent bitmap for the cache replacement policy [40]
  * multi-versioned timestamp-ordering (MVTO) concurrency control protocol [39]
  * concurrent B+Tree for indexing with optimistic lock-coupling [24]
  * lightweight latches for thread-safe page migrations
    * [数据库中 lock 和 latch 的区别](https://www.zhihu.com/question/309342903/answer/1699205097)

6.Experimental Evaluation

6.1 workload

* YCSB: Zipfian Distribution
* TPC-C

6.2 Benefits of NVM and App-Direct Mode

* memory-mode requires an upfront NVM capacity at least equal to the size of DRAM. In contrast, with app-direct mode, it could give a higher buffer capacity due to its cost advantage, though the NVM is bit slower. This is especially useful with a large working set.
* with app-direct mode, Spitfire exploits the persistence property of NVM to reduce the overhead of recovery protocol by eliminating the need to flush modified pages in NVM buffer.

6.3 Data Migration Policies

* With eager policies, more pages are updated in DRAM, and they must be flushed down to lower tiers of the storage system (even when the update is localized to a small chunk of the page). In contrast, with a lazy scheme, Spitfire updates page in NVM, thereby reducing write amplification.
* impact of storage hierarchy
  * DRAM/NVM的比例越大，migration probability对吞吐的影响越大。假如DRAM非常小，with the eager policy, the performance improvement brought by adding the comparatively smaller DRAM buffer (1.25 GB) is shadowed by the cost of data migration between DRAM and NVM

6.5 Revisiting Hymem’s Optimizations

* We attribute the 1.1× lower throughput at 64 B granularity (relative to 256B granularity) to the I/O amplification stemming from the mismatch between the device-level block size and loading granularity.

6.6 Storage System Design

* 指标：performance/price numbers

* insights	
  * To achieve the highest absolute performance, the hierarchy usually consists ofDRAM (since DRAM has the lowest latency).
  * If the workload is read-intensive, DRAM-NVM-SSD hierarchy is the best choice from a performance/price standpoint, since it is able to ensure the hottest data resides in DRAM.
  * If the workload is write-intensive, NVM-SSD hierarchy is the best choice from a performance/price standpoint, since NVM is able to reduce the recovery protocol overhead.



