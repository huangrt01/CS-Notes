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

NOT NULL

* [为什么数据库字段要使用 NOT NULL？](https://mp.weixin.qq.com/s/XyOU6dimeZNjzIYyTv_fPA)

AUTO INCREMENT



SELECT

```mysql
# 判断范围
select min(record_id),max(record_id) from service_status where service_type='abc' limit 10;
# 精确查询规模
explain select * from  service_status where service_type='abc' and (record_id between 1 and 100000) 
# 从后往前
select * from service_status order by id desc
# 不等于
<>
```



ALTER TABLE

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



TIMESTAMP

```mysql
`record_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '记录产生时间'
```



### 特征工程

[滴滴特征工程](https://mp.weixin.qq.com/s/vUP4LAA7gAYDo91Wd5rSQQ)


### OLAP

DSL like ES -> (forward/aggregation/post-aggregation) -> view -> physical table

* 难点是异构数据源查询，使用 query engine 封装收口了所有 olap 引擎的查询，同时用服务聚合多数据源数据，格式化&聚合 后返回给用户，带来的挑战是研发同学需要熟悉各个数据源特性时延精确去重，异构数据源聚合等
* 指标杂乱，一张物理表几万个指标，配置分散在上十个元文件中
* [浅谈数据治理、数据管理、数据资源与数据资产管理内涵及差异点](https://mp.weixin.qq.com/s/B9t1ZdNEl8u0mhxDLyS8-A)

* [滴滴指标体系](https://mp.weixin.qq.com/s/-pLpLD_HMiasyyRxo5oTRQ)



