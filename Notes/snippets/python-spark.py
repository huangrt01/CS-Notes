from pyspark.sql import SparkSession

# 初始化 SparkSession
spark = SparkSession.builder \
    .appName("Read HDFS JSON") \
    .getOrCreate()

# 读取 HDFS 上的 JSON 文件
# 注意：这里的路径 'hdfs://tt/...' 必须是您的集群可以访问的有效路径
df = spark.read.json('hdfs://xxx.ztsd')

# 打印 Schema 和显示数据
df.printSchema()
df.show()

# 停止 SparkSession
spark.stop()