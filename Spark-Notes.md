# Background Knowledge
## The structure of the normal big data platform
[大数据全体系年终总结](https://www.cnblogs.com/yangsy0915/p/6159756.html)
<br> 底层HDFS，上面跑MapReduce／Tez／Spark，在上面跑Hive，Pig。或者HDFS上直接跑Impala，Drill，Presto。这解决了中低速数据处理的要求。<br>

- [与 Hadoop 对比，如何看待 Spark 技术？](https://www.zhihu.com/question/26568496/answer/41608400)
- [带你入门Spark（资源整理)](https://zhuanlan.zhihu.com/p/22427880)
- [基于Spark的DL4J如何进行分布式的网络训练](https://deeplearning4j.org/cn/spark)

# Ecological System of Spark
![Ecological System](https://miro.medium.com/max/1250/1*z0Vm749Pu6mHdlyPsznMRg.png)<br>


# Spark Configuration
[SparkConf()](https://spark.apache.org/docs/1.6.0/api/java/org/apache/spark/SparkConf.html)
Configuration for a Spark application. Used to set various Spark parameters as key-value pairs.
Most of the time, you would create a SparkConf object with `new SparkConf()`, which will load values from any spark.* Java system properties set in your application as well. In this case, parameters you set directly on the SparkConf object take priority over system properties.

For unit tests, you can also call new SparkConf(false) to skip loading external settings and get the same configuration no matter what the system properties are.

All setter methods in this class support chaining. For example, you can write new SparkConf().setMaster("local").setAppName("My app").

Note that once a SparkConf object is passed to Spark, it is cloned and can no longer be modified by the user. Spark does not support modifying the configuration at runtime.

param: loadDefaults whether to also load values from Java system properties
# [Spark context](https://www.cnblogs.com/xia520pi/p/8609602.html)<br>
注释的第一句话就是说SparkContext为Spark的主要入口点，简明扼要，如把Spark集群当作服务端那Spark Driver就是客户端，SparkContext则是客户端的核心；
如注释所说 SparkContext用于连接Spark集群、创建RDD、累加器（accumlator）、广播变量（broadcast variables），所以说SparkContext为Spark程序的根本
都不为过。也就是说SparkContext是Spark的入口，相当于应用程序的main函数。目前在一个JVM进程中可以创建多个SparkContext，但是只能有一个active级别的。
如果你需要创建一个新的SparkContext实例，必须先调用stop方法停掉当前active级别的SparkContext实例。

# [Spark Session](https://blog.csdn.net/u013063153/article/details/54615378)
SparkConf、SparkContext和SQLContext都已经被封装在SparkSession当中。
[Difference between SparkContext, JavaSparkContext, SQLContext and SparkSession](https://stackoverflow.com/questions/43802809/difference-between-sparkcontext-javasparkcontext-sqlcontext-and-sparksession)

# Basic Operation
[map](https://www.zybuluo.com/jewes/note/35032)
map是对RDD中的每个元素都执行一个指定的函数来产生一个新的RDD。任何原RDD中的元素在新RDD中都有且只有一个元素与之对应。<br>
[flatMap](https://blog.csdn.net/YQlakers/article/details/73042098)
flatmap()是将函数应用于RDD中的每个元素，将返回的迭代器的所有内容构成新的RDD,这样就得到了一个由各列表中的元素组成的RDD,而不是一个列表组成的RDD。<br>

# RDD, DataSet, Dataframe
[Spark DataSet](https://www.jianshu.com/p/77811ae29fdd)
A Dataset can be constructed from JVM objects and then manipulated using functional transformations (map, flatMap, filter, etc.).


# Spark SQL
[Spark学习笔记——Spark SQL的操作实例](https://andone1cc.github.io/2017/03/05/Spark/sparksql/)
[spark-daria](https://github.com/MrPowers/spark-daria/?source=post_page---------------------------)
## Handling the timestamp
[DateTime Difference](https://docs.snowflake.net/manuals/sql-reference/functions/datediff.html)
[calculating-duration-by-subtracting-two-datetime-columns](https://stackoverflow.com/questions/30283415/calculating-duration-by-subtracting-two-datetime-columns-in-string-format/30315921)
[Window Functions](https://databricks.com/blog/2015/07/15/introducing-window-functions-in-spark-sql.html)


# [Spark MLib](https://spark.apache.org/mllib/)
## [Spark PipeLine](https://www.ibm.com/developerworks/cn/opensource/os-cn-spark-practice5/index.html)
## [DataTypes](https://spark.apache.org/docs/1.1.0/mllib-data-types.html)
### [StringIndexer- IndexToString-VectorIndexer](http://dblab.xmu.edu.cn/blog/1297-2/)
StringIndexer是指把一组字符型标签编码成一组标签索引，索引的范围为0到标签数量，索引构建的顺序为标签的频率，优先编码频率较大的标签，
所以出现频率最高的标签为0号。有的时候我们通过一个数据集构建了一个StringIndexer，然后准备把它应用到另一个数据集上的时候，
会遇到新数据集中有一些没有在前一个数据集中出现的标签，这时候一般有两种策略来处理：第一种是抛出一个异常（默认情况下），
第二种是通过掉用 setHandleInvalid("skip")来彻底忽略包含这类标签的行。
- index_transformers - (Using StringIndexer) - setInputCol, setOutputCol
- index_pipeline - setStages(index_transformers) - fit() - transform()
-- here the fit is to fit on ds1, transform is to apply on ds2. 
- index_Columns - filter to get the index columns
```scala
val index_transformers = featureCols.map(
 cname => new StringIndexer()
   .setInputCol(cname)
   .setOutputCol(s"${cname}_index")
)
```
### One_hot_encoders
```scala
val index_transformers: Array[org.apache.spark.ml.PipelineStage] = stringColumns.map(
  cname => new StringIndexer()
    .setInputCol(cname)
    .setOutputCol(s"${cname}_index")
)
```
### VectorAssembler
setInputCol, setOutputCol

从源数据中提取特征指标数据，这是一个比较典型且通用的步骤，因为我们的原始数据集里，经常会包含一些非指标数据，如 ID，Description 等。
为方便后续模型进行特征输入，需要部分列的数据转换为特征向量，并统一命名，VectorAssembler类完成这一任务。VectorAssembler是一个transformer，
将多列数据转化为单列的向量列。


# Spark UI
[Databricks](
https://databricks.com/blog/2015/06/22/understanding-your-spark-application-through-visualization.html) <br>

# [GraphX](https://www.cnblogs.com/wei-li/p/graphx.html)


# [Spark Jobs Tuning](https://www.zybuluo.com/xiaop1987/note/76737)

# Apache Zeppelin (https://www.iteblog.com/archives/1575.html)
[可视化分析工具Apache Zeppelin](https://blog.csdn.net/majianxiong_lzu/article/details/89838774)
