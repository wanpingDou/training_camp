# spark 读取parquet

列式存储布局（比如 Parquet）可以加速查询，因为它只检查所有需要的列并对它们的值执行计算，因此只读取一个数据文件或表的小部分数据。Parquet 还支持灵活的压缩选项，因此可以显著减少磁盘上的存储。

如果您在 HDFS 上拥有基于文本的数据文件或表，而且正在使用 Spark SQL 对它们执行查询，那么强烈推荐将文本数据文件转换为 Parquet 数据文件，以实现性能和存储收益。当然，转换需要时间，但查询性能的提升在某些情况下可能达到 30 倍或更高，存储的节省可高达 75%！

## 1. 读取parquet

```python
val userDF = spark.read.parquet("file:///usr/local/Cellar/spark-2.3.0/examples/src/main/resources/users.parquet")
```

可以直接read一个parquet文件，就转成了dataframe。因为parquet文件里有比较丰富的信息，不像普通的文件。所以推荐是把其他文件的格式，清洗后转换成parquet数据格式。

## 2. 把dataframe 转成 parquet 文件

```python
val jsonPeopleDF = spark.read.json("/usr/local/Cellar/spark-2.3.0/examples/src/main/resources/people.json")
 jsonPeopleDF.write.parquet("/Users/walle/Documents/D3/d1.parquet")
val d1DF = spark.read.parquet("file:///Users/walle/Documents/D3/d1.parquet")
d1DF.show(5)
```

### [文章来源](http://www.waitingfy.com/archives/4334)