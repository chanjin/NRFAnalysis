

### Dataframe creation in PySpark

```Python
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *

sc = SparkContext(master="local[*]", appName="App")
sql_context = SQLContext(sc)

# create schema
fields = []
for i in range(1,14):
    numeric_features.append('n' + str(i))
for feature in numeric_features:
    fields.append(StructField(name=feature, dataType=StringType(), nullable=True))
schema = StructType(fields=fields)

# create dataframe from rdd
pyspark_dataframe = sql_context.createDataFrame(rdd, schema=schema)

# get pandas' dataframe from pyspark's dataframe
pandas_dataframe = pyspark_dataframe.toPandas()

# create dataframe from pandas' dataframe
pyspark_dataframe = sql_context.createDataFrame(pandas_dataframe)
```

### One-hot Encoding
One-hot Encoding이 필요한 경우, Spark의 [OneHotEncoder](http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.feature.OneHotEncoder)를 다음과 같이 이용할 수 있네요.
```Scala
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}

val df = sqlContext.createDataFrame(Seq(
  (0, "a"),
  (1, "b"),
  (2, "c"),
  (3, "a"),
  (4, "a"),
  (5, "c")
)).toDF("id", "category")

val indexer = new StringIndexer()
  .setInputCol("category")
  .setOutputCol("categoryIndex")
  .fit(df)
val indexed = indexer.transform(df)

val encoder = new OneHotEncoder()
  .setInputCol("categoryIndex")
  .setOutputCol("categoryVec")
val encoded = encoder.transform(indexed)
encoded.select("id", "categoryVec").show()
```

encoding이 되면 결과로 지정한 column에 sparse feature vector가 생기게 되는데, 모델 training을 위해서는 이 vector들을 merge 할 필요가 있습니다. merge는 [VectorAssembler](https://spark.apache.org/docs/latest/ml-features.html#vectorassembler)로 가능합니다.

```Scala
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.linalg.Vectors

val dataset = sqlContext.createDataFrame(
  Seq((0, 18, 1.0, Vectors.dense(0.0, 10.0, 0.5), 1.0))
).toDF("id", "hour", "mobile", "userFeatures", "clicked")

val assembler = new VectorAssembler()
  .setInputCols(Array("hour", "mobile", "userFeatures"))
  .setOutputCol("features")

val output = assembler.transform(dataset)
println(output.select("features", "clicked").first())
```

### Simple One hot encoding
범주로 정의된 문자열 배열로부터 각 문자열에 대한 Encoded Array를 만드는 방법
```scala
def onehotEncode(arr: Array[String]): Map[String, Array[Double]] = {
   arr.indices.map(i => {
     val a = new Array[Double](arr.size)
     a(i) = 1.0
     (arr(i), a)
   }).toMap
}
```
사용 방법
```scala
onehotEncode(Array("A", "B", "C", "C", "D").distinct)
res1: Map[String,Array[Double]] = Map(A -> Array(1.0, 0.0, 0.0, 0.0), B -> Array(0.0, 1.0, 0.0, 0.0), C -> Array(0.0, 0.0, 1.0, 0.0), D -> Array(0.0, 0.0, 0.0, 1.0))
```

### Excluding CSV header using MapPartitionWithIndex
CSV의 헤더를 제외하고 Data만 분석 (전체 범주를 알고있다는 가정)
``` scala 
val data = sc.textFile(path).mapPartitionsWithIndex((i, iterator) => 
  if (i == 0 && iterator.hasNext) {
    iterator.next;
    iterator
  } 
  else iterator
)
```


