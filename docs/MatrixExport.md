
```scala
object MatrixExport extends PreProcessing {
  def main(args: Array[String]) : Unit = {
    import java.io._

    val conf = new SparkConf(true).setMaster("local").setAppName("NSFLDA")
    val sc = new SparkContext(conf)
    Logger.getLogger("org").setLevel(Level.ERROR)

    val (docs, vocab, matrix) = getInputData(sc)

    println(docs.count)
    println(vocab.size)

    matrix.saveAsTextFile("data/matrix")
    docs.saveAsTextFile("data/docs")
    sc.parallelize(vocab.map(kv=> kv._1 + ":" + kv._2).toSeq).saveAsTextFile("data/vocab")
  }
}

object MatrixLoader {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf(true).setMaster("local").setAppName("NSFLDA")
    val sc = new SparkContext(conf)
    Logger.getLogger("org").setLevel(Level.ERROR)

    def parse(rdd: RDD[String]): RDD[(Long, Vector)] = {
      val pattern: scala.util.matching.Regex = "\\(([0-9]+),(.*)\\)".r
      rdd .map{
        case pattern(k, v) => (k.toLong, Vectors.parse(v))
      }
    }

    val tfidf = parse(sc.textFile("data/corpus/part-*"))
  }
}
```