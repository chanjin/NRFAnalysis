package termfiltering

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by chanjinpark on 2016. 9. 15..
  */
object TermFrequency extends App {

  /* TODO: 1. 많이 나오는 단어 분포 2. TFIDF로 중요 단어 분포
      중요하지 않으면서 자주 나오지 않은 단어를 제거
    */
  val conf = new SparkConf(true).setMaster("local").setAppName("NSFLDA")
  val sc = new SparkContext(conf)
  Logger.getLogger("org").setLevel(Level.ERROR)

  import org.apache.spark.sql.SparkSession
  import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
  import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}

  val spark = SparkSession
    .builder()
    .appName("Spark SQL Example")
    .config("spark.some.config.option", "some-value")
    .getOrCreate()


  val sqlContext = new org.apache.spark.sql.SQLContext(sc)

  import org.apache.spark.ml.linalg.{Vectors, VectorUDT, SparseVector, Vector}
  import org.apache.spark.ml.feature.LabeledPoint
  import org.apache.spark.rdd.RDD

  val dir = "/Users/chanjinpark/GitHub/NRFAnalysis/"

  val docs = sc.textFile(dir + "data/docs")
  val corpus = sc.textFile(dir + "data/corpus").map(_.split(","))

  case class MetaData(title: String, mainArea: Array[String], nationArea: Array[String], sixTArea: Array[String]) {
    override def toString: String = title + ":::" + mainArea.mkString(",") + ":::" +
      nationArea.mkString(",") + ":::" + sixTArea.mkString(",")
  }

  def getMetadata(s: String) = {
    val attr = s.split(":::")
    new MetaData(attr(0), attr(1).split(","), attr(2).split(","), attr(3).split(","))
  }

  val metadata = {
    scala.io.Source.fromFile(dir + "data/meta.txt").getLines().map(l => {
      val id = l.substring(0, l.indexOf("-"))
      val meta = getMetadata(l.substring(l.indexOf("-") + 1))
      (id, meta)
    }).toMap
  }

  val crb: Map[String, Int] = metadata.map(_._2.mainArea(1)).toList.distinct.sortWith(_.compare(_) < 0).zipWithIndex.toMap
  val classes = crb.map(_.swap)

  val vocab = corpus.flatMap(x => x).distinct.collect.zipWithIndex.toMap
  val matrix: RDD[Vector] = corpus.map {
    case tokens => {
      //val counts = new scala.collection.mutable.HashMap[Int, Double]()
      val vec = tokens.foldLeft(Map[Int, Double]())((res, t) => {
        val vocabid = vocab(t)
        res + (vocabid -> (res.getOrElse(vocabid, 0.0) + 1.0))
      })
      //val (indices, values) = vec.keys
      new SparseVector(vocab.size, vec.keys.toArray, vec.values.toArray)
    }
  }


  //val cvModel: CountVectorizerModel = new CountVectorizer().setInputCol("words").setOutputCol("features").setVocabSize(vocab.size).fit(sqlContext.createDataFrame(corpus.zipWithIndex()).toDF("words", "idx"))

  def getLabel(m: MetaData): Double = crb(m.mainArea(1)) // CRB 분류




}
