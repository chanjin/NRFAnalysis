package examples

/**
  * Created by chanjinpark on 2016. 9. 14..
  */

import org.apache.log4j.{Level, Logger}

// import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.{SparkConf, SparkContext}

object NBAnalysisDFMC extends App {
  val conf = new SparkConf(true).setMaster("local").setAppName("NSFLDA")
  val sc = new SparkContext(conf)
  Logger.getLogger("org").setLevel(Level.ERROR)

  import org.apache.spark.sql.SparkSession

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

  import org.apache.spark.ml.classification.{OneVsRest, NaiveBayes, LogisticRegression}

  val parsedData = sqlContext.createDataFrame(docs.zip(matrix).map(d => {
    (getLabel(metadata(d._1)), d._2)
  })).toDF("label", "features").randomSplit(Array(0.7, 0.25, 0.05))

  val (training, test, eval) = (parsedData(0), parsedData(1), parsedData(2))

  import spark.implicits._

  val models = classes.map(c => {
    val data = training.select("label", "features").map(t => {
      val l = t.getDouble(0)
      LabeledPoint(if (l == c._1) 1.0 else 0.0, t.getAs[Vector](1))
    })
    val model = new NaiveBayes().setModelType("multinomial").setSmoothing(0.05)
    (c._1, model.fit(data))
  })

  import org.apache.spark.ml.linalg.DenseVector
  val predictions = models.map(m => {
    val data = test.select("label", "features").map(t => {
      val l = t.getDouble(0)
      LabeledPoint(if (l == m._1) 1.0 else 0.0, t.getAs[Vector](1))
    })
    m._2.transform(data).select("prediction", "label", "probability").map(row => (row.getDouble(0), row.getDouble(1), row.getAs[DenseVector](2)))
  })


  predictions.map(p => p.take(5)).foreach(s => {
    s.map(x => x._1 + ":" + x._2 + " - " + x._3.toArray.map(y => f"$y%1.3f").mkString(", ")).foreach(println)
    println
  })



}
