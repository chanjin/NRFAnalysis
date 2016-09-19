package examples

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import viz.ClassificationResult

/**
  * Created by chanjinpark on 2016. 9. 15..
  */

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.ml.linalg.{Vectors, VectorUDT, SparseVector, Vector}
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.rdd.RDD


object DataFrameViz {

  val conf = new SparkConf(true).setMaster("local").setAppName("NSFLDA")
  val sc = new SparkContext(conf)
  Logger.getLogger("org").setLevel(Level.ERROR)
  val spark = SparkSession
    .builder()
    .appName("Spark SQL Example")
    .config("spark.some.config.option", "some-value")
    .getOrCreate()

  val sqlContext = spark.sqlContext

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

  /*
    val termfreq = corpus.flatMap(x => {
      x.foldLeft(Map[String, Int]())((res, cnt) => res + (cnt -> (res.getOrElse(cnt, 0) + 1)))
    }).reduceByKey(_ + _).toDF("term", "freq")
    termfreq.registerTempTable("termfreq")
   */
  def wordDistribution = {
    val termfreq = corpus.flatMap(x => {
      x.foldLeft(Map[String, Int]())((res, cnt) => res + (cnt -> (res.getOrElse(cnt, 0) + 1)))
    }).reduceByKey(_ + _)
    termfreq.sortBy(_._2)
  }

  def getLabel(m: MetaData): Double = crb(m.mainArea(1)) // CRB 분류

  /*
    tfByArea.map(
   */
  def WordByCRBDistribution = {
    val termsByArea = docs.map(d => (d, getLabel(metadata(d)))).zip(corpus)
    val tfByArea = termsByArea.groupBy(ta => ta._1._2).flatMap(g => {
      val tf = g._2.map(_._2).flatMap(x => {
        x.foldLeft(Map[String, Int]())((res, cnt) => res + (cnt -> (res.getOrElse(cnt, 0) + 1)))
      }).foldLeft(Map[String, Int]())((res, item) => res + (item._1 -> (res.getOrElse(item._1, 0) + 1)))
      tf.map(f => (g._1, f._1, f._2))
    })
    tfByArea
  }


  def alldata = {
    val documents = docs.map(d => (d, getLabel(metadata(d)).toInt)).zip(corpus).map(x => (x._1._1, x._1._2, x._2) )
  }



  def main(args: Array[String]) : Unit = {
    val doccorpus = sqlContext.createDataFrame(docs.zip(matrix)).toDF("docid", "corpus")

    //val cvModel: CountVectorizerModel = new CountVectorizer().setInputCol("words").setOutputCol("features").setVocabSize(vocab.size).fit(sqlContext.createDataFrame(corpus.zipWithIndex()).toDF("words", "idx"))



    import org.apache.spark.ml.classification.{OneVsRest, NaiveBayes, LogisticRegression}

    val parsedData = sqlContext.createDataFrame(docs.zip(matrix).map(d => {
      LabeledPoint(getLabel(metadata(d._1)), d._2)
    })).toDF("label", "features").randomSplit(Array(0.7, 0.25, 0.05))

    val (training, test, eval) = (parsedData(0), parsedData(1), parsedData(2))

    // instantiate the One Vs Rest Classifier.
    val clf = new NaiveBayes()
    clf.setModelType("multinomial")
    clf.setSmoothing(0.05)

    val model = clf.fit(training)
    val predictions = model.transform(test)

    import spark.implicits._

    //predictions.columns

    val predictionsAndLabels = predictions.select("prediction", "label").map(row => (row.getDouble(0), row.getDouble(1)))
    predictionsAndLabels.take(10).foreach(println)


    import org.apache.spark.ml.linalg.DenseVector
    val pl = predictions.select("prediction", "label", "probability").map(row => (row.getDouble(0), row.getDouble(1), row.getAs[DenseVector](2)))
    pl.take(100).map(x => x._1 + ":" + x._2 + " - " + x._3.toArray.map(y => f"$y%1.3f").mkString(", ")).foreach(println)

    val metrics = new MulticlassMetrics(predictionsAndLabels.rdd)


    ClassificationResult.saveMulticlassMetrics(dir + "data/temp/", metrics, classes)
  }
}
