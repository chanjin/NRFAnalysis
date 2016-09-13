package examples

import org.apache.log4j.{Level, Logger}

import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by chanjinpark on 2016. 9. 13..
  */
object DTMultiLabel extends App {

  val conf = new SparkConf(true).setMaster("local").setAppName("NSFLDA")
  val sc = new SparkContext(conf)
  Logger.getLogger("org").setLevel(Level.ERROR)

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

  def getLabel(m: MetaData): Double = crb(m.mainArea(1)) // CRB 분류

  import org.apache.spark.ml.classification.{DecisionTreeClassifier}
  import org.apache.spark.mllib.evaluation.MulticlassMetrics

  val parsedData = sqlContext.createDataFrame(docs.zip(matrix).map(d => {
    LabeledPoint(getLabel(metadata(d._1)), d._2)
  })).toDF("label", "features").randomSplit(Array(0.7, 0.25, 0.05))

  val (training, test, eval) = (parsedData(0), parsedData(1), parsedData(2))

  val dt = new DecisionTreeClassifier().setLabelCol("label").setFeaturesCol("features")

  val model = dt.fit(training)
  val predictions = model.transform(test)

  val predictionsAndLabels = predictions.select("prediction", "label").map(row => (row.getDouble(0), row.getDouble(1)))

  val metrics = new MulticlassMetrics(predictionsAndLabels.rdd)

  metrics.labels.foreach { l =>
    println(s"Precision($l, ${ classes(l.toInt) }) = " + metrics.precision(l))
  }

  // Recall by label
  metrics.labels.foreach { l =>
    println(s"Recall($l, ${ classes(l.toInt) }) = " + metrics.recall(l))
  }
}
