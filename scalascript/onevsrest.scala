/**
  * Created by chanjinpark on 2016. 9. 12..
  */


import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
val dir = "/Users/chanjinpark/GitHub/NRFAnalysis/"


val docs = sc.textFile(dir + "data/docs")
val corpus = sc.textFile(dir + "data/corpus").map(_.split(","))

case class MetaData(title: String, mainArea: Array[String], nationArea: Array[String], sixTArea: Array[String]) {
  override def toString: String = title + ":::" + mainArea.mkString(",") + ":::" +
    nationArea.mkString(",") +  ":::" +  sixTArea.mkString(",")
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


import org.apache.spark.ml.classification.{OneVsRest, LogisticRegression}
import org.apache.spark.ml.util.MetadataUtils
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.DataFrame


val parsedData = docs.zip(matrix).map(d => {
  LabeledPoint(getLabel(metadata(d._1)), d._2.toDense)
}).map(sqlContext.createDataFrame(_)).randomSplit(Array(0.7, 0.25, 0.05))

val (training, test, eval) = (parsedData(0), parsedData(1), parsedData(2))

val model = NaiveBayes.train(training, lambda = 0.05, modelType = "multinomial")

val ovr = new OneVsRest()
ovr.setClassifier(model)