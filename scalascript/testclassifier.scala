/**
  * Created by chanjinpark on 2016. 9. 11..
  */


import java.io._

import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.linalg.{DenseVector, Vector}
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

val meta = {
  scala.io.Source.fromFile(dir + "data/meta.txt").getLines().map(l => {
    val id = l.substring(0, l.indexOf("-"))
    val meta = getMetadata(l.substring(l.indexOf("-") + 1))
    (id, meta)
  }).toMap
}

val crb: Map[String, Int] = meta.map(_._2.mainArea(1)).toList.distinct.sortWith(_.compare(_) < 0).zipWithIndex.toMap
val classes = crb.map(_.swap)


import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.feature.IDF

val hashingTF = new HashingTF()
val tf: RDD[Vector] = hashingTF.transform(corpus.map(c => c.toIndexedSeq))
val idf = new IDF().fit(tf)
val tfidf = idf.transform(tf)

def getLabel(m: MetaData): Double = crb(m.mainArea(1)) // CRB 분류

val parsedData = docs.zip(tfidf).map(d => {
  (getLabel(meta(d._1)), d._2.toDense)
}).randomSplit(Array(0.8, 0.2))

val (training, test) = (parsedData(0), parsedData(1))
val lambdaval = 1.0
val models = classes.map(c => {
  val data : RDD[LabeledPoint] = training.map(t => LabeledPoint(if (t._1 == c._1.toDouble) 1.0 else 0.0, t._2))
  val model = NaiveBayes.train(data, lambda = lambdaval, modelType = "multinomial")
  (c._1, model)
})
models.foreach(_.cache())

val predLabels = models.map(m => {
  val data = test.map(t => LabeledPoint(if (t._1 == m._1) 1.0 else 0.0, t._2))
  (m._1, data.map(p => (m._2.predict(p.features), p.label)))
})

predLabels.foreach(_.cache())

def logLoss(labelAndPreds: RDD[(Double, Double)]): Double= {
  import breeze.numerics.log
  import breeze.linalg._
  //val logloss = (p:Double, y:Double) => - ( y * log(p) + (1-y) * log( 1 - p) )
  val maxmin = (p:Double) => max(min(p, 1.0 - 1e-14), 1e-14)
  val logloss = (p:Double, y:Double) => - (y * log(maxmin(p)) + (1-y) * log(1 - maxmin(p)))

  val loglossErr = labelAndPreds.map {
    case (label, pred) => logloss(pred, label)
  }.sum() / labelAndPreds.count()

  loglossErr
}

def MSE(labelAndPreds: RDD[(Double, Double)]): Double = {
  labelAndPreds.map{case(v, p) => math.pow(v - p, 2)}.mean()
}

import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}

predLabels.foreach(pl => {
  println(pl._1 + ": " + classes(pl._1))

  val metrics = new BinaryClassificationMetrics(pl._2)
  // Precision by threshold
  val precision = metrics.precisionByThreshold
  precision.foreach { case (t, p) =>
    println(s"Threshold: $t, Precision: $p")
  }

  // Recall by threshold
  val recall = metrics.recallByThreshold
  recall.foreach { case (t, r) =>
    println(s"Threshold: $t, Recall: $r")
  }

  val f1Score = metrics.fMeasureByThreshold
  f1Score.foreach { case (t, f) =>
    println(s"Threshold: $t, F-score: $f, Beta = 1")
  }

  /*val mse = MSE(pl.map(_.swap))
  println("training Mean Squared Error = " + mse)

  val loglossErr = logLoss(pl.map(_.swap))
  println("Log Loss Error = " + loglossErr)
  // Save and load model
  */
})



def predictMulticlass(label: Int, features: DenseVector) = {
  val ps = models.map(m => (m._1, m._2.predict(features)))
  ps.tail.foldLeft(ps.head)((r, x) => {
    if (r._2 > x._2) r else x
  })._1
}

val predLabelMC = test.map(p => {
  (predictMulticlass(p._1.toInt, p._2).toDouble, p._1.toDouble)
})

val metrics = new MulticlassMetrics(predLabelMC)

metrics.labels.foreach { l =>
  println(s"Precision($l, ${ classes(l.toInt) }) = " + metrics.precision(l))
}

metrics.labels.foreach { l =>
  println(s"Recall($l, ${classes(l.toInt)}) = " + metrics.recall(l))
}

metrics.labels.foreach { l =>
  println(s"FPR($l, ${classes(l.toInt)}) = " + metrics.falsePositiveRate(l))
}

metrics.labels.foreach { l =>
  println(s"F1-Score($l, ${classes(l.toInt)}) = " + metrics.fMeasure(l))
}

// Weighted stats
println(s"Weighted precision: ${metrics.weightedPrecision}")
println(s"Weighted recall: ${metrics.weightedRecall}")
println(s"Weighted F1 score: ${metrics.weightedFMeasure}")
println(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}")


val eval = docs.zip(corpus).randomSplit(Array(0.01, 0.99))(0).take(5)

eval.foreach(e => {
  val (d, c) = (e._1, e._2.toIndexedSeq)
  println(d)
  println(c.length)
  println(c.mkString(","))

  val tf = hashingTF.transform(c)  //println(c.map(x => hashingTF.indexOf(x)))
  val tfidf = idf.transform(tf)

  println(model.predict(tfidf))
  println()
})