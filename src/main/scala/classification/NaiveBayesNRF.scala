package classification

import basic.{MetaData, NRFData}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.{SparseVector, Vector, Vectors}

/**
  * Created by chanjinpark on 2016. 6. 21..
  */

class NaiveBayesNRF(docs: RDD[String], corpus: RDD[Array[String]], metadata: Map[String, MetaData])
  extends Serializable with  basic.TFIDF with basic.Evaluation {



  def run = {
    val (tfidf, hashtf) = getMatrix(corpus)
    def isICTConv(s: String) = if (s.equals("ICT·융합연구")) 1.0 else 0.0
    val parsedData = docs.zip(tfidf).map(d => {
      LabeledPoint(isICTConv(metadata(d._1).mainArea(0)), d._2.toDense)
    }).randomSplit(Array(0.8, 0.2))

    val (training, test) = (parsedData(0), parsedData(1))

    val model = NaiveBayes.train(training, lambda = 1.0, modelType = "multinomial")
    val valuesAndPreds = test.map(p => (p.label, model.predict(p.features)))

    val mse = MSE(valuesAndPreds)
    println("training Mean Squared Error = " + mse)

    val loglossErr = logLoss(valuesAndPreds)
    println("Log Loss Error = " + loglossErr)
    // Save and load model

    val mean = valuesAndPreds.values.mean()
    println(f"Prediction mean - $mean%1.2f")

    val accuracy = 1.0 * valuesAndPreds.filter(x => x._1 == x._2).count() / test.count()
    println(f"Accuracy - ${accuracy}%1.2f")

    val (tp, tn, fp, fn, count) = precisionNFalsePositive(valuesAndPreds, 0.8)
    val cntIctconv = test.filter(_.label == 1.0).count
    println(s"$cntIctconv = $tp + $fn")

    println(f"전체 ${count} 개, 융합과제수는 ${cntIctconv} 개")
    println(f"융합과제 맞춘 것은 ${tp}개, 비융합과제를 융합과제로 예측한 것은 ${fp} 개")
    println(f"Precision = ${tp.toDouble/(tp + fp)}%1.2f, Recall = ${tp.toDouble/(tp + fn)}%1.2f")

    println(f"Accuracy = ${(tp + tn).toDouble/(tp + tn + fp + fn)}")
  }


  def runMulticlass(getLabel: MetaData => Int, classes: Map[Int, String]) = {
    val (tfidf, hashtf) = getMatrix(corpus)
    val parsedData = docs.zip(tfidf).map(d => {
      LabeledPoint(getLabel(metadata(d._1)), d._2.toDense)
    }).randomSplit(Array(0.8, 0.2))

    val (training, test) = (parsedData(0), parsedData(1))

    val model = NaiveBayes.train(training, lambda = 1.0, modelType = "multinomial")
    val predictionAndLabels = test.map(p => (model.predict(p.features), p.label))


    val metrics = new MulticlassMetrics(predictionAndLabels)
    println("Confusion matrix:")
    println(metrics.confusionMatrix)

    // Precision by label
    val labels = metrics.labels
    labels.foreach { l =>
      println(s"Precision($l, ${ classes(l.toInt) }) = " + metrics.precision(l))
    }

    // Recall by label
    labels.foreach { l =>
      println(s"Recall($l, ${ classes(l.toInt) }) = " + metrics.recall(l))
    }

    // False positive rate by label
    labels.foreach { l =>
      println(s"FPR($l, ${ classes(l.toInt) }) = " + metrics.falsePositiveRate(l))
    }

    // F-measure by label
    labels.foreach { l =>
      println(s"F1-Score($l, ${ classes(l.toInt) }) = " + metrics.fMeasure(l))
    }

    // Weighted stats
    println(s"Weighted precision: ${metrics.weightedPrecision}")
    println(s"Weighted recall: ${metrics.weightedRecall}")
    println(s"Weighted F1 score: ${metrics.weightedFMeasure}")
    println(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}")
  }
}

object NaiveBayesNRF  {

  def apply(docs: RDD[String], corpus: RDD[Array[String]], meta: Map[String, MetaData]) =
    new NaiveBayesNRF(docs, corpus, meta)

  val workspace ="/Users/chanjinpark/data/NRFdata/"
  val metafile = Array("NRF2013Meta.csv", "NRF2014Meta.csv", "NRF2015Meta.csv").map(x => workspace + x)
  val contdir = Array("content2013/", "content2014/", "content2015/").map(workspace + _)

  def main(args: Array[String]): Unit = {

    //TODO: 결과를 Save하고 Python으로 그림 그리는 것

    //For this, data should be binary, or 0 or 1, rather than frequency of words

    // Save and load model
    //model.save(sc, "target/tmp/myNaiveBayesModel")
    //val sameModel = NaiveBayesModel.load(sc, "target/tmp/myNaiveBayesModel")
    val conf = new SparkConf(true).setMaster("local").setAppName("NSFLDA")
    val sc = new SparkContext(conf)
    Logger.getLogger("org").setLevel(Level.ERROR)

    val (docs, corpus, meta) = NRFData.load(sc)
    val dt = NaiveBayesNRF(docs, corpus, meta)


    val crb: Map[String, Int] = meta.map(_._2.mainArea(1)).toList.distinct.sortWith(_.compare(_) < 0).zipWithIndex.toMap
    val crpcls = meta.groupBy(_._2.mainArea(1)).map(x => (x._1, x._2.size))
    val i2crb = crb.map(_.swap)
    (0 until crb.size).foreach( x => println(x + ": " + i2crb(x) + " - " + crpcls(i2crb(x))))

    def getLabel(m: MetaData) = crb(m.mainArea(1)) // CRB 분류
    dt.runMulticlass(getLabel, crb.map(_.swap))
  }

}
