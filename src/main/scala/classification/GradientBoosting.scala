package classification

import basic.{MetaData, NRFData}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.annotation.Since
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree._
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.configuration.QuantileStrategy.{apply => _, _}
import org.apache.spark.mllib.tree.configuration._
import org.apache.spark.mllib.tree.impurity.{Gini, Impurity}
import org.apache.spark.mllib.tree.loss.LogLoss
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.beans.BeanProperty

/**
  * Created by chanjinpark on 2016. 7. 9..
  */

class GradientBoosting(docs: RDD[String], corpus: RDD[Array[String]], metadata: Map[String, MetaData])
  extends Serializable with basic.TFIDF with basic.Evaluation {

  def isICTConv(s: String) = if (s.equals("ICT·융합연구")) 1.0 else 0.0

  def run = {
    val (tfidf, hashtf) = getMatrix(corpus)

    def isICTConv(s: String) = if (s.equals("ICT·융합연구")) 1.0 else 0.0
    val parsedData = docs.zip(tfidf).map(d => {
      LabeledPoint(isICTConv(metadata(d._1).mainArea(0)), d._2.toDense)
    })
    val split = parsedData.randomSplit(Array(0.8, 0.2))
    val (training, test) = (split(0), split(1))

    // Train a GradientBoostedTrees model.
    // The defaultParams for Classification use LogLoss by default.
    val boostingStrategy = BoostingStrategy.defaultParams("Classification")
    boostingStrategy.setNumIterations(10) // Note: Use more iterations in practice.
    //boostingStrategy.setTreeStrategy(new Strategy())

    val model = GradientBoostedTrees.train(training, boostingStrategy)

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
    println(f"융합과제 맞춘 것은 ${tp}개, 비융합과제를 융합과제로 예측한 것은 ${fp}개")
    println(f"비융합과제 맞춘 것은 ${tn}개, 융합과제를 비융합과제로 예측한 것은 ${fn}개")
    println(f"Precision = ${tp.toDouble/(tp + fp)}%1.2f, Recall = ${tp.toDouble/(tp + fn)}%1.2f")
    println(f"Accuracy = ${(tp + tn).toDouble/(tp + tn + fp + fn)}")
  }


  def runMulticlass(sc: SparkContext, getLabel: MetaData => Int, classes: Map[Int, String]) = {

    // Gradient Boosting in spark does not support multiclass yet.
    import java.io._
    val dir = "data/gradientboost/"

    val (tfidf, hashtf) = getMatrix(corpus)
    val parsedData = docs.zip(tfidf).map(d => {
      LabeledPoint(getLabel(metadata(d._1)), d._2.toDense)
    }).randomSplit(Array(0.8, 0.2))

    val (training, test) = (parsedData(0), parsedData(1))

    // Train a GradientBoostedTrees model.
    // The defaultParams for Classification use LogLoss by default.

    def getStrategy: BoostingStrategy = {
      val strategy = new Strategy(Algo.Classification, Gini, maxDepth = 3, numClasses = 2)
      BoostingStrategy(strategy, LogLoss, numIterations = 20, learningRate = 0.2)
    }

    val boostingStrategy = getStrategy
    val model = GradientBoostedTrees.train(training, boostingStrategy)

    val modelfile = dir + "model"
    if (new File(modelfile).exists()) println(s"$modelfile exists, so skip file generation ")
    else
      model.save(sc, modelfile)

    val predictionAndLabels = test.map(p => (model.predict(p.features), p.label))

    val predlabelfile = dir + "predlabel"
    if (new File(predlabelfile).exists()) println(s"$predlabelfile exists, so skip file generation ")
    else
      predictionAndLabels.saveAsTextFile(predlabelfile)

    // 결과 저장 (docid, pred, real)

    val resultfile = dir + "testresult"
    if (new File(resultfile).exists()) println(s"$resultfile exists, so skip file generation ")
    else
      predictionAndLabels.map(x => x._1 + ", " + x._2).saveAsTextFile(resultfile)

    val metrics = new MulticlassMetrics(predictionAndLabels)
    println("Confusion matrix:")
    println(metrics.confusionMatrix)

    val f = new BufferedWriter(new FileWriter(new File(dir + "metrics.txt")))
    f.write("Confusion Matrix\n")
    val m = metrics.confusionMatrix
    val cmstr = (0 until m.numRows).map(i =>
      (0 until m.numCols).map(j => m(i, j)).mkString(",")
    ).mkString("\n")

    f.write(cmstr)

    // Precision by label
    f.write("\nPrecision by Label\n")
    val labels = metrics.labels
    labels.foreach { l =>
      println(s"Precision($l, ${ classes(l.toInt) }) = " + metrics.precision(l))
      f.write(s"$l, ${classes(l.toInt)}, ${metrics.precision(l)}\n")
    }

    // Recall by label
    f.write("\nRecall by Label\n")
    labels.foreach { l =>
      println(s"Recall($l, ${ classes(l.toInt) }) = " + metrics.recall(l))
      f.write(s"$l, ${classes(l.toInt)}, ${metrics.recall(l)}\n")
    }

    // False positive rate by label
    f.write("\nFPR by Label\n")
    labels.foreach { l =>
      println(s"FPR($l, ${ classes(l.toInt) }) = " + metrics.falsePositiveRate(l))
      f.write(s"$l, ${classes(l.toInt)}, ${metrics.falsePositiveRate(l)}\n")
    }

    // F-measure by label
    f.write("\nF1 Score by Label\n")
    labels.foreach { l =>
      println(s"F1-Score($l, ${ classes(l.toInt) }) = " + metrics.fMeasure(l))
      f.write(s"$l, ${classes(l.toInt)}, ${metrics.fMeasure(l)}\n")
    }

    f.write(s"\nWeighted precision: ${metrics.weightedPrecision}\n")
    f.write(s"\nWeighted recall: ${metrics.weightedRecall}\n")
    f.write(s"\nWeighted F1 score: ${metrics.weightedFMeasure}\n")
    f.write(s"\nWeighted false positive rate: ${metrics.weightedFalsePositiveRate}\n")

    // Weighted stats
    println(s"Weighted precision: ${metrics.weightedPrecision}")
    println(s"Weighted recall: ${metrics.weightedRecall}")
    println(s"Weighted F1 score: ${metrics.weightedFMeasure}")
    println(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}")

    f.close()
  }


}


object GradientBoosting extends basic.PreProcessing with basic.Evaluation {

  def apply(docs: RDD[String], corpus: RDD[Array[String]], meta: Map[String, MetaData]) =
    new GradientBoosting(docs, corpus, meta)


  def main(args: Array[String]): Unit = {
    val conf = new SparkConf(true).setMaster("local").setAppName("NSFLDA")
    val sc = new SparkContext(conf)
    Logger.getLogger("org").setLevel(Level.ERROR)

    val (docs, corpus, meta) = NRFData.load(sc)
    val gbt = GradientBoosting(docs, corpus, meta)


    val crb: Map[String, Int] = meta.map(_._2.mainArea(1)).toList.distinct.sortWith(_.compare(_) < 0).zipWithIndex.toMap
    val crpcls = meta.groupBy(_._2.mainArea(1)).map(x => (x._1, x._2.size))
    val i2crb = crb.map(_.swap)
    (0 until crb.size).foreach( x => println(x + ": " + i2crb(x) + " - " + crpcls(i2crb(x))))

    def getLabel(m: MetaData) = crb(m.mainArea(1)) // CRB 분류
    gbt.runMulticlass(sc, getLabel, crb.map(_.swap))

  }


}
