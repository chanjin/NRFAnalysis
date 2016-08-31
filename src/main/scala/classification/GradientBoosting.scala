package classification

import basic.MetaData
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree._
import org.apache.spark.mllib.tree.configuration._
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

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

}


object GradientBoosting extends basic.PreProcessing with basic.Evaluation {
  def apply(docs: RDD[String], corpus: RDD[Array[String]], meta: Map[String, MetaData]) =
    new GradientBoosting(docs, corpus, meta)


  def main(args: Array[String]): Unit = {

  /*
    val (docs, vocab, matrix) = getInputData(sc)

    val metadata:  RDD[(String, (String, Array[String], Array[String], Array[String]))] = getMetaData(sc)
    val docids = docs.zipWithIndex()

    val dataJoined = metadata.zipWithIndex().map(_.swap).join(matrix).values
    val split0 = dataJoined.randomSplit(Array(0.9, 0.1))
    val (data, eval) = (split0(0), split0(1))

    val parsedData = data.map(d => LabeledPoint(isICTConv(d._1._2._2(0)), d._2.toDense))
    val split = parsedData.randomSplit(Array(0.9, 0.1))
    val (training, test) = (split(0), split(1))
  */


  }


}
