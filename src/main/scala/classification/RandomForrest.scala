package classification

import basic.MetaData
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.{SparseVector, Vector, Vectors}

/**
  * Created by chanjinpark on 2016. 7. 9..
  */
class RandomForreste(docs: RDD[String], corpus: RDD[Array[String]], metadata: Map[String, MetaData])
  extends Serializable with  basic.TFIDF with basic.Evaluation {

  def run() = {

    val (tfidf, hashtf) = getMatrix(corpus)

    def isICTConv(s: String) = if (s.equals("ICT·융합연구")) 1.0 else 0.0
    val parsedData = docs.zip(tfidf).map(d => {
      LabeledPoint(isICTConv(metadata(d._1).mainArea(0)), d._2.toDense)
    })
    val split = parsedData.randomSplit(Array(0.8, 0.2))
    val (training, test) = (split(0), split(1))

    // Train a RandomForest model.
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 3 // Use more in practice.
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "gini"
    val maxDepth = 4
    val maxBins = 32


    val model = RandomForest.trainClassifier(training, numClasses, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

    val valuesAndPreds = test.map(p => (p.label, model.predict(p.features)))

    // Save and load model
    //println("Learned classification forest model:\n" + model.toDebugString)
    //model.save(sc, "target/tmp/myRandomForestClassificationModel")
    //val sameModel = RandomForestModel.load(sc, "target/tmp/myRandomForestClassificationModel")

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
    printMetrics(tp, tn, fp, fn, count)


    def printMetrics(tp: Int, tn: Int, fp: Int, fn: Int, count: Int) = {
      println(f"융합과제 맞춘 것은 ${tp}개, 비융합과제를 융합과제로 예측한 것은 ${fp}개")
      println(f"비융합과제 맞춘 것은 ${tn}개, 융합과제를 비융합과제로 예측한 것은 ${fn}개")
      println(f"Precision = ${tp.toDouble/(tp + fp)}%1.2f, Recall = ${tp.toDouble/(tp + fn)}%1.2f")
      println(f"Accuracy = ${(tp + tn).toDouble/(tp + tn + fp + fn)}")
    }

    println("Evaluation")
  }
}

object RandomForrest extends basic.PreProcessing with basic.Evaluation with basic.TFIDF {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf(true).setMaster("local").setAppName("NSFLDA")
    val sc = new SparkContext(conf)
    Logger.getLogger("org").setLevel(Level.ERROR)


    /*
    val evalresult = eval.map {
      case (metadata, vec) => {
        val point = LabeledPoint(isICTConv(metadata._2._2(0)), vec.toDense)
        val value = point.label
        val pred = model.predict(point.features)
        val (tp, tn, fp, fn, _) = classificationValue(value, pred, 0.8)
        (fp, fn, tp, tn, metadata)
      }
    }

    val metric = evalresult.map(x => (x._1, x._2, x._3, x._4)).reduce((a,b) => (a._1 + b._1, a._2 + b._2, a._3 + b._3, a._4 + b._4))
    val sum = metric._1 + metric._2 + metric._3 + metric._4
    printMetrics(metric._1, metric._2, metric._3, metric._4, sum)
    println(evalresult.count + " = " + sum)

    println("비융합과제인데 융합과제라고 한것의 내용 확인")
    println("False Positives - ")
    val falsepositives = evalresult.filter(_._1 == 1).map(_._5)
    falsepositives.take(5).map(md => Array(md._1, md._2._1, md._2._2.mkString(":")).mkString(", ")).foreach(println)

    println("융합과제인데 융합과제가 아니라고 한 것의 내용 확인")
    println("False Negatives")
    val falsenegatives = evalresult.filter(_._2 == 1).map(_._5)
    falsenegatives.take(5).map(md => Array(md._1, md._2._1, md._2._2.mkString(":")).mkString(", ")).foreach(println)

    println("융합과제인데 융합과제라고 잘 맞춘것의 내용 확인")
    println("True Positives")
    val truepositives = evalresult.filter(_._3 == 1).map(_._5)
    truepositives.take(5).map(md => Array(md._1, md._2._1, md._2._2.mkString(":")).mkString(", ")).foreach(println)

    println("비융합과제인데 비융합과제라고 잘 맞춘것의 내용 확인")
    println("True Negatives")
    val truenegatives = evalresult.filter(_._4 == 1).map(_._5)
    truenegatives.take(5).map(md => Array(md._1, md._2._1, md._2._2.mkString(":")).mkString(", ")).foreach(println)
*/

  }

}
