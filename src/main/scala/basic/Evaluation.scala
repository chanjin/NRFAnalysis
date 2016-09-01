package basic

import org.apache.spark.mllib.evaluation._
import org.apache.spark.rdd.RDD

/**
  * Created by chanjinpark on 2016. 7. 8..
  */
trait Evaluation {
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

  def classificationValue(v: Double, p: Double, threshold: Double) = {
    var (tp, tn, fp, fn) = (0, 0, 0, 0)
    if ( v == 1.0 ) {
      if ( p > threshold) tp = 1
      else fn = 1
    }
    else if (v == 0.0) {
      if ( p > threshold ) fp = 1
      else tn = 1
    }
    else {
      assert(false, "no label")
    }
    (tp, tn, fp, fn, 1)
  }

  def precisionNFalsePositive(labelAndPreds: RDD[(Double, Double)], threshold: Double): (Int, Int, Int, Int, Int)= {
    val pf = labelAndPreds.map(d => classificationValue(d._1, d._2, threshold)).
    reduce((a,b) => (a._1 + b._1, a._2 + b._2, a._3 + b._3, a._4 + b._4, a._5 + b._5))
    pf
  }
}

trait SparkEvaluation {
  def evalBinaryClassification(predictionAndLabels: RDD[(Double, Double)]) = {
    val metrics = new BinaryClassificationMetrics(predictionAndLabels)
    val precision = metrics.precisionByThreshold
    precision.foreach { case (t, p) =>
      println(s"Threshold: $t, Precision: $p")
    }

    val recall = metrics.recallByThreshold
    recall.foreach { case (t, r) =>
      println(s"Threshold: $t, Recall: $r")
    }

    val PRC = metrics.pr // Precision-Recall Curve

    val f1Score = metrics.fMeasureByThreshold    // F-measure
    f1Score.foreach { case (t, f) =>
      println(s"Threshold: $t, F-score: $f, Beta = 1")
    }

    val beta = 0.5
    val fScore = metrics.fMeasureByThreshold(beta)
    f1Score.foreach { case (t, f) =>
      println(s"Threshold: $t, F-score: $f, Beta = 0.5")
    }

    // AUPRC
    val auPRC = metrics.areaUnderPR
    println("Area under precision-recall curve = " + auPRC)

    val thresholds = precision.map(_._1)   // Compute thresholds used in ROC and PR curves

    val roc = metrics.roc     // ROC Curve
    val auROC = metrics.areaUnderROC     // AUROC
    println("Area under ROC = " + auROC)
    (precision, recall, auPRC, auROC)
  }


  def evalMulticlassClassificaiton(predictionAndLabels: RDD[(Double, Double)]) = {
    // Instantiate metrics object
    val metrics = new MulticlassMetrics(predictionAndLabels)

    // Confusion matrix
    println("Confusion matrix:")
    println(metrics.confusionMatrix)

    // Precision by label
    val labels = metrics.labels
    labels.foreach { l =>
      println(s"Precision($l) = " + metrics.precision(l))
    }

    // Recall by label
    labels.foreach { l =>
      println(s"Recall($l) = " + metrics.recall(l))
    }

    // False positive rate by label
    labels.foreach { l =>
      println(s"FPR($l) = " + metrics.falsePositiveRate(l))
    }

    // F-measure by label
    labels.foreach { l =>
      println(s"F1-Score($l) = " + metrics.fMeasure(l))
    }

    // Weighted stats
    println(s"Weighted precision: ${metrics.weightedPrecision}")
    println(s"Weighted recall: ${metrics.weightedRecall}")
    println(s"Weighted F1 score: ${metrics.weightedFMeasure}")
    println(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}")
  }

  def evalMultilabelClassification(scoreAndLabels: RDD[(Array[Double], Array[Double])]) = {
    // Instantiate metrics object
    val metrics = new MultilabelMetrics(scoreAndLabels)

    // Summary stats
    println(s"Recall = ${metrics.recall}")
    println(s"Precision = ${metrics.precision}")
    println(s"F1 measure = ${metrics.f1Measure}")
    println(s"Accuracy = ${metrics.accuracy}")

    // Individual label stats
    metrics.labels.foreach(label =>
      println(s"Class $label precision = ${metrics.precision(label)}"))
    metrics.labels.foreach(label => println(s"Class $label recall = ${metrics.recall(label)}"))
    metrics.labels.foreach(label => println(s"Class $label F1-score = ${metrics.f1Measure(label)}"))

    // Micro stats
    println(s"Micro recall = ${metrics.microRecall}")
    println(s"Micro precision = ${metrics.microPrecision}")
    println(s"Micro F1 measure = ${metrics.microF1Measure}")

    // Hamming loss
    println(s"Hamming loss = ${metrics.hammingLoss}")

    // Subset accuracy
    println(s"Subset accuracy = ${metrics.subsetAccuracy}")
  }

  def evalRanking(relevantDocuments: RDD[(Array[Double], Array[Double])]) = {
    
  }
}
