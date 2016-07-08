package basic

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
