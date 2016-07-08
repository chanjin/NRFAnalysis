package Regression

import basic._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.{SparseVector, Vector}
import org.apache.spark.mllib.regression.LabeledPoint
/**
  * Created by chanjinpark on 2016. 7. 7..
  */
object ICTConvRegression extends basic.PreProcessing with basic.Evaluation {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf(true).setMaster("local").setAppName("NSFLDA")
    val sc = new SparkContext(conf)
    Logger.getLogger("org").setLevel(Level.ERROR)

    val (docs, vocab, matrix) = getInputData(sc)

    def isICTConv(s: String) = {
      if (s.equals("ICT·융합연구")) 1.0 else 0.0
    }


    val labels: RDD[(Long, Double)] = docs.zipWithIndex().join(getMetaData(sc).
      map(d => (d._1, isICTConv(d._2._2(0))))).values
    val features: RDD[(Long, Vector)] = matrix.mapValues(v => v.toDense)

    val logsLabelFeatures = labels.join(features).values

    val scaler = new StandardScaler(withMean = true, withStd = true).fit(logsLabelFeatures.map(x => x._2))

    val parsedData = logsLabelFeatures.map { d =>
      LabeledPoint(d._1.toDouble, scaler.transform(d._2))
    }


    val split = parsedData.randomSplit(Array(0.6, 0.3))
    val (training, test) = (split(0), split(1))

    import org.apache.spark.mllib.regression.LinearRegressionWithSGD

    val numIterations = 100
    val stepSize = 0.01
    val model = new LinearRegressionWithSGD().setIntercept(true).run(training)

    // Evaluate model on training examples and compute training error
    println("ICT count - " + test.filter(x => x.label == 1.0).count)

    val valuesAndPreds = test.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }

    val mse = MSE(valuesAndPreds)
    println("training Mean Squared Error = " + mse)

    val loglossErr = logLoss(valuesAndPreds)
    println("Log Loss Error = " + loglossErr)
    // Save and load model

    val mean = valuesAndPreds.values.mean()
    println(mean)

    val pf = precisionNFalsePositive(valuesAndPreds, mean)
    println(f"Precision - ${pf._1}%1.2f, False Positive - ${pf._2}%1.2f, Count - ${pf._3}")

    println(model)
  }
}
