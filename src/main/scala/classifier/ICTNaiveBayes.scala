package classifier

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.{SparseVector, Vector, Vectors}

/**
  * Created by chanjinpark on 2016. 6. 21..
  */
object ICTNaiveBayes extends basic.PreProcessing with basic.Evaluation with basic.TFIDF {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf(true).setMaster("local").setAppName("NSFLDA")
    val sc = new SparkContext(conf)
    Logger.getLogger("org").setLevel(Level.ERROR)

    val corpus = getCorpus(sc)
    val (tfidf, hashtf) = getMatrix(corpus)


    def isICTConv(s: String) = if (s.equals("ICT·융합연구")) 1.0 else 0.0

    val metadata:  RDD[(String, (String, Array[String], Array[String], Array[String]))] = getMetaData(sc)

    val data = metadata.zip(tfidf)
    //val split0 = dataJoined.randomSplit(Array(0.99, 0.1))
    //val (data, eval) = (split0(0), split0(1))

    val parsedData = data.map(d => LabeledPoint(isICTConv(d._1._2._2(0)), d._2.toDense))
    val split = parsedData.randomSplit(Array(0.8, 0.2))
    val (training, test) = (split(0), split(1))

    val model = NaiveBayes.train(training, lambda = 1.0, modelType = "multinomial")
    val valuesAndPreds = test.map(p => (p.label, model.predict(p.features)))

    //TODO: Bernoulli naive bayse. modelType = "bernoulli".
    //For this, data should be binary, or 0 or 1, rather than frequency of words

    // Save and load model
    //model.save(sc, "target/tmp/myNaiveBayesModel")
    //val sameModel = NaiveBayesModel.load(sc, "target/tmp/myNaiveBayesModel")

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

/*
    val evalresult = eval.map {
      case (metadata, vec) => {
        val point = LabeledPoint(isICTConv(metadata._2._2(0)), vec.toDense)
        val value = point.label
        val pred = model.predict(point.features)
        val (tp, tn, fp, fn, _) = classificationValue(value, pred, 0.8)
        (fp, fn, metadata)
      }
    }

    println("비융합과제인데 융합과제라고 한것의 내용 확인")
    println("False Positives")
    val falsepositives = evalresult.filter(_._1 == 1).map(_._3)
    falsepositives.take(5).map(md => Array(md._1, md._2._1, md._2._2.mkString(":")).mkString(", ")).foreach(println)

    println("융합과제인데 융합과제가 아니라고 한 것의 내용 확인")
    println("False Negatives")
    val falsenegatives = evalresult.filter(_._2 == 1).map(_._3)
    falsepositives.take(5).map(md => Array(md._1, md._2._1, md._2._2.mkString(":")).mkString(", ")).foreach(println)
    */
  }


}
