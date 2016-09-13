package examples




/**
  * Created by chanjinpark on 2016. 9. 12..
  */
object NBAnaysis extends App {
  import org.apache.log4j.{Level, Logger}
  import org.apache.spark.{SparkConf, SparkContext}


  val conf = new SparkConf(true).setMaster("local").setAppName("NSFLDA")
  val sc = new SparkContext(conf)
  Logger.getLogger("org").setLevel(Level.ERROR)

  import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
  import org.apache.spark.mllib.util.MLUtils

  // Load and parse the data file.
  val dir = "/Users/chanjinpark/GitHub/spark/"
  val data = MLUtils.loadLibSVMFile(sc, dir + "data/mllib/sample_libsvm_data.txt")

  // Split data into training (60%) and test (40%).
  val Array(training, test) = data.randomSplit(Array(0.6, 0.4))

  import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
  val model = NaiveBayes.train(training, lambda = 0.5, modelType = "multinomial")
  //val model = new LogisticRegressionWithLBFGS().setNumClasses(2).setIntercept(true).run(training)
  //model.clearThreshold()

  val predictionAndLabel = test.map(p => (model.predict(p.features), p.label))
  val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()
  predictionAndLabel.foreach(println)

  //model.
  val pl = test.map(p => (model.predictProbabilities(p.features), p.label))
  pl.foreach(println)
}
