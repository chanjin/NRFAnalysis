package classification

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by chanjinpark on 2016. 8. 31..
  */
object ClassifcationMain extends basic.PreProcessing {
  val workspace ="/Users/chanjinpark/data/NRFdata/"
  val metafile = Array("NRF2013Meta.csv", "NRF2014Meta.csv", "NRF2015Meta.csv").map(x => workspace + x)
  val contdir = Array("content2013/", "content2014/", "content2015/").map(workspace + _)

  def main(args: Array[String]) : Unit = {
    val conf = new SparkConf(true).setMaster("local").setAppName("NSFLDA")
    val sc = new SparkContext(conf)
    Logger.getLogger("org").setLevel(Level.ERROR)


    val meta = getMetaData(sc, metafile).collect().toMap
    val (docs, corpus) = getCorpus(sc, contdir, meta)

    //val dt = ICTDecisionTree(docs, corpus, meta)
    //val dt = GradientBoosting(docs, corpus, meta)
    //val dt = NaiveBayesNRF(docs, corpus, meta)
    //
    // val dt = Regression(docs, corpus, meta)

    val dt = NaiveBayesNRF(docs, corpus, meta)
    dt.run
  }
}
