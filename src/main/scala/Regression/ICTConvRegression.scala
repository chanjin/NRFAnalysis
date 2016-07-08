package Regression

import basic._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by chanjinpark on 2016. 7. 7..
  */
object ICTConvRegression extends basic.PreProcessing {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf(true).setMaster("local").setAppName("NSFLDA")
    val sc = new SparkContext(conf)
    Logger.getLogger("org").setLevel(Level.ERROR)

    val (docs, vocab, matrix) = getInputData(sc)

    def isICTConv(s: String) = {
      if (s.equals("ICT·융합연구")) 1.0 else 0.0
    }

    val labels: RDD[(Long, Double)] = docs.zipWithIndex().join(getMetaData(sc).map(d => (d._1, isICTConv(d._2._2(0))))).values
    val features: RDD[(Long, Vector)] = matrix


  }

}
