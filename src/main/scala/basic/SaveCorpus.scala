package basic

import java.io.{BufferedWriter, File, FileWriter}

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by chanjinpark on 2016. 9. 16..
  */
object SaveCorpus extends App {
  val conf = new SparkConf(true).setMaster("local").setAppName("NSFLDA")
  val sc = new SparkContext(conf)
  Logger.getLogger("org").setLevel(Level.ERROR)

  import org.apache.spark.sql.SparkSession

  val spark = SparkSession
    .builder()
    .appName("Spark SQL Example")
    .config("spark.some.config.option", "some-value")
    .getOrCreate()


  val sqlContext = new org.apache.spark.sql.SQLContext(sc)

  val dir = "/Users/chanjinpark/GitHub/NRFAnalysis/"

  val docs = sc.textFile(dir + "data/docs")
  var f = new BufferedWriter(new FileWriter(new File(dir + "data/temp/docs.txt")))
  f.write(docs.collect.mkString(","))
  f.close()

  val corpus = sc.textFile(dir + "data/corpus").map(_.split(","))
  f = new BufferedWriter(new FileWriter(new File(dir + "data/temp/corpus.csv")))
  corpus.map(strs => strs.mkString(","))
  f.close()

  val vocab = corpus.flatMap(x => x).collect().distinct
  f = new BufferedWriter(new FileWriter(new File(dir + "data/temp/vocab.txt")))
  f.write(vocab.mkString(","))
  f.close()
}
