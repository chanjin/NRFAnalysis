package basic

import java.io.{BufferedWriter, File, FileWriter}

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by chanjinpark on 2016. 9. 8..
  */
object NRFDataSample extends PreProcessing {

  val workspace ="/Users/chanjinpark/data/NRFdata/"
  val metafile = Array("NRF2015Meta.csv").map(x => workspace + x)
  val contdir = Array("content2015/").map(workspace + _)

  def save(sc: SparkContext) = {
    val meta = getMetaData(sc, metafile).collect().toMap
    val (docs, corpus) = getCorpus(sc, contdir, meta)

    docs.saveAsTextFile("datasample/docs")
    corpus.map(_.mkString(",")).saveAsTextFile("datasample/corpus")

    val file = new java.io.File("datasample/meta.txt")
    val bw = new BufferedWriter(new FileWriter(file))
    meta.foreach(m => bw.write(m._1 + "-" + m._2.toString +"\n"))
    bw.close()
    (docs, corpus, meta)
  }

  def load(sc: SparkContext) = {
    if ( ! new File("datasample/docs").exists()) {
      save(sc)
    }
    else {
      val dir = "/Users/chanjinpark/GitHub/NRFAnalysis/"
      val meta = {
        scala.io.Source.fromFile(dir + "datasample/meta.txt").getLines().map(l => {
          val id = l.substring(0, l.indexOf("-"))
          val meta = MetaData(l.substring(l.indexOf("-") + 1))
          (id, meta)
        }).toMap
      }

      val docs = sc.textFile(dir + "datasample/docs")
      val corpus = sc.textFile(dir + "datasample/corpus").map(_.split(","))
      (docs, corpus, meta)
    }
  }

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf(true).setMaster("local").setAppName("NSFLDA")
    val sc = new SparkContext(conf)
    Logger.getLogger("org").setLevel(Level.ERROR)

    save(sc)
  }
}
