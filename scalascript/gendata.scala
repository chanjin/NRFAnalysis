import java.io.{BufferedWriter, File, FileWriter}

import basic.MetaData
import basic.NRFData._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by chanjinpark on 2016. 9. 8..
  */

val workspace ="/Users/chanjinpark/data/NRFdata/"
val metafile = Array("NRF2013Meta.csv", "NRF2014Meta.csv", "NRF2015Meta.csv").map(x => workspace + x)
val contdir = Array("content2013/", "content2014/", "content2015/").map(workspace + _)

case class MetaData(title: String, mainArea: Array[String], nationArea: Array[String], sixTArea: Array[String]) {
  override def toString: String = title + ":::" + mainArea.mkString(",") + ":::" +
    nationArea.mkString(",") +  ":::" +  sixTArea.mkString(",")
}

def getMetadata(s: String) = {
  val attr = s.split(":::")
  new MetaData(attr(0), attr(1).split(","), attr(2).split(","), attr(3).split(","))
}

def getCorpus(sc: SparkContext, dir: Array[String], meta: Map[String, MetaData]) = {
  val inputdata = getInputData(sc, dir)
  val docs = inputdata.keys//.map(s => s.substring(0, s.lastIndexOf(".txt")))
  val corpus = inputdata.map{
      case (id, strs) => {
        (strs ++ preprocess(meta(id).title).split(" ")).filter(s => !stopWords.contains(s) && s.length != 0)
      }
    }
  (docs, corpus)
}

val (docs, corpus, meta) = {
  val meta = getMetaData(sc, metafile).collect().toMap
  val (docs, corpus) = getCorpus(sc, contdir, meta)

  docs.saveAsTextFile("data/docs")
  corpus.map(_.mkString(",")).saveAsTextFile("data/corpus")

  val file = new java.io.File("data/meta.txt")
  val bw = new BufferedWriter(new FileWriter(file))
  meta.foreach(m => bw.write(m._1 + "-" + m._2.toString +"\n"))
  bw.close()
  (docs, corpus, meta)
}

