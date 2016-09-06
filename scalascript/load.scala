
/**
  * Created by chanjinpark on 2016. 9. 6..
  */

val dir = "/Users/chanjinpark/GitHub/NRFAnalysis/"
val docs = sc.textFile(dir + "data/docs")

case class MetaData(title: String, mainArea: Array[String], nationArea: Array[String], sixTArea: Array[String]) {
  override def toString: String = title + ":::" + mainArea.mkString(",") + ":::" +
    nationArea.mkString(",") +  ":::" +  sixTArea.mkString(",")
}

def getMetadata(s: String) = {
  val attr = s.split(":::")
  new MetaData(attr(0), attr(1).split(","), attr(2).split(","), attr(3).split(","))
}

val meta = {
  scala.io.Source.fromFile(dir + "data/meta.txt").getLines().map(l => {
    val id = l.substring(0, l.indexOf("-"))
    val meta = getMetadata(l.substring(l.indexOf("-") + 1))
    (id, meta)
  }).toMap
}

val docs = sc.textFile(dir + "data/docs")
val corpus = sc.textFile(dir + "data/corpus").map(_.split(","))

import org.apache.spark.mllib.feature.{HashingTF, IDF}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD

val hashingTF = new HashingTF() // Hashing을 이용해서 Term Frequency를 구함
val tf: RDD[Vector] = hashingTF.transform(corpus.map(c => c.toIndexedSeq))
tf.cache()

val idf = new IDF().fit(tf)
val (tfidf, hashtf) = (idf.transform(tf), hashingTF)



