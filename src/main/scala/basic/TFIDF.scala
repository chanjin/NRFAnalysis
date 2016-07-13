package basic

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.rdd._
import org.apache.spark.mllib.linalg.{SparseVector, Vector, Vectors}

/**
  * Created by chanjinpark on 2016. 7. 11..
  */
trait TFIDF {

  def getMatrix(corpus:  RDD[Array[String]]) : (RDD[Vector], HashingTF) = {
    val hashingTF = new HashingTF()
    val tf: RDD[Vector] = hashingTF.transform(corpus.map(c => c.toIndexedSeq))

    import org.apache.spark.mllib.feature.IDF
    tf.cache()
    val idf = new IDF().fit(tf)
    (idf.transform(tf), hashingTF)
  }
}


object TFIDFTest extends TFIDF with PreProcessing {

  def main(args: Array[String]) : Unit = {
    val conf = new SparkConf(true).setMaster("local").setAppName("NSFLDA")
    val sc = new SparkContext(conf)
    Logger.getLogger("org").setLevel(Level.ERROR)

    val corpus = getCorpus(sc)

    val (tfidf, hashtf) = getMatrix(corpus)

    //tfidf.saveAsTextFile("data/corpus")
    println(tfidf.count)
    println(corpus.count)

  }
}
