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
  def getMatrix(corpus:  RDD[Array[String]]) = {
    val hashingTF = new HashingTF()
    val tf: RDD[Vector] = hashingTF.transform(corpus.map(c => c.toIndexedSeq))

    import org.apache.spark.mllib.feature.IDF
    tf.cache()
    val idf = new IDF().fit(tf)
    val tfidf: RDD[Vector] = idf.transform(tf)
    tfidf
  }
}


object TFIDFTest extends TFIDF with PreProcessing {
  def getCorpus(sc: SparkContext) = {
    val inputdata = sc.wholeTextFiles(contdir + "*.txt").
      map(x => {
        val name = x._1.substring(contdir.length + 5)
        val lines = x._2.split("\n")
        var words = Array[String]()
        try {
          words = lines.flatMap(l => vocabWords(preprocess(l)))
        } catch {
          case e: Exception => { println(e); println(name); println(lines)}
        }
        (name, words)
      })

    val stopWordsSet = Set("of", "and", "in", "the", "Mixed", "근간한", "대한", "활용", "관련", "가능", "연구", "개발", "통한",
      "제시", "제공", "이용", "적용", "다양")
    stopWords = sc.broadcast(stopWordsSet).value

    val docs = inputdata.keys.map(s => s.substring(0, s.lastIndexOf(".txt")))
    val metadata = getMetaData(sc).collect.toMap

    val corpus = inputdata.map{
      case (id, strs) => {
        val meta = metadata(id.substring(0, id.lastIndexOf(".")))
        strs.filter(s => !stopWords.contains(s)) ++ Array(meta._1) ++ meta._2 ++ meta._3 ++ meta._4
      }
    }
    corpus
  }

  def main(args: Array[String]) : Unit = {
    val conf = new SparkConf(true).setMaster("local").setAppName("NSFLDA")
    val sc = new SparkContext(conf)
    Logger.getLogger("org").setLevel(Level.ERROR)

    val corpus = getCorpus(sc)
    val tfidf = getMatrix(corpus)

    tfidf.saveAsTextFile("data/corpus")
  }
}

object TFIDFLoader {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf(true).setMaster("local").setAppName("NSFLDA")
    val sc = new SparkContext(conf)
    Logger.getLogger("org").setLevel(Level.ERROR)

    def parse(rdd: RDD[String]): RDD[(Long, Vector)] = {
      val pattern: scala.util.matching.Regex = "\\(([0-9]+),(.*)\\)".r
      rdd .map{
        case pattern(k, v) => (k.toLong, Vectors.parse(v))
      }
    }

    val tfidf = parse(sc.textFile("data/corpus/part-*"))
  }
}