package interpretation

import basic.NRFData
import clustering.NRFLDAMain._
import org.apache.log4j.{Level, Logger}

import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by chanjinpark on 2016. 9. 7..
  */
class LDATopicCRB {

}

object LDATopicCRB extends App {
  val dir = "/Users/chanjinpark/GitHub/NRFAnalysis/"

  // 각 주제 별 구성 CRB 분류체계는 어떻게 되는 지를 보여줌
  val conf = new SparkConf(true).setMaster("local").setAppName("NSFLDA")
  val sc = new SparkContext(conf)
  Logger.getLogger("org").setLevel(Level.ERROR)

  val (docs, corpus, meta) = NRFData.load(sc)
  val (vocab, matrix) = getMatrix(corpus)
  val id2vocab = vocab.map(_.swap)

  import org.apache.spark.mllib.clustering.DistributedLDAModel

  // 각 Topic을 출력
  // 한 토픽에 대해서
  import org.apache.spark.mllib.clustering.LDA
  // Set LDA parameters
  val numTopics = 20
  val numTerms = 10

  val lda = new LDA().setK(numTopics).setMaxIterations(100)
  val model = lda.run(matrix)
  val distmodel = model.asInstanceOf[DistributedLDAModel]

  val docsl = docs.collect
  val topics = distmodel.describeTopics(maxTermsPerTopic = numTerms)
  // topics(i)._1.zip(topics(i)._2) 각 토픽의 Term-Weight 리스트

  // 주제 별 탑 문서 50개
  val topDocsPerTopic = distmodel.topDocumentsPerTopic(50)
  (0 until numTopics).foreach(k => {
    val terms = topics(k)._1.zip(topics(k)._2).map { case (t, w) =>
        s"${id2vocab(t)}\n" + f"$w%1.3f"
    }
    val doclist = topDocsPerTopic(k)._1.zip(topDocsPerTopic(k)._2).filter(dw => dw._2 > 0.1)
    val crbarea = crbArea(doclist.map(_._1))
  })

  val docs2topics = distmodel.topTopicsPerDocument(20).collect()

  def crbArea(doclist: Array[Long]) = {
    def getFirstOrNone(arr: Array[String]) = if ( arr.length == 0 ) "NONE" else arr(0)
    def makestr(m: Map[String, Int]) = m.toList.sortBy(-_._2).map(kv => kv._1 + " - " + kv._2).mkString(",\t")

    val summary = doclist.map(did => {
      val a = meta(docsl(did.toInt))
      if ( a.mainArea.length < 2) println( did.toInt + " -- " + a.mainArea.mkString(","))
      (a.mainArea(0), a.mainArea(1))
    }).foldLeft((Map[String, Int](), Map[String, Int]()))((r, as) => {
      (r._1 + (as._1 -> (r._1.getOrElse(as._1, 0) + 1)), r._2 + (as._2 -> (r._2.getOrElse(as._2, 0) + 1) ))
    })
    makestr(summary._2)
  }

}


