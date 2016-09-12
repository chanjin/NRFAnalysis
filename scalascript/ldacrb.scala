/**
  * Created by chanjinpark on 2016. 9. 8..
  */

import org.apache.spark.mllib.linalg.{SparseVector, Vector}
import org.apache.spark.rdd.RDD

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

val (vocab, matrix) = {
  val vocab = corpus.flatMap(x => x).distinct.collect.zipWithIndex.toMap
  val matrix: RDD[(Long, Vector)] = corpus.zipWithIndex.map {
    case (tokens, docid) => {
      //val counts = new scala.collection.mutable.HashMap[Int, Double]()
      val vec = tokens.foldLeft(Map[Int, Double]())((res, t) => {
        val vocabid = vocab(t)
        res + (vocabid -> (res.getOrElse(vocabid, 0.0) + 1.0))
      })
      //val (indices, values) = vec.keys
      (docid, new SparseVector(vocab.size, vec.keys.toArray, vec.values.toArray))
    }
  }
  (vocab, matrix)
}

import org.apache.spark.mllib.clustering.DistributedLDAModel
import org.apache.spark.mllib.clustering.LDA

val numTopics = 20
val numTerms = 10
val lda = new LDA().setK(numTopics).setMaxIterations(100)
val model = lda.run(matrix)
val distmodel = model.asInstanceOf[DistributedLDAModel]

val docsl = docs.collect
val topics = distmodel.describeTopics(maxTermsPerTopic = numTerms)
