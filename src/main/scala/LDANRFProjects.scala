
import basic._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg._
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.Map
import scala.collection.JavaConversions._
/**
  * Created by chanjinpark on 2016. 6. 17..
  */



object LDANRFProjects extends PreProcessing {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf(true).setMaster("local").setAppName("NSFLDA")
    val sc = new SparkContext(conf)
    Logger.getLogger("org").setLevel(Level.ERROR)

    val (docs, vocab, matrix) = getInputData(sc)
    val meta = getMetaData(sc).collect().toMap

    import org.apache.spark.mllib.clustering.LDA
    // Set LDA parameters
    val numTopics = 20
    docs.cache()
    matrix.cache()

    val lda = new LDA().setK(numTopics).setMaxIterations(200)
    val ldaModel = lda.run(matrix)

    val id2vocab = vocab.map(_.swap)

    LDAVizHTML.generatePages(ldaModel, id2vocab, docs.collect(), meta)

    val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 10)

    topicIndices.take(numTopics).foreach { case (terms, termWeights) =>
      println("TOPIC:")
      terms.zip(termWeights).foreach {
        case (term, weight) =>
          println(s"${id2vocab(term)}\t$weight")
      }
      println()
    }

    println(vocab.size)
    println(vocab.keys.mkString(", "))
  }
}

