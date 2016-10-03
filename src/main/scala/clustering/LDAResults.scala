package clustering

import java.io.{BufferedWriter, File, FileWriter}

import basic.MetaData
import org.apache.spark.mllib.clustering.LDAModel

/**
  * Created by chanjinpark on 2016. 9. 18..
  */
class LDAResults( ldaModel: LDAModel,  vocabArray: Map[Int, String],  docs: Array[String],
                  area: Map[String, MetaData],  numTerms: Int,
                  docpath: String => String, dir: String = "ldaviz/")  {


  import org.apache.spark.mllib.clustering.DistributedLDAModel
  val distmodel = ldaModel.asInstanceOf[DistributedLDAModel]
  val topics = ldaModel.describeTopics(maxTermsPerTopic = numTerms)
  val numTopics = ldaModel.k

  def crbArea(did: Int) = area(docs(did)).mainArea(1)

  def writeTopicInfo(topDocsPerTopic: Array[(Array[Long], Array[Double])]) = {
    var f = new BufferedWriter(new FileWriter(new File(dir + "topics_terms.csv")))
    f.write("topicid, term, weight" + "\n")
    f.write((0 until numTopics).map(k => {
      topics(k)._1.zip(topics(k)._2).map { case (t, w) => k + ", " + vocabArray(t) + ", " + w }.mkString("\n")
    }).mkString("\n"))
    f.close()

    f = new BufferedWriter(new FileWriter(new File(dir + "topics_docs.csv")))
    f.write("topicid, doc, crb, weight" + "\n")
    f.write((0 until numTopics).map(k => {
      val doclist = topDocsPerTopic(k)._1.zip(topDocsPerTopic(k)._2).filter(dw => dw._2 > 0.2)
      doclist.map { case (d, w) => k + ", " + docs(d.toInt) + ", " + crbArea(d.toInt) + ", " + w }.mkString("\n")
    }).mkString("\n"))
    f.close()
  }

  def writeDocInfo(doc2topics: Array[(Long, Array[Int], Array[Double])]) = {
    val f = new BufferedWriter(new FileWriter(new File(dir + "docs_topics.csv")))
    f.write("doc, title, crb\n")
    f.write(doc2topics.map {
      case (did, tids, ws) => {
        val s = docs(did.toInt)
        val areas = area(s)
        s + ", " + areas.title.replaceAll(",", " ") + ", " + areas.mainArea(1) + ", " + tids.zip(ws).map { case (t, w) => t + ":" + w }.mkString(", ")
      }
    }.mkString("\n"))
    f.close()
  }

  def writeAll() = {
    val topDocsPerTopic = distmodel.topDocumentsPerTopic(1000)
    writeTopicInfo(topDocsPerTopic)

    val docs2topics = distmodel.topTopicsPerDocument(30).collect()
    writeDocInfo(docs2topics)
  }

}
