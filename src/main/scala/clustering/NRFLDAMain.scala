package clustering


import basic._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.clustering.DistributedLDAModel
import org.apache.spark.{SparkConf, SparkContext}
/**
  * Created by chanjinpark on 2016. 6. 17..
  */



object NRFLDAMain extends PreProcessing {

  val workspace ="/Users/chanjinpark/data/NRFdata/"
  //val metafile = Array("NRF2015Meta.csv").map(workspace + _)
  //val contdir = Array("content2015-sample/").map(workspace+ _)
  val metafile = Array("NRF2013Meta.csv", "NRF2014Meta.csv", "NRF2015Meta.csv").map(x => workspace + x)
  val contdir = Array("content2013/", "content2014/", "content2015/").map(workspace + _)

  def docpath(id: String): String = {
    val idx = id.substring(0, 4).toInt - 2013
    if ( idx >= 0 && idx < contdir.size) contdir(idx) + id + ".txt"
    else contdir(0) + id + ".txt"
  }

  def loadModel = {
    val conf = new SparkConf(true).setMaster("local").setAppName("NSFLDA")
    val sc = new SparkContext(conf)
    Logger.getLogger("org").setLevel(Level.ERROR)

    val model = DistributedLDAModel.load(sc, "lda/NRFLDAModel")
  }


  def main(args: Array[String]): Unit = {
    val conf = new SparkConf(true).setMaster("local").setAppName("NSFLDA")
    val sc = new SparkContext(conf)
    Logger.getLogger("org").setLevel(Level.ERROR)

    /*
    val meta = getMetaData(sc, metafile).collect().toMap
    val (docs, corpus) = getCorpus(sc, contdir, meta)
    */

    //TODO: Processing Missing Meta Data

    val (docs, corpus, meta) = NRFData.load(sc)
    val (vocab, matrix) = getMatrix(corpus)
    val docsid = docs.zipWithIndex().map(_.swap)
    //val (vocab, matrix) = getMatrixTFIDF(corpus)

    import org.apache.spark.mllib.clustering.LDA
    // Set LDA parameters
    val numTopics = 20
    val numTerms = 20
    docs.cache()
    matrix.cache()

    val lda = new LDA().setK(numTopics).setMaxIterations(100)
    val ldaModel = lda.run(matrix)
    val id2vocab = vocab.map(_.swap)

    // ldaModel.save(sc, "data/lda/NRFLDAModel")

    val ldahtml = new LDAVizHTML(ldaModel, id2vocab, docs.collect(), meta, numTerms, docpath, "ldaviz2020/")

    ldahtml.generatePages()


    val ldares = new LDAResults(ldaModel, id2vocab, docsid.sortBy(_._1).map(_._2).collect(), meta, numTerms, docpath, "ldaviz2020/")
    ldares.writeAll()

    val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = numTerms)

    topicIndices.foreach { case (terms, termWeights) =>
      println("TOPIC:")
      terms.zip(termWeights).foreach {
        case (term, weight) =>
          println(s"${id2vocab(term)}\t$weight")
      }
      println()
    }

    println(vocab.size)
    println(vocab.keys.mkString(", "))

    //if (! new java.io.File("lda/NRFLDAModel").exists()) ldaModel.save(sc, "lda/NRFLDAModel")

    sc.stop()
  }
}

object NRFLDAByYear extends PreProcessing {

  val workspace ="/Users/chanjinpark/data/NRFdata/"
  //val metafile = Array("NRF2015Meta.csv").map(workspace + _)
  //val contdir = Array("content2015-sample/").map(workspace+ _)
  val metafile = Array("NRF2013Meta.csv", "NRF2014Meta.csv", "NRF2015Meta.csv").map(x => workspace + x)
  val contdir = Array("content2013/", "content2014/", "content2015/").map(workspace + _)

  def docpath(id: String): String = {
    val idx = id.substring(0, 4).toInt - 2013
    if ( idx >= 0 && idx < contdir.size) contdir(idx) + id + ".txt"
    else contdir(0) + id + ".txt"
  }


  def main(args: Array[String]): Unit = {
    val conf = new SparkConf(true).setMaster("local").setAppName("NSFLDA")
    val sc = new SparkContext(conf)
    Logger.getLogger("org").setLevel(Level.ERROR)


    val (docsall, corpusall, meta) = NRFData.load(sc)

    import org.apache.spark.mllib.clustering.LDA

    val (numTopics, numTerms) = (20, 20)
    docsall.cache()

    val years = Array("2013", "2014", "2015")
    for (year <- years) {
      val docsid = docsall.zipWithIndex().filter(d => d._1.startsWith(year)).map(_.swap)
      val corpus = corpusall.zipWithIndex().map(_.swap).join(docsid).map(x => (x._1, x._2._1)).sortBy(_._1).map(_._2)

      val (vocab, matrix) = getMatrix(corpus)
      //val (vocab, matrix) = getMatrixTFIDF(corpus)
      matrix.cache()

      val lda = new LDA().setK(numTopics).setMaxIterations(100)
      val ldaModel = lda.run(matrix)
      val id2vocab = vocab.map(_.swap)

      // ldaModel.save(sc, "data/lda/NRFLDAModel")

      val ldahtml = new LDAVizHTML(ldaModel, id2vocab, docsid.sortBy(_._1).map(_._2).collect(), meta, numTerms, docpath, "ldaviz" + year +"/")
      ldahtml.generatePages()

      val ldares = new LDAResults(ldaModel, id2vocab, docsid.sortBy(_._1).map(_._2).collect(), meta, numTerms, docpath, "ldaviz" + year +"/")
      ldares.writeAll()

      val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = numTerms)

      topicIndices.foreach { case (terms, termWeights) =>
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
    //if (! new java.io.File("lda/NRFLDAModel").exists()) ldaModel.save(sc, "lda/NRFLDAModel")

    sc.stop()

  }

}